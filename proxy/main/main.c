#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "sdkconfig.h"
#include "button.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "driver/usb_serial_jtag.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "lwip/tcp.h"

/* Matches cam/main/main.c on TCP. USB uses <IBBBBIIHHQ, then JPEG.
 * S3 debounces/classifies LEFT input and fills flags + button_result.
 * Native USB Serial/JTAG exposes CDC ACM on GPIO19 (D-) / GPIO20 (D+).
 * Use the board's native USB connector, not its USB-to-UART connector.
 * PC must resynchronize on CAM2 + a valid header after reopening USB or
 * a transfer timeout. Camera timestamps are local, not synchronized clocks.
 */
#define WIFI_SSID "CAM_RECEIVER2"
#define WIFI_PASSWORD "12345678"
#define WIFI_CHANNEL 6
#define SERVER_PORT 5000
#define CAMERA_COUNT 2
#define FRAME_MAGIC UINT32_C(0x324D4143)
#define FRAME_FLAG_STREAM_PAUSED (1U << 3)
#define MAX_JPEG_SIZE (1024U * 1024U)
#define MAX_FRAME_DIMENSION 4096U
#define FRAME_TIMEOUT_US INT64_C(5000000)
#define HEADER_TIMEOUT_US INT64_C(15000000)
#define STATS_INTERVAL_US INT64_C(5000000)
#define BUTTON_STREAM_GAP_US INT64_C(2000000)

#if !CONFIG_IDF_TARGET_ESP32S3
#error "Build this proxy with idf.py set-target esp32s3"
#endif
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG || CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG || CONFIG_ESP_CONSOLE_USB_CDC
#error "Select UART console and no secondary console: USB carries binary frames"
#endif

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t camera_id;
    uint8_t flags;
    uint8_t button_result; /* TCP: reserved zero; USB: last SHORT/LONG result. */
    uint8_t reserved;
    uint32_t frame_id;
    uint32_t jpeg_size;
    uint16_t width;
    uint16_t height;
    uint64_t timestamp_us;
} frame_header_t;
_Static_assert(sizeof(frame_header_t) == 28, "CAM2 header must be 28 bytes");

typedef struct {
    frame_header_t header;
    uint8_t *jpeg;
} frame_t;

typedef struct {
    unsigned index;
    QueueHandle_t sockets;
} receiver_t;

static const char *TAG = "proxy";
static receiver_t receivers[CAMERA_COUNT];
static QueueHandle_t available_receivers;
static QueueHandle_t output_frames;
static portMUX_TYPE resolution_lock = portMUX_INITIALIZER_UNLOCKED;
static uint8_t requested_resolution = 0;

static uint8_t get_resolution(void)
{
    portENTER_CRITICAL(&resolution_lock);
    uint8_t mode = requested_resolution;
    portEXIT_CRITICAL(&resolution_lock);
    return mode;
}

static void usb_command_task(void *arg)
{
    (void)arg;
    uint8_t commands[32];
    for (;;) {
        int count = usb_serial_jtag_read_bytes(commands, sizeof(commands), pdMS_TO_TICKS(100));
        for (int i = 0; i < count; ++i) {
            uint8_t mode = commands[i];
            if (mode >= '0' && mode <= '2') {
                mode -= '0';
            }
            if (mode > 2) {
                continue;
            }
            portENTER_CRITICAL(&resolution_lock);
            requested_resolution = mode;
            portEXIT_CRITICAL(&resolution_lock);
            ESP_LOGI(TAG, "Resolution requested: mode=%u (0=both 320, 1=LEFT 640, 2=LEFT 1280)",
                     (unsigned)mode);
        }
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    (void)arg;
    (void)base;
    if (id == WIFI_EVENT_AP_STACONNECTED) {
        const wifi_event_ap_staconnected_t *event = data;
        ESP_LOGI(TAG, "Station connected, aid=%u", (unsigned)event->aid);
    } else if (id == WIFI_EVENT_AP_STADISCONNECTED) {
        const wifi_event_ap_stadisconnected_t *event = data;
        ESP_LOGI(TAG, "Station disconnected, aid=%u", (unsigned)event->aid);
    }
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *ap = esp_netif_create_default_wifi_ap();
    if (ap == NULL) {
        abort();
    }
    ESP_ERROR_CHECK(esp_netif_dhcps_stop(ap));
    esp_netif_ip_info_t ip = {0};
    IP4_ADDR(&ip.ip, 192, 168, 4, 1);
    IP4_ADDR(&ip.gw, 192, 168, 4, 1);
    IP4_ADDR(&ip.netmask, 255, 255, 255, 0);
    ESP_ERROR_CHECK(esp_netif_set_ip_info(ap, &ip));
    ESP_ERROR_CHECK(esp_netif_dhcps_start(ap));
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                             wifi_event_handler, NULL));
    const wifi_config_t config = {
        .ap = {
            .ssid = WIFI_SSID,
            .ssid_len = sizeof(WIFI_SSID) - 1,
            .password = WIFI_PASSWORD,
            .channel = WIFI_CHANNEL,
            .max_connection = CAMERA_COUNT,
            .authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {.required = false},
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_LOGI(TAG, "AP %s ready at 192.168.4.1:%d", WIFI_SSID, SERVER_PORT);
}

/* TCP may split a header/payload across arbitrary reads. An absolute deadline
 * also disconnects peers that trickle bytes without ever finishing a frame. */
static bool receive_exact(int sock, void *buffer, size_t length, int64_t deadline)
{
    uint8_t *cursor = buffer;
    while (length > 0 && esp_timer_get_time() < deadline) {
        int count = recv(sock, cursor, length, 0);
        if (count > 0) {
            cursor += count;
            length -= (size_t)count;
        } else if (count < 0 && (errno == EINTR || errno == EAGAIN ||
                                 errno == EWOULDBLOCK)) {
            continue;
        } else {
            return false;
        }
    }
    return length == 0;
}

static bool valid_header(const frame_header_t *h)
{
    if (h->magic == FRAME_MAGIC && h->camera_id == 1 &&
        h->flags == FRAME_FLAG_STREAM_PAUSED && h->button_result == 0 &&
        h->reserved == 0 && h->jpeg_size == 0 && h->width == 0 && h->height == 0) {
        return true; /* RIGHT pause heartbeat: header only, no JPEG allocation. */
    }
    return h->magic == FRAME_MAGIC && h->camera_id < CAMERA_COUNT &&
           h->button_result == 0 && h->reserved == 0 &&
           (h->flags == 0 ||
            (h->camera_id == 0 &&
             (h->flags == FRAME_FLAG_BUTTON_VALID ||
              h->flags == (FRAME_FLAG_BUTTON_VALID | FRAME_FLAG_BUTTON_PRESSED)))) &&
           h->jpeg_size >= 4 && h->jpeg_size <= MAX_JPEG_SIZE &&
           h->width > 0 && h->width <= MAX_FRAME_DIMENSION &&
           h->height > 0 && h->height <= MAX_FRAME_DIMENSION;
}

static void receiver_task(void *arg)
{
    receiver_t *receiver = arg;
    for (;;) {
        int sock;
        xQueueReceive(receiver->sockets, &sock, portMAX_DELAY);
        int camera_id = -1;
        int sent_resolution = -1;
        button_state_t button = {0}; /* Reset for each camera TCP connection. */
        int64_t last_left_at = 0;
        uint32_t received = 0;
        uint32_t dropped = 0;
        int64_t stats_start = esp_timer_get_time();
        for (;;) {
            uint8_t mode = get_resolution();
            if (sent_resolution != mode) {
                /* Opposite TCP direction carries only one-byte controls. */
                if (send(sock, &mode, sizeof(mode), 0) != sizeof(mode)) {
                    break;
                }
                sent_resolution = mode;
            }
            frame_t frame = {0};
            if (!receive_exact(sock, &frame.header, sizeof(frame.header),
                               esp_timer_get_time() + HEADER_TIMEOUT_US)) {
                break;
            }
            if (!valid_header(&frame.header) ||
                (camera_id >= 0 && camera_id != frame.header.camera_id)) {
                ESP_LOGW(TAG, "Receiver %u: invalid header", receiver->index);
                break; /* Reconnect instead of guessing the TCP frame boundary. */
            }
            camera_id = frame.header.camera_id;
            if (frame.header.flags == FRAME_FLAG_STREAM_PAUSED) {
                if (xQueueSend(output_frames, &frame, 0) != pdTRUE) {
                    ++dropped;
                }
                continue;
            }
            frame.jpeg = heap_caps_malloc(frame.header.jpeg_size,
                                          MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (frame.jpeg == NULL) {
                frame.jpeg = heap_caps_malloc(frame.header.jpeg_size, MALLOC_CAP_8BIT);
            }
            if (frame.jpeg == NULL) {
                ESP_LOGW(TAG, "JPEG allocation failed (%" PRIu32 "); enable board PSRAM",
                         frame.header.jpeg_size);
                break;
            }
            if (!receive_exact(sock, frame.jpeg, frame.header.jpeg_size,
                               esp_timer_get_time() + FRAME_TIMEOUT_US)) {
                free(frame.jpeg);
                break;
            }
            const size_t length = frame.header.jpeg_size;
            if (frame.jpeg[0] != 0xff || frame.jpeg[1] != 0xd8 ||
                frame.jpeg[length - 2] != 0xff || frame.jpeg[length - 1] != 0xd9) {
                free(frame.jpeg);
                ESP_LOGW(TAG, "Camera %d: invalid JPEG", camera_id);
                break;
            }
            ++received;
            if (camera_id == 1 && get_resolution() > 0) {
                /* Discard old in-flight RIGHT images while the pause command arrives. */
                free(frame.jpeg);
                ++dropped;
                continue;
            }
            int64_t now = esp_timer_get_time(); /* Also used for existing stats. */
            if (camera_id == 0) {
                if (last_left_at != 0 && now - last_left_at > BUTTON_STREAM_GAP_US) {
                    button_reset(&button);
                }
                last_left_at = now;
                uint8_t event = button_update(&button,
                    (frame.header.flags & FRAME_FLAG_BUTTON_VALID) != 0,
                    (frame.header.flags & FRAME_FLAG_BUTTON_PRESSED) != 0);
                frame.header.flags = button_output_flags(&button);
                frame.header.button_result = button.last_result;
                if (event != BUTTON_RESULT_NONE) {
                    ESP_LOGI(TAG, "LEFT button: %s (frame %" PRIu32 ")",
                             event == BUTTON_RESULT_LONG ? "LONG" : "SHORT",
                             frame.header.frame_id);
                }
            }
            /* Ownership passes to the sole USB writer. A bounded queue avoids
             * unbounded memory growth when the PC stops reading. */
            if (xQueueSend(output_frames, &frame, 0) != pdTRUE) {
                free(frame.jpeg);
                ++dropped;
            }
            if (now - stats_start >= STATS_INTERVAL_US) {
                ESP_LOGI(TAG, "Camera %d: RX %.1f fps, dropped=%" PRIu32,
                         camera_id, received * 1000000.0 / (now - stats_start), dropped);
                received = dropped = 0;
                stats_start = now;
            }
        }
        close(sock);
        ESP_LOGI(TAG, "Receiver %u disconnected (camera %d)", receiver->index, camera_id);
        xQueueSend(available_receivers, &receiver->index, portMAX_DELAY);
    }
}

static bool usb_write_all(const void *buffer, size_t length, int64_t deadline)
{
    const uint8_t *cursor = buffer;
    while (length > 0 && esp_timer_get_time() < deadline) {
        if (!usb_serial_jtag_is_connected()) {
            return false;
        }
        size_t chunk = length > 4096 ? 4096 : length;
        int count = usb_serial_jtag_write_bytes(cursor, chunk, pdMS_TO_TICKS(100));
        if (count < 0) {
            return false;
        }
        cursor += count;
        length -= (size_t)count;
        if (count == 0) {
            vTaskDelay(1);
        }
    }
    return length == 0;
}

static void usb_sender_task(void *arg)
{
    (void)arg;
    int64_t last_warning = -STATS_INTERVAL_US;
    for (;;) {
        frame_t frame;
        xQueueReceive(output_frames, &frame, portMAX_DELAY);
        if (frame.header.camera_id == 1 && frame.header.jpeg_size > 0 &&
            get_resolution() > 0) {
            free(frame.jpeg);
            continue;
        }
        if (usb_serial_jtag_is_connected()) {
            int64_t deadline = esp_timer_get_time() + FRAME_TIMEOUT_US;
            bool sent = usb_write_all(&frame.header, sizeof(frame.header), deadline) &&
                        usb_write_all(frame.jpeg, frame.header.jpeg_size, deadline);
            if (!sent && esp_timer_get_time() - last_warning >= STATS_INTERVAL_US) {
                ESP_LOGW(TAG, "USB transfer interrupted; PC must resync on CAM2 header");
                last_warning = esp_timer_get_time();
            }
        }
        free(frame.jpeg);
    }
}

static void tcp_server_task(void *arg)
{
    (void)arg;
    for (;;) {
        int server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (server < 0) {
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        const int enabled = 1;
        const struct sockaddr_in address = {
            .sin_family = AF_INET,
            .sin_port = htons(SERVER_PORT),
            .sin_addr.s_addr = htonl(INADDR_ANY),
        };
        if (setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) < 0 ||
            bind(server, (const struct sockaddr *)&address, sizeof(address)) < 0 ||
            listen(server, CAMERA_COUNT) < 0) {
            ESP_LOGE(TAG, "TCP listen failed: errno=%d", errno);
            close(server);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        for (;;) {
            int sock = accept(server, NULL, NULL);
            if (sock < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            const struct timeval timeout = {.tv_sec = 1};
            unsigned index;
            if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0 ||
                setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0 ||
                setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) < 0) {
                close(sock);
                continue;
            }
            if (xQueueReceive(available_receivers, &index, 0) != pdTRUE) {
                close(sock);
                continue;
            }
            ESP_LOGI(TAG, "TCP client assigned to receiver %u", index);
            xQueueSend(receivers[index].sockets, &sock, portMAX_DELAY);
        }
        close(server);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void app_main(void)
{
    esp_err_t result = nvs_flash_init();
    if (result == ESP_ERR_NVS_NO_FREE_PAGES || result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        result = nvs_flash_init();
    }
    ESP_ERROR_CHECK(result);
    available_receivers = xQueueCreate(CAMERA_COUNT, sizeof(unsigned));
    output_frames = xQueueCreate(CAMERA_COUNT, sizeof(frame_t));
    if (available_receivers == NULL || output_frames == NULL) {
        abort();
    }
    usb_serial_jtag_driver_config_t usb = {
        .tx_buffer_size = 8192,
        .rx_buffer_size = 256,
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb));
    wifi_init();
    for (unsigned i = 0; i < CAMERA_COUNT; ++i) {
        receivers[i].index = i;
        receivers[i].sockets = xQueueCreate(1, sizeof(int));
        if (receivers[i].sockets == NULL ||
            xTaskCreate(receiver_task, "camera_rx", 4096, &receivers[i], 5, NULL) != pdPASS) {
            abort();
        }
        xQueueSend(available_receivers, &i, portMAX_DELAY);
    }
    if (xTaskCreate(usb_sender_task, "usb_tx", 4096, NULL, 5, NULL) != pdPASS ||
        xTaskCreate(usb_command_task, "usb_commands", 3072, NULL, 5, NULL) != pdPASS ||
        xTaskCreate(tcp_server_task, "tcp_server", 4096, NULL, 4, NULL) != pdPASS) {
        abort();
    }
    ESP_LOGI(TAG, "USB CDC binary output ready; logs use UART0");
}
