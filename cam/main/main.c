#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "esp_camera.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "lwip/tcp.h"

#define CAMERA_LEFT  0
#define CAMERA_RIGHT 1
#define CAMERA_ID CAMERA_LEFT /* Change only this to CAMERA_RIGHT for RIGHT. */
// #define CAMERA_ID CAMERA_LEFT /* Change only this to CAMERA_RIGHT for RIGHT. */

#define WIFI_SSID "CAM_RECEIVER"
#define WIFI_PASSWORD "12345678"
#define SERVER_IP "192.168.4.1"
#define SERVER_PORT 5000
#define FRAME_MAGIC 0x324D4143UL
#define MAX_JPEG_SIZE (1024U * 1024U)
#define WIFI_READY BIT0
#define RETRY_MS 1000
#define CONNECT_TIMEOUT_MS 5000
#define SEND_TIMEOUT_MS 3000
#define FRAME_SEND_TIMEOUT_US INT64_C(5000000)
#define FPS_INTERVAL_US INT64_C(5000000)

#define CAM_PIN_PWDN 32
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK 0
#define CAM_PIN_SIOD 26
#define CAM_PIN_SIOC 27
#define CAM_PIN_D7 35
#define CAM_PIN_D6 34
#define CAM_PIN_D5 39
#define CAM_PIN_D4 36
#define CAM_PIN_D3 21
#define CAM_PIN_D2 19
#define CAM_PIN_D1 18
#define CAM_PIN_D0 5
#define CAM_PIN_VSYNC 25
#define CAM_PIN_HREF 23
#define CAM_PIN_PCLK 22

#if CAMERA_ID != CAMERA_LEFT && CAMERA_ID != CAMERA_RIGHT
#error "CAMERA_ID must be CAMERA_LEFT or CAMERA_RIGHT"
#endif
#if !CONFIG_IDF_TARGET_ESP32
#error "This sender requires the classic ESP32 AI-Thinker ESP32-CAM"
#endif
#if CONFIG_FREERTOS_UNICORE
#error "Enable both cores: capture uses Core 1, TCP sender uses Core 0"
#endif

/* ESP32 is little endian; this layout matches Python <IB3xIIHHQ. */
typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t camera_id;
    uint8_t reserved[3];
    uint32_t frame_id;
    uint32_t jpeg_size;
    uint16_t width;
    uint16_t height;
    uint64_t timestamp_us;
} frame_header_t;
_Static_assert(sizeof(frame_header_t) == 28, "Frame header must be 28 bytes");

typedef struct {
    camera_fb_t *fb;
    uint32_t frame_id;
    uint64_t timestamp_us;
} frame_item_t;

typedef struct {
    int64_t start_us;
    uint32_t count;
} fps_counter_t;

static const char *TAG = "cam";
static QueueHandle_t frame_queue;
static EventGroupHandle_t wifi_events;

static const char *camera_name(void)
{
    return CAMERA_ID == CAMERA_LEFT ? "LEFT" : "RIGHT";
}

static void report_fps(fps_counter_t *counter, const char *kind)
{
    ++counter->count;
    int64_t now = esp_timer_get_time();
    int64_t elapsed = now - counter->start_us;
    if (elapsed >= FPS_INTERVAL_US) {
        ESP_LOGI(TAG, "[%s] %s fps=%.1f", camera_name(), kind,
                 counter->count * 1000000.0 / elapsed);
        counter->count = 0;
        counter->start_us = now;
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    (void)arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        ESP_ERROR_CHECK(esp_wifi_connect());
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(wifi_events, WIFI_READY);
        const wifi_event_sta_disconnected_t *event = data;
        ESP_LOGW(TAG, "[%s] Wi-Fi disconnected, reason=%u",
                 camera_name(), (unsigned)event->reason);
        /* The sender retries once per second, outside the event loop. */
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *event = data;
        xEventGroupSetBits(wifi_events, WIFI_READY);
        ESP_LOGI(TAG, "[%s] Wi-Fi ready, IP=" IPSTR,
                 camera_name(), IP2STR(&event->ip_info.ip));
    } else if (base == IP_EVENT && id == IP_EVENT_STA_LOST_IP) {
        xEventGroupClearBits(wifi_events, WIFI_READY);
    }
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    if (esp_netif_create_default_wifi_sta() == NULL) {
        ESP_LOGE(TAG, "[%s] Cannot create STA interface", camera_name());
        abort();
    }
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                             wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, ESP_EVENT_ANY_ID,
                                             wifi_event_handler, NULL));
    wifi_config_t config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
            .channel = 6,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {.capable = true, .required = false},
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
}

static esp_err_t camera_init(void)
{
    bool use_psram = false;
    ESP_LOGI(TAG, "[%s] Free PSRAM: %zu bytes", camera_name(),
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#if CONFIG_SPIRAM
    use_psram = esp_psram_is_initialized();
#endif
    /* Without PSRAM, use a small JPEG and one internal RAM framebuffer.
     * The sender must return that buffer before the next capture can finish. */
    ESP_LOGI(TAG, "[%s] Camera mode: %s", camera_name(),
             use_psram ? "PSRAM, VGA 640x480, quality=12, buffers=2" :
                         "DRAM, QVGA 320x240, quality=20, buffers=1");
    ESP_LOGI(TAG, "[%s] Free internal RAM: %zu bytes, largest block: %zu bytes",
             camera_name(), heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
             heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    const camera_config_t config = {
        .pin_pwdn = CAM_PIN_PWDN,
        .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = use_psram ? FRAMESIZE_VGA : FRAMESIZE_QVGA,
        .jpeg_quality = use_psram ? 12 : 20,
        .fb_count = use_psram ? 2 : 1,
        .fb_location = use_psram ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM,
        .grab_mode = use_psram ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY,
    };
    return esp_camera_init(&config);
}

/* Only the producer discards queued frames. A dequeued framebuffer belongs
 * exclusively to that task until it is returned to the camera driver. */
static void discard_queued_frame(void)
{
    frame_item_t old;
    if (xQueueReceive(frame_queue, &old, 0) == pdTRUE) {
        esp_camera_fb_return(old.fb);
    }
}

static void camera_task(void *arg)
{
    (void)arg;
    uint32_t frame_id = 0;
    fps_counter_t fps = {.start_us = esp_timer_get_time()};
    int64_t last_error_us = -FPS_INTERVAL_US;
    for (;;) {
        /* Release a queued buffer BEFORE fb_get. With two buffers this keeps
         * capture running during TX. With one, capture waits for TX to return
         * its buffer; no frame data is copied or modified during sending. */
        discard_queued_frame();
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb == NULL) {
            int64_t now = esp_timer_get_time();
            if (now - last_error_us >= FPS_INTERVAL_US) {
                ESP_LOGW(TAG, "[%s] Camera capture failed", camera_name());
                last_error_us = now;
            }
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        frame_item_t item = {
            .fb = fb,
            .frame_id = frame_id++,
            .timestamp_us = (uint64_t)esp_timer_get_time(),
        };
        report_fps(&fps, "CAPTURE");
        if (fb->format != PIXFORMAT_JPEG || fb->len == 0 ||
            fb->len > MAX_JPEG_SIZE || fb->width == 0 || fb->height == 0 ||
            fb->width > 2000 || fb->height > 2000) {
            esp_camera_fb_return(fb);
        } else {
            if (xQueueSend(frame_queue, &item, 0) != pdTRUE) {
                discard_queued_frame();
                if (xQueueSend(frame_queue, &item, 0) != pdTRUE) {
                    esp_camera_fb_return(fb);
                }
            }
        }
        /* Let camera driver/idle tasks run even when capturing quickly. */
        vTaskDelay(1);
    }
}

static bool send_all(int sock, const void *buffer, size_t length)
{
    const uint8_t *cursor = buffer;
    const int64_t deadline = esp_timer_get_time() + FRAME_SEND_TIMEOUT_US;
    while (length > 0) {
        if (!(xEventGroupGetBits(wifi_events) & WIFI_READY) ||
            esp_timer_get_time() >= deadline) {
            return false;
        }
        int sent = send(sock, cursor, length, 0);
        if (sent > 0) {
            cursor += sent;
            length -= (size_t)sent;
        } else if (sent < 0 && errno == EINTR) {
            continue;
        } else {
            return false; /* Includes timeout, peer close and socket errors. */
        }
    }
    return true;
}

static int connect_server(void)
{
    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) {
        return -1;
    }
    const int enabled = 1;
    const struct timeval send_timeout = {
        .tv_sec = SEND_TIMEOUT_MS / 1000,
        .tv_usec = (SEND_TIMEOUT_MS % 1000) * 1000,
    };
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) < 0 ||
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &send_timeout,
                   sizeof(send_timeout)) < 0) {
        close(sock);
        return -1;
    }
    /* A bounded nonblocking connect also handles an AP with no TCP server. */
    int nonblocking = 1;
    if (ioctl(sock, FIONBIO, &nonblocking) < 0) {
        close(sock);
        return -1;
    }
    struct sockaddr_in address = {
        .sin_family = AF_INET,
        .sin_port = htons(SERVER_PORT),
    };
    if (inet_pton(AF_INET, SERVER_IP, &address.sin_addr) != 1) {
        close(sock);
        return -1;
    }
    int result = connect(sock, (struct sockaddr *)&address, sizeof(address));
    if (result < 0) {
        if (errno != EINPROGRESS) {
            close(sock);
            return -1;
        }
        int64_t deadline = esp_timer_get_time() + CONNECT_TIMEOUT_MS * INT64_C(1000);
        do {
            fd_set writable;
            FD_ZERO(&writable);
            FD_SET(sock, &writable);
            struct timeval timeout = {.tv_sec = 0, .tv_usec = 200000};
            result = select(sock + 1, NULL, &writable, NULL, &timeout);
            if (result > 0) {
                int error = 0;
                socklen_t size = sizeof(error);
                if (getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &size) < 0 ||
                    error != 0) {
                    close(sock);
                    return -1;
                }
                break;
            }
            if (result < 0 && errno != EINTR) {
                close(sock);
                return -1;
            }
        } while (esp_timer_get_time() < deadline &&
                 (xEventGroupGetBits(wifi_events) & WIFI_READY));
        if (result <= 0) {
            close(sock);
            return -1;
        }
    }
    nonblocking = 0;
    if (ioctl(sock, FIONBIO, &nonblocking) < 0) {
        close(sock);
        return -1;
    }
    return sock;
}

static void tcp_sender_task(void *arg)
{
    (void)arg;
    int sock = -1;
    fps_counter_t fps = {.start_us = esp_timer_get_time()};
    int64_t last_connect_error_us = -FPS_INTERVAL_US;
    for (;;) {
        if (!(xEventGroupGetBits(wifi_events) & WIFI_READY)) {
            if (sock >= 0) {
                close(sock);
                sock = -1;
            }
            EventBits_t bits = xEventGroupWaitBits(wifi_events, WIFI_READY,
                                                  pdFALSE, pdTRUE,
                                                  pdMS_TO_TICKS(RETRY_MS));
            if (!(bits & WIFI_READY)) {
                /* An attempt already in progress may return an error. */
                (void)esp_wifi_connect();
                continue;
            }
        }
        if (sock < 0) {
            sock = connect_server();
            if (sock < 0) {
                int64_t now = esp_timer_get_time();
                if (now - last_connect_error_us >= FPS_INTERVAL_US) {
                    ESP_LOGW(TAG, "[%s] TCP unavailable; retrying %s:%d",
                             camera_name(), SERVER_IP, SERVER_PORT);
                    last_connect_error_us = now;
                }
                vTaskDelay(pdMS_TO_TICKS(RETRY_MS));
                continue;
            }
            ESP_LOGI(TAG, "[%s] TCP connected", camera_name());
        }
        frame_item_t item;
        if (xQueueReceive(frame_queue, &item, pdMS_TO_TICKS(500)) != pdTRUE) {
            continue;
        }
        const frame_header_t header = {
            .magic = FRAME_MAGIC,
            .camera_id = CAMERA_ID,
            .reserved = {0, 0, 0},
            .frame_id = item.frame_id,
            .jpeg_size = (uint32_t)item.fb->len,
            .width = (uint16_t)item.fb->width,
            .height = (uint16_t)item.fb->height,
            .timestamp_us = item.timestamp_us,
        };
        bool sent = send_all(sock, &header, sizeof(header)) &&
                    send_all(sock, item.fb->buf, item.fb->len);
        esp_camera_fb_return(item.fb); /* Also returned on every send failure. */
        if (sent) {
            report_fps(&fps, "TX");
        } else {
            ESP_LOGW(TAG, "[%s] TCP disconnected; reconnecting", camera_name());
            close(sock); /* Never resume a partially transmitted frame. */
            sock = -1;
            vTaskDelay(pdMS_TO_TICKS(RETRY_MS));
        }
    }
}

void app_main(void)
{
    esp_err_t result = nvs_flash_init();
    if (result == ESP_ERR_NVS_NO_FREE_PAGES ||
        result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        result = nvs_flash_init();
    }
    ESP_ERROR_CHECK(result);
    ESP_ERROR_CHECK(camera_init());
    ESP_LOGI(TAG, "[%s] Camera ready", camera_name());
    frame_queue = xQueueCreate(1, sizeof(frame_item_t));
    wifi_events = xEventGroupCreate();
    if (frame_queue == NULL || wifi_events == NULL) {
        ESP_LOGE(TAG, "[%s] Queue/event allocation failed", camera_name());
        abort();
    }
    wifi_init();
    if (xTaskCreatePinnedToCore(camera_task, "camera", 4096, NULL, 5, NULL, 1)
            != pdPASS ||
        xTaskCreatePinnedToCore(tcp_sender_task, "tcp_sender", 6144, NULL, 5,
                                NULL, 0) != pdPASS) {
        ESP_LOGE(TAG, "[%s] Task creation failed", camera_name());
        abort();
    }
}
