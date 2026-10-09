idf_component_register(SRCS "main.c"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES esp_wifi esp_event esp_netif nvs_flash
                                  esp_psram esp_timer esp_driver_gpio lwip esp32-camera)
