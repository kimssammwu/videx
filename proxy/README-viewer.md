# ESP32-S3 USB 이미지 뷰어

`main/main.c`의 native USB CDC 출력(CAM2 헤더 + JPEG)을 수신하여
카메라 0(LEFT), 카메라 1(RIGHT)을 한 창에 표시합니다.
수신/디코딩은 별도 스레드에서 실행하고 화면에는 각 카메라의 최신 프레임을 표시합니다.
2초 이상 새 프레임이나 상태가 없으면 STALE 표시가 나타납니다.
RIGHT가 정지하면 마지막 이미지를 지우고 PAUSED를 표시합니다.
두 카메라의 영상은 시간 동기화된 쌍이 아닙니다.
LEFT 영상에는 접점 흔들림을 걸러낸 버튼 상태를 `BUTTON: PRESSED` 또는
`BUTTON: RELEASED`로 표시하고 마지막 판정은 `S3: SHORT`/`S3: LONG`으로 유지합니다.
이전 펌웨어에서 상태를 보내지 않거나 초기 안정 상태를 확인하기 전에는 UNKNOWN입니다.
STALE 표시가 있으면 버튼 표시도 마지막 수신 프레임의 상태입니다.

## LEFT 버튼 연결

LEFT ESP32-CAM의 GPIO13과 GND 사이에 버튼을 연결합니다.
내부 풀업을 사용하므로 열림은 해제, GND로 연결되면 눌림입니다.
`cam/main/main.c`의 `BUTTON_GPIO`에서 핀을 변경할 수 있습니다.
GPIO13은 microSD와 공유하므로 이 구성에서는 microSD를 사용하지 마세요.
RIGHT는 버튼 핀을 초기화하지 않고 버튼 정보를 보내지 않습니다.
cam, proxy 펌웨어를 각각 다시 빌드하고 플래시한 뒤 업데이트된 viewer.py를 실행하세요.
CAMERA_ID가 CAMERA_LEFT인지 확인하세요.

버튼 입력은 프레임마다 한 번 읽으며 누른 시간이나 길게 누름 여부를 계산하지 않습니다.
ESP32-CAM에서는 디바운싱하지 않으며 S3가 수신 상태를 처리합니다.
프레임 사이의 짧은 입력은 검출하지 못할 수 있습니다. 전송이 누락된 프레임의 버튼 상태도 전달되지 않습니다.

## SHORT / LONG 판정 (ESP32-S3)

ESP32-CAM은 원시 상태만 보내고, S3의 TCP 수신 태스크가 완전히 수신한 LEFT
프레임마다 한 번씩 상태를 처리합니다. Python은 S3가 보낸 상태와 결과만 표시합니다.
USB 프레임이 버려져도 S3 판정의 틱은 진행됩니다. RIGHT 프레임은 틱으로 세지 않습니다.
길이 판정에는 시계 조회나 sleep을 사용하지 않습니다.

- 기본 LONG 기준: 눌림 30틱. S3의 LEFT 수신이 약 30 FPS이면 대략 1초입니다.
- 접점 흔들림: 같은 값이 3프레임 연속 들어와야 눌림/해제를 확정합니다.
- 눌림 확정에 사용한 3틱도 길이에 포함합니다. 잠깐의 해제 후보 동안 카운트는 멈춥니다.
- 30틱 도달 시 LONG 이벤트를 한 번 생성합니다. 그 전에 해제가 확정되면 SHORT입니다.
- LONG 이후 해제에는 SHORT를 생성하지 않습니다. 결과는 S3의 UART 로그에 표시됩니다.
- 마지막 결과를 이후 LEFT 프레임에 계속 보내므로 화면에 `S3: SHORT`/`S3: LONG`이 유지됩니다.
  여러 번의 입력이 USB 전송 중 누락되면 중간 이벤트의 개수까지 보장하지는 않습니다.
- LEFT TCP 재접속, 원시 버튼 상태 UNKNOWN 또는 LEFT 수신 간격 2초 초과 시 판정 상태를
  초기화합니다. 간격 확인은 기존 수신 통계 시각을 재사용하며 누름 길이는 틱만으로 판단합니다.
- USB 재연결만으로는 S3의 판정 상태를 초기화하지 않습니다.

기준 변경은 `main/button.h`의 `BUTTON_LONG_TICKS`(기본 30),
`BUTTON_DEBOUNCE_TICKS`(기본 3)를 수정하고 S3 펌웨어를 다시 빌드/플래시하세요.
실제 S3 LEFT 수신이 15 FPS 정도라면 LONG 기준을 15로 맞출 수 있습니다.
FPS나 프레임 누락에 따라 실제 시간 기준은 달라지며 정확한 1초를 보장하지 않습니다.
확정 틱 수보다 짧은 입력은 잡음으로 무시합니다. DEBOUNCE는 1 이상, LONG 이하이어야 합니다.

버튼 원시 상태 전송 방식은 유지합니다. 해상도 제어 기능까지 사용하려면
아래 안내대로 CAM과 S3를 모두 업데이트하세요.
Python의 `--long-ticks` / `--debounce-ticks` 옵션은 제거했습니다.

## CAM 역할 선택: 변수 하나만 변경

`cam/main/main.c` 상단의 `CAMERA_ID`만 바꾸고 빌드/플래시합니다.

```c
#define CAMERA_ID CAMERA_LEFT   // LEFT 보드
// RIGHT를 빌드할 때 위 줄을 CAMERA_RIGHT로 변경
```

LEFT는 감지된 PSRAM과 버튼(GPIO13)을 사용합니다.
RIGHT는 PSRAM을 사용하지 않고 내부 RAM, JPEG quality=20, 버퍼 1개로 320×240만 송출합니다.
공통 sdkconfig는 PSRAM 자동 탐색/40 MHz/미검출 시 계속 부팅을 설정하므로
역할 변경 시 sdkconfig나 핀 설정을 별도로 바꿀 필요가 없습니다.
현재 소스의 기본 역할은 LEFT입니다.

## 서버에서 해상도 전환

PC → S3 native USB → 카메라의 기존 TCP 연결로 명령을 보냅니다.
S3는 카메라 재접속 시 최근 요청을 다시 보내며 기본 모드는 0입니다.

| 명령 | LEFT | RIGHT |
| --- | --- | --- |
| 0 | 320×240 | 320×240 송출 재개 |
| 1 | 640×480 | 캡처/이미지 송출 정지 |
| 2 | 1280×1024 | 캡처/이미지 송출 정지 |

3 모드는 제거했으며 binary 3 또는 ASCII '3'은 무시합니다.

```bash
python viewer.py --port /dev/ttyACM0 --resolution 0
```

뷰어 창에서 숫자 키 `0`, `1`, `2`로 전환합니다. `--resolution 2`로 시작할 수도 있습니다.
하단 Requested는 요청 크기, 영상 상단의 크기는 실제 수신한 JPEG 크기입니다.
미리보기는 칸에 맞춰 축소되지만 전송 원본의 해상도는 그대로입니다.
USB 재접속 시 현재 선택값을 재전송합니다.

다른 서버 프로그램은 USB로 `bytes([0])`, `bytes([1])`, `bytes([2])`를 쓰면 됩니다.
ASCII `b"0"`, `b"1"`, `b"2"`도 허용합니다. 하나의 프로그램에서 해당 USB 포트를 사용하세요.
S3 → CAM 명령도 같은 1바이트 형식입니다.

RIGHT는 1/2 요청 시 진행 중인 버퍼가 반환된 뒤 카메라 드라이버를 해제합니다.
이미지는 전송하지 않지만 TCP 연결을 유지하며 1초마다 JPEG 없는 28바이트 PAUSED 상태를 보냅니다.
S3는 이 상태를 USB로 전달하므로 뷰어는 RIGHT 영상을 지우고 PAUSED를 표시합니다.
0 요청 시 320×240으로 다시 초기화하여 송출합니다. 정지 중에도 0 명령을 받을 수 있습니다.
전환 도중 이미 전송 중이던 이미지 한 장이나 상태가 잠깐 표시될 수 있습니다.
S3는 1/2 모드에서 남아 있던 RIGHT JPEG를 수신/USB 큐에서 버립니다.

LEFT의 1/2는 정상 작동하는 PSRAM이 필요합니다. 초기화 실패 시 이전 해상도로 복구하고
CAM 로그에 원인을 남깁니다. 이때 RIGHT는 요청에 따라 계속 정지하며 0으로 재개할 수 있습니다.
LEFT JPEG quality는 PSRAM 사용 시 12, PSRAM 미검출 시 모드 0에서 20입니다.
S3도 큰 JPEG를 받을 메모리가 필요합니다. `JPEG allocation failed`가 나면 S3 보드의
실제 PSRAM 유무와 Quad/Octal 방식에 맞춘 설정을 확인하세요.
해상도에 따른 FPS 변화는 S3의 30틱 LONG 기준에 해당하는 실제 시간에도 영향을 줍니다.

CAM(LEFT/RIGHT)과 S3를 다시 빌드/플래시하고 새 뷰어를 사용하세요.

## 설치

Python 3.9 이상, GUI가 있는 데스크톱 환경에서 실행하세요.

```bash
cd /home/wonyj/Desktop/videx/proxy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-viewer.txt
```

필요한 외부 라이브러리:

- `pyserial`: USB 시리얼 포트 수신
- `numpy`: JPEG 바이트와 이미지 배열 처리
- `opencv-python`: JPEG 디코딩과 화면 표시 (`opencv-python-headless`는 창 표시 불가)

## 연결 및 실행

1. ESP32-S3의 **native USB 포트**를 PC에 연결합니다. USB-to-UART 포트는 로그용입니다.
2. 같은 native USB 포트를 사용하는 시리얼 모니터를 종료합니다.
3. 포트를 확인하고 뷰어를 실행합니다.

```bash
python viewer.py --list-ports
python viewer.py --port /dev/ttyACM0
```

Windows에서는 `python viewer.py --port COM5`처럼 해당 포트를 지정합니다.
`Q`, `Esc`, 창 닫기 또는 `Ctrl+C`로 종료합니다.
USB 연결 오류가 발생하면 지정한 포트로 1초 간격으로 재접속합니다.
재연결 후 포트 이름이 바뀌면 새 포트를 지정하여 다시 실행하세요.
Linux에서 Permission denied가 발생하면 해당 장치의 시리얼 접근 권한을 확인하세요.
`--baudrate` 기본값은 115200이며 native USB 전송 속도를 제한하지 않습니다.

## 전송 형식

USB 출력은 28바이트 little-endian 헤더 `struct.Struct("<IBBBBIIHHQ")` 뒤에 JPEG가 옵니다.
필드 순서는 magic(0x324D4143, 바이트로 CAM2), camera_id, flags, button_result,
reserved(0인 1바이트), frame_id, jpeg_size, width, height, timestamp_us입니다.
기존 예약 바이트를 사용하므로 헤더 크기와 이미지 메타데이터 오프셋은 그대로입니다.

- flags bit 0: 눌림. bit 1: 상태 유효. bit 2: S3 판정 상태.
- S3 LEFT 출력: 초기 UNKNOWN=4, 안정 해제=6, 안정 눌림=7.
- button_result: 0=아직 판정 없음, 1=최근 SHORT, 2=최근 LONG.
- RIGHT 이미지 프레임은 flags와 button_result가 모두 0입니다.
- RIGHT PAUSED 상태는 flags bit 3(값 8), button_result=0, jpeg_size=width=height=0입니다.
  28바이트 헤더만 전송하며 JPEG는 없고 timestamp_us는 상태 송신 시각입니다.
- CAM → S3 입력 형식은 기존 `<IBB2xIIHHQ`입니다. LEFT 원시 해제=2, 눌림=3이며
  예약 바이트는 0입니다. 원시 판정과 S3 판정 형식은 입력 검증에서 구별됩니다.
- 새 뷰어는 이전 프록시의 원시 입력도 표시하지만 `RAW (update S3)`를 표시하고 재판정하지 않습니다.
  이전 뷰어는 새 형식을 읽지 못하므로 S3와 뷰어를 함께 업데이트하세요.

JPEG 최대 크기는 1 MiB, 가로/세로 최대값은 각각 4096입니다.
임의 크기로 나뉜 USB 읽기를 처리하고, 잘못된 헤더/JPEG 또는 미완성 프레임의
6초 제한 초과 시 CAM2를 다시 검색합니다. timestamp_us는 카메라별 로컬 시각입니다.

## 검증

```bash
python -m unittest -v test_viewer
cc -std=c11 -Wall -Wextra -Werror test_button.c -o /tmp/test_proxy_button
/tmp/test_proxy_button
```

C 테스트는 S3 펌웨어가 사용하는 `main/button.h`를 직접 검증합니다.
