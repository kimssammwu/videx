# VIDEX 물체 인식 Python 모듈

## 설치

`app` 디렉터리에서 아래 명령을 실행한다.

```powershell
python -m pip install -r requirements.txt
```

## 실행

실제 Vision API를 호출하지 않고 Mock 제품명으로 앞면/뒷면 병합 전체 흐름을 확인한다.

```powershell
python -m object_recognition --manual-pair front.jpg back.jpg --mock-product "테스트 제품 1L" --verbose
```

`--live`가 없으면 항상 Mock을 사용한다. 실제 OpenAI Chat Completions API를 사용할 때만
`VIDEX_VISION_API_KEY`를 환경변수로 설정하고 `--live`를 추가한다. 기본 모델은
`gpt-4.1-mini`이며 `--model`로 변경할 수 있다.

Windows CMD 예시:

```bat
cd /d C:\Users\chaeeun\Desktop\videx\app
set "VIDEX_VISION_API_KEY=YOUR_OPENAI_API_KEY"
python -m object_recognition --manual-pair "C:\images\product_front.jpg" "C:\images\product_back.jpg" --live --model gpt-4.1-mini --verbose
set "VIDEX_VISION_API_KEY="
```

키를 배치 파일, 소스 코드, 로그 또는 Git 저장소에 기록하지 않는다. `--live`는 병합된 JPEG
한 장을 `https://api.openai.com/v1/chat/completions`로 한 번만 전송한다.

시간순 JPEG 프레임 또는 동영상 자동 선별도 지원한다.

```powershell
python -m object_recognition --images frames\*.jpg --fps 10 --verbose
python -m object_recognition --video sample.mp4 --verbose
```

## ESP32-S3 CAM2 USB 실시간 입력

`proxy/viewer.py`와 동일한 CAM2 프로토콜을 사용한다. ESP32의 `timestamp_us`는 수신 로그에
보존하고, 안정화 시간은 PC의 monotonic 수신 시각으로만 계산한다. 선택하지 않은 카메라 프레임은
JPEG 디코딩 전에 제외한다. 기본 처리 상한은 10 FPS이다.

포트 확인과 Mock 실행:

```bat
python -m serial.tools.list_ports -v
python -m object_recognition --usb-port COM5 --camera-id 0 --verbose
```

실제 OpenAI API 실행:

```bat
set /P "VIDEX_VISION_API_KEY=OpenAI API key: "
python -m object_recognition --usb-port COM5 --camera-id 0 --live --model gpt-4.1-mini --verbose
set "VIDEX_VISION_API_KEY="
```

실행 중 `C`는 다음 유효 프레임 수동 확정, `R`은 ERROR 상태 재시도, `Q`는 종료이다.
자동 촬영 timeout은 기본 한 번 재시도하며 `--usb-retries`로 변경할 수 있다. Ctrl+C와 `Q` 종료 시
시리얼 포트를 닫는다. 다른 시리얼 모니터가 같은 native USB 포트를 사용 중이면 먼저 종료해야 한다.

PowerShell은 프로그램에 wildcard를 자동 확장하지 않으므로 `--images`에는 실제 파일 경로 목록을
전달해야 한다. 테스트는 다음 명령으로 실행한다.

```powershell
python -m unittest discover -s tests -v
```

출력은 기본적으로 `app/test_output/`에 `*_front.jpg`, `*_back.jpg`,
`*_merged.jpg`로 저장된다. `--output-dir`로 변경할 수 있다.

## 실제 연결 전 확인할 정보

- ESP32 프레임 전송 방식(HTTP/WebSocket/USB 등)과 JPEG 메시지 경계
- 카메라 ID, 프레임 ID, 촬영 타임스탬프 필드 형식
- 실제 카메라의 FPS, 해상도, 회전 방향
- 대회에서 허용된 Vision API와 중계 서버의 요청/응답 규격
- 실제 영상으로 보정할 안정성, 회전 차이, 선명도, 밝기 임계값

`HttpVisionClient`는 OpenAI Chat Completions API 호출을 담당하지만 기본 CLI에서는 사용하거나
호출하지 않는다. 키는 코드가 아니라 `VIDEX_VISION_API_KEY` 환경변수에서만 읽는다.
