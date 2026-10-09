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

최신 `proxy/viewer.py`와 동일한 28바이트 CAM2 헤더
`<IBBBBIIHHQ>`를 사용한다. 추가 필드는 `flags`, `button_result`, `reserved`이며, 이 세 바이트가
모두 0인 기존 `<IB3xIIHHQ>` 일반 이미지 프레임도 그대로 호환된다. LEFT 버튼 메타데이터는
보존하지만 상품 인식에는 전달하지 않고, RIGHT `STREAM_PAUSED` 메시지는 JPEG로 디코딩하지 않는다.

USB 포트를 열거나 다시 열 때마다 Viewer와 같은 한 바이트 명령 `0x01`(LEFT 640x480)을 전송한다. 이
프로토콜에는 ACK가 없으므로 성공적으로 썼더라도 펌웨어 지원 여부는 확정할 수 없다. 쓰기에
실패하면 경고를 남기고 구형 펌웨어 수신을 계속한다. ESP32의 `timestamp_us`는 수신 로그에
보존하고, 안정화 시간은 PC의 monotonic 수신 시각으로만 계산한다. 선택하지 않은 RIGHT 프레임은
JPEG 디코딩 전에 제외한다. 기본 처리 상한은 10 FPS이다.

포트 확인과 Mock 실행:

```bat
python -m serial.tools.list_ports -v
python -m object_recognition --usb-port COM3 --camera-id 0 --preview --verbose
```

실제 OpenAI API 실행:

```bat
set /P "VIDEX_VISION_API_KEY=OpenAI API key: "
python -m object_recognition --usb-port COM3 --camera-id 0 --preview --live --model gpt-4.1-mini --verbose
set "VIDEX_VISION_API_KEY="
```

상품 인식 해상도는 모드 1(640x480)로 고정된다. 실행 중 `0`/`1`/`2` 키는 해상도를 변경하지
않는다. `C`는 LEFT 현재 프레임 수동 확정, `R`은 앞면부터 재시도, `Q`는 종료이다.
자동 촬영 timeout은 기본 한 번 재시도하며 `--usb-retries`로 변경할 수 있다. Ctrl+C와 `Q` 종료 시
시리얼 포트를 닫는다. 다른 시리얼 모니터가 같은 native USB 포트를 사용 중이면 먼저 종료해야 한다.

PowerShell은 프로그램에 wildcard를 자동 확장하지 않으므로 `--images`에는 실제 파일 경로 목록을
전달해야 한다. 테스트는 다음 명령으로 실행한다.

```powershell
python -m unittest discover -s tests -v
```

출력은 기본적으로 `app/test_output/`에 `*_front.jpg`, `*_back.jpg`,
`*_merged.jpg`로 저장된다. `--output-dir`로 변경할 수 있다.

## LEFT 카메라 전용 미리보기 및 안정성 판정

USB 상품 인식은 LEFT(`camera_id=0`)만 사용한다. CAM2 헤더에서 RIGHT(`camera_id=1`)를
확인하면 JPEG를 디코딩하지 않고 건너뛴다. 두 카메라를 비교하거나 자동 전환하지 않는다.
`--dual-camera`는 호환을 위해 입력된 경우에도 LEFT 전용 `--preview` 별칭으로만 처리한다.

Windows CMD에서 패키지를 설치하고 포트를 확인한다.

```bat
cd /d C:\Users\chaeeun\Desktop\videx\app
python -m pip install -r requirements.txt
python -m serial.tools.list_ports -v
```

Mock 모드는 API 키나 네트워크 요청 없이 전체 촬영·병합 흐름을 확인한다.

```bat
python -m object_recognition --usb-port COM3 --camera-id 0 --preview --mock-product "테스트 제품 1L" --verbose
```

실제 API는 환경변수를 입력하고 `--live`를 명시했을 때만 호출한다.

```bat
set /P "VIDEX_VISION_API_KEY=OpenAI API key: "
python -m object_recognition --usb-port COM3 --camera-id 0 --preview --live --model gpt-4.1-mini --verbose
set "VIDEX_VISION_API_KEY="
```

기본 분석 영역은 전체 프레임(100%)이다. 따라서 제품이 중앙에서 벗어나도 움직임·선명도·밝기
검사에 포함되며, 중앙 ROI로 오해할 수 있는 노란 사각형은 표시하지 않는다. 화면에는 LEFT 영상,
USB 연결 상태, 실제 camera/frame ID와 해상도, STALE/STREAM_PAUSED, 자동 촬영 상태, 움직임,
선명도, 밝기, 안정 비율과 진행률, 실패 이유, API 처리 상태, 앞면·뒷면·병합 이미지와 제품명을
표시한다. `C`는 LEFT 현재 프레임을 수동 확정하고, `R`은 앞면부터 재시도하며, `Q` 또는
`Esc`는 종료한다. 인식 완료 후에도 결과 창은 종료 키를 누를 때까지 유지된다.

GUI에는 `Resolution: 640x480 (Mode 1)`과 실제 LEFT 프레임 해상도를 별도로 표시한다. 실제
해상도가 640x480과 계속 다르면 GUI 상태와 경고 로그로 펌웨어가 요청을 적용하지 않았음을 알린다.
상품 인식 실행 중에는 다른 해상도 명령을 보내지 않는다.

전체 프레임 분석은 주변 배경 움직임도 포함한다. 이를 완화하기 위해 기존 시간 기반 판정을 그대로
유지한다. 기본 움직임 임계값은 실제 로그를 반영한 초기 실험값 `0.040`이다. 최근 1초를 실제 경과시간으로
가중하여 80% 이상 안정적일 때만 촬영한다. `0.100` 이상의 큰 움직임 또는 0.35초 이상의 프레임
공백은 즉시 안정 이력과 후보를 초기화한다. 그보다 작은 단일 튐은 해당 프레임을 후보로 쓰지
않지만 기존 후보와 안정 이력 전체를 삭제하지 않는다.

실제 영상에 맞춰 다음 값을 명령행에서 조정할 수 있다.

```bat
python -m object_recognition --usb-port COM3 --camera-id 0 --preview --motion-threshold 0.04 --motion-reset-threshold 0.10 --stable-ratio 0.80 --stability-window 1.0 --max-frame-gap 0.35 --verbose
```

단일 카메라의 픽셀 변화만으로 실제 반대쪽 면인지 의미적으로 완벽하게 보장할 수는 없다.
자동 판정이 어려우면 `C`로 수동 촬영하되 기존 앞면 중복 검사는 유지된다.

## 실제 연결 전 확인할 정보

- 실제 보드 펌웨어가 mode 1 해상도 명령을 읽고 적용하는지(프로토콜 ACK 없음)
- mode 1에서 LEFT가 640x480으로 계속 전송되는지
- 실제 카메라의 FPS와 프레임 간격 및 USB 분리/재연결 동작
- 고정된 조명에서의 안정 motion 분포와 제품을 움직일 때의 motion 분포
- 대회에서 허용된 Vision API와 중계 서버의 요청/응답 규격
- 실제 영상으로 보정할 움직임, 큰 움직임, 회전 차이, 선명도, 밝기 임계값

`HttpVisionClient`는 OpenAI Chat Completions API 호출을 담당하지만 기본 CLI에서는 사용하거나
호출하지 않는다. 키는 코드가 아니라 `VIDEX_VISION_API_KEY` 환경변수에서만 읽는다.
