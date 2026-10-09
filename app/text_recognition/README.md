# VIDEX 범용 글씨 인식

이 패키지는 기존 물품 인식과 독립된 Mode 1용 글씨 인식 기능이다. CAM2 USB의
LEFT(`camera_id=0`)만 사용하며, 촬영 한 번마다 JPEG 한 장과 Vision API 요청 한 번만 처리한다.
정상 인식 결과는 항상 `{"text": "..."}` 형식이다. TTS, 번역, 요약, 제품 식별 기능은 포함하지
않는다.

## 설치 및 환경변수

프로젝트 루트에서:

```powershell
python -m pip install -r app\requirements.txt
$env:VIDEX_LEFT_CAMERA_PORT = "COM3"
$env:VIDEX_VISION_API_KEY = "YOUR_OPENAI_API_KEY"
```

선택 환경변수:

- `VIDEX_VISION_MODEL`: 기본값 `gpt-4.1-mini`
- `VIDEX_VISION_ENDPOINT`: 기본값 `https://api.openai.com/v1/chat/completions`
- `VIDEX_CAMERA_TIMEOUT`: 단일 촬영 대기 시간, 기본값 `10`초
- `VIDEX_VISION_TIMEOUT`: API 대기 시간, 기본값 `30`초

API 키는 소스 코드, 로그 또는 저장소에 기록하지 않는다.

## 실시간 Preview

```powershell
python -m app.text_recognition.cli preview
```

포트를 명령행에서 지정할 수도 있다.

```powershell
python -m app.text_recognition.cli preview --usb-port COM3 --verbose
```

키보드 조작:

- `Space`: 화면의 최신 LEFT 프레임을 JPEG로 저장하고, 저장된 것과 동일한 바이트를 API에 전송
- `S`: 최신 LEFT 프레임만 저장하고 API는 호출하지 않음
- `Q` 또는 `Esc`: Preview, 카메라 연결 및 OpenCV 창 정리 후 종료

Vision 요청은 작업 스레드에서 실행된다. 처리 중 추가 Space 입력은 저장/API 요청 없이 거부된다.
API 오류는 stderr에 표시되며 Preview는 계속 실행된다. 인식 완료 결과만 stdout에 JSON으로
출력된다.

## 단일 촬영 및 인식 명령

API를 호출하지 않고 LEFT JPEG 한 장을 저장한다.

```powershell
python -m app.text_recognition.cli capture --usb-port COM3
```

LEFT JPEG 한 장을 촬영하고 글씨를 인식한다. stdout에는 JSON 객체 하나만 출력된다.

```powershell
python -m app.text_recognition.cli recognize --usb-port COM3
```

카메라 없이 저장된 JPEG로 API만 확인한다.

```powershell
python -m app.text_recognition.cli recognize-image --image .\sample.jpg
```

네트워크 없는 CLI 형식 점검에는 `--mock-text`를 사용할 수 있다.

```powershell
python -m app.text_recognition.cli recognize-image --image .\sample.jpg --mock-text "서울책방"
```

`app` 디렉터리에서 `python -m text_recognition ...` 형식으로도 같은 명령을 실행할 수 있다.

## 촬영 이미지

기본 저장 폴더는 다음과 같다.

```text
app/text_recognition/captures/
```

파일명은 `text_YYYYMMDD_HHMMSS_microseconds.jpg` 형식이며 기존 파일을 덮어쓰지 않는다.
Preview 저장 시 경로, 해상도, 바이트 크기를 stderr에 표시한다. `capture` 명령은 같은 정보를
stdout JSON으로 반환한다. `--output-dir`과 `--jpeg-quality`로 저장 설정을 변경할 수 있다. 생성된
JPEG는 Git 추적에서 제외된다.

## 외부 Mode 1 호출

버튼 또는 Android 연동 계층은 기존 모드 제어 코드를 변경하지 않고 다음 함수를 호출한다.

```python
from app.text_recognition import recognize_text

result = recognize_text()
# {"text": "출입구는 오른쪽에 있습니다."}
```

동시 호출은 `RecognitionBusyError`로 거부된다. 카메라·인증·timeout·API·JSON 오류 역시 예외로
전달되어 정상적인 `{"text": ""}` 결과와 구분된다.

## 테스트

mock 기반 신규 테스트:

```powershell
cd app
python -m unittest discover -s text_recognition\tests -v
```

기존 기능 회귀 테스트:

```powershell
python -m unittest discover -s tests -v
```

실제 하드웨어 수동 확인은 Preview를 실행한 뒤 책, 간판 사진, 안내문, 팸플릿 또는 여러 줄 문서를
화면에 맞추고 Space를 누른다. `captures/`의 JPEG와 stdout JSON을 직접 비교한다. 자동 테스트는
실제 ESP32-CAM 연결이나 실제 API 계정 권한까지 보장하지 않는다.

## 카메라 공유 주의사항

Preview는 기존 `object_recognition.usb_source.UsbCameraSource`의 CAM2 파서, Mode 1 명령 및
LEFT 필터를 읽기 전용으로 재사용한다. Windows COM 포트는 보통 한 프로세스만 열 수 있으므로
물품 인식 Preview나 `proxy/viewer.py`가 같은 포트를 사용 중이면 먼저 종료해야 한다. 두 기능을 한
프로세스에 통합할 경우 상위 실행기가 카메라 소유권을 직렬화해야 한다.
