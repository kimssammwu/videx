# VIDEX 노트북 독립 실행

웹앱, 브라우저, 휴대폰 없이 **Python 프로그램 하나**에서 카메라·모드·인식·화면·노트북 음성을 실행합니다.
기본 실행은 실제 인식입니다. 글씨와 물체 인식에는 인터넷 및 `VIDEX_VISION_API_KEY`가 필요하고,
장애물 감지는 PC에서 Flood Fill만 실행합니다.

## 설치 및 실행

Windows Python 3.10 이상이 필요합니다. Python 설치 시 Tcl/Tk 항목을 포함하세요.
저장소 루트에서 최초 한 번 설치합니다. OpenCV GUI 버전과 headless 버전이 섞이지 않도록 별도 환경을 권장합니다.

```powershell
cd C:\videx_app\videx
python -m venv .venv-desktop
.\.venv-desktop\Scripts\python.exe -m pip install -r proxy\requirements-desktop.txt
```

이후 노트북 웹캠으로 실행:

```powershell
$env:VIDEX_VISION_API_KEY = "본인의 API 키"
.\.venv-desktop\Scripts\python.exe run_videx.py
```

ESP32 안경 카메라로 실행:

```powershell
.\.venv-desktop\Scripts\python.exe run_videx.py --list-ports
.\.venv-desktop\Scripts\python.exe run_videx.py --port COM5
```

환경을 활성화했다면 `python run_videx.py`만 실행하면 됩니다.
`python proxy/viewer.py` 또는 `python -m proxy.viewer`도 같은 통합 시스템을 실행합니다.
창 위쪽에서 웹캠 번호나 COM 포트를 선택하고 **연결 / 재연결**을 누를 수도 있습니다.
같은 COM 포트를 사용하는 기존 뷰어/CLI는 종료하세요.

## 모드와 조작

| 키 / 모드 | 처리 | ESP32 요청 해상도 |
| --- | --- | --- |
| `0` 보행 감지 | LEFT → 기존 `flood_fill.Detector` → 연속 감지·경고 | 명령 0: 320×240 |
| `1` 글씨 읽기 | 촬영한 JPEG 한 장 → 글씨 Vision API → 결과·음성 | 명령 1: LEFT 640×480 |
| `2` 물체 인식 | 앞면·뒷면 촬영 → JPEG 병합 → 물체 Vision API → 결과·음성 | 명령 1: LEFT 640×480 |

**앱 모드 번호와 카메라 해상도 명령은 별개입니다.** 물체 모드 2에서도 최신 물체 인식 모듈과 동일하게
해상도 명령 1을 사용합니다. LEFT는 기본 90도 시계 방향 회전, 노트북 웹캠은 회전하지 않습니다.
필요하면 `--rotation none`, `cw90`, `ccw90`, `180`으로 지정하세요.

- `Space` / 촬영 버튼: 글씨 촬영 또는 물체 앞·뒷면 수동 확정. 물체 앞면을 확정한 뒤에는
  뒤집힘·안정화 검사를 계속 수행하여 뒷면을 자동 선택할 수도 있습니다. 같은 면은 중복으로 거부합니다.
- `R`: 현재 모드를 유지하며 앞에서부터 다시 시작합니다.
- `V`: 현재 결과 또는 안내를 다시 읽습니다.
- `Q`, `Esc`, 창 닫기: 종료합니다.
- ESP32 버튼: 짧게 눌렀다 떼면 촬영/확정, 길게 눌렀다 떼면 다음 모드입니다.
  S3가 마지막 판정을 계속 전송하므로 **눌림→해제 한 쌍당 한 번** 처리합니다. 재접속 때 과거 판정은 재생하지 않습니다.
  USB로 눌림 상태 자체가 누락된 입력은 처리할 수 없으며 키보드로 대체할 수 있습니다.

인식 완료·실패 후에도 모드는 유지됩니다. 다시 Space를 누르면 같은 모드에서 새 인식을 시작합니다.
종료할 때 마지막 모드를 저장하고 다음 실행에서 복원합니다. `--mode 0` 등으로 시작 모드를 지정할 수도 있습니다.

## 화면과 음성

Tkinter 창에 현재 영상, 선택 모드, 처리 상태, 한글 결과와 오류를 표시합니다.
기본 음성 출력은 `pyttsx3`를 통한 노트북 스피커이며 Windows에 한국어 음성이 있으면 우선 선택합니다.
`--mute`는 음성을 끕니다. 이 노트북 실행에는 서버나 브라우저가 필요 없습니다.

화면 수신·Flood Fill·Vision API 작업을 분리하여 API 대기 중에도 영상과 모드 전환이 동작합니다.
처리 중 추가 API 요청은 거부합니다. 모드를 바꾸면 이전 작업의 결과·음성은 무시하며,
이미 전송된 HTTP 요청 자체는 취소할 수 없어 해당 요청이 끝난 뒤 다음 API 요청을 시작합니다.
이때도 보행 감지는 별도 작업으로 계속 실행할 수 있습니다.

USB가 끊기면 재접속하고 현재 해상도를 다시 요청합니다. 현재 모드에 맞는 새 해상도 프레임만 사용하며,
오래된 영상은 인식에 넣지 않습니다. 재접속 후 물체의 진행 중 촬영 세션은 초기화합니다.
현재 구현의 Flood Fill은 장애물 후보를 검출하는 휴리스틱이며 거리 측정이나 보행 안전 판정을 제공하지 않습니다.

## 카메라와 API 없이 확인

```powershell
python run_videx.py --demo
python run_videx.py --demo --headless --mute --mode 0 --seconds 5
```

데모에서는 합성 앞·뒷면 영상과 **명시적인 모의 API 응답**을 사용합니다. 화면에 모의 인식임을 표시합니다.
실제 카메라 촬영 흐름만 확인하려면 `--mock`을 붙일 수 있습니다.
API 키가 없거나 통신에 실패했다고 실제 모드가 모의 응답으로 자동 전환되지는 않습니다.

`--model` 또는 `VIDEX_VISION_MODEL`, `VIDEX_VISION_ENDPOINT`로 기존 API 설정을 바꿀 수 있습니다.
키는 프로세스 환경변수로만 사용하며 설정·결과 파일에 저장하지 않습니다.

## 결과와 구성

기본 `desktop-output/`에 OCR 촬영 JPEG, 물체 세션별 앞면·뒷면·병합 JPEG,
마지막 실행 결과 `session.json`, 마지막 모드 `settings.json`을 저장합니다.
`--output-dir`로 위치를 바꿀 수 있으며 기본 출력 폴더는 Git에서 제외됩니다.

- `run_videx.py` / `proxy/viewer.py`: 실행 진입점, 공유 USB 파서와 수신기
- `proxy/desktop.py`: PC 창과 실행 옵션
- `proxy/runtime.py`: 카메라 소유권, 지속 모드, 처리 작업, 결과 연결
- `proxy/local_speech.py`: PC 자체 음성 출력
- 기존 `proxy/flood_fill.py`, `app/text_recognition/vision.py`, `app/object_recognition/recognizer.py`: 실제 처리 모듈

## GitHub 기준과 검증

2026-10-09 확인한 본 저장소 `kimssammwu/videx`의 `main` 기준은
[`bce3a82`](https://github.com/kimssammwu/videx/commit/bce3a82a050952646d4aad83aa14c160b4d7cf77)입니다.
해당 main에 포함된 물체 촬영 안정화, 글씨 API, Flood Fill 모듈을 그대로 재사용합니다.
별도로 개발 중인 스마트폰 TTS 및 스테레오 작업은 이 통합 변경에 포함하지 않습니다.

```powershell
python -m unittest proxy.test_runtime proxy.test_desktop -v
python -m unittest discover -s proxy -p "test_flood_fill*.py" -v
cd app
python -m unittest discover -s tests -v
python -m unittest discover -s text_recognition\tests -v
```

반복 인식, 실제 앞·뒷면 병합, API 실패 후 재시도, 느린 API 중 모드 전환,
이전 세션 결과 무시, USB 버튼 중복 방지, stale 영상 거부, 재연결 시 모드 유지 등을 자동 검증합니다.
실제 ESP32 수신 및 유료 API 호출은 장치·인증을 갖춘 현장에서 별도로 확인해야 합니다.
