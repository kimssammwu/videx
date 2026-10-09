# VIDEX Stereo Calibration — 1차 개발

ESP32-S3의 CAM2 스트림에서 수동으로 L/R JPEG 쌍을 수집하고, 저장된 이미지로
개별 카메라 보정 → Stereo Calibration → Stereo Rectification을 수행합니다.
StereoSGBM과 거리 측정은 포함하지 않습니다.

`proxy/viewer.py`의 `FrameParser`와 `Frame` 형식을 그대로 재사용합니다.
기존 뷰어와 ESP32 펌웨어는 수정하지 않습니다. 현재 저장소의 `cam/`에는 촬영 코드가 없고,
`proxy/main/main.c`의 `app_main()`은 비어 있습니다. 실제 송신 펌웨어와 장치 테스트는 미검증입니다.

## 설치 및 실행 위치

Python 3.10 이상, GUI가 있는 데스크톱을 사용합니다. 모든 `python -m app.stereo...`
명령은 **레포지토리 루트 `C:\videx_app\videx`**에서 실행해야 합니다.
`app`과 `proxy`는 Python namespace package로 불러오므로 별도의 `__init__.py` 변경이 필요 없습니다.

PowerShell:

```powershell
Set-Location C:\videx_app\videx
python -m venv app/stereo/.venv
& .\app\stereo\.venv\Scripts\python.exe -m pip install -r app/stereo/requirements.txt
```

아래 예시는 가상환경을 활성화한 뒤의 `python`을 사용합니다.
활성화가 허용되지 않은 환경에서는 `python` 대신 위의 가상환경 실행 파일을 사용하세요.

```powershell
.\app\stereo\.venv\Scripts\Activate.ps1
```

의존성은 `numpy`, GUI 지원 `opencv-python`, `pyserial`입니다.
`opencv-python-headless`만 설치된 환경에서는 오프라인 보정·PNG 생성은 가능하지만 캡처 창은 사용할 수 없습니다.
기존 viewer의 Python 3.9 요구사항과 달리 이 모듈은 Python 3.10 이상의 타입 문법을 사용합니다.

## 체커보드와 촬영 방법

- 설정은 **내부 코너 수**입니다. 예: `--columns 9 --rows 6`은 10×7칸의 체커보드입니다.
- `--square-size 24 --unit mm`는 실제 인쇄된 정사각형 한 변이 24mm라는 뜻입니다.
  예시값을 그대로 사용하지 말고 인쇄 후 실측값을 입력하세요. `mm`, `cm`, `m`를 지원합니다.
- 보드는 평평하고 단단한 판에 부착하고 주변에 밝은 여백을 남깁니다. 양쪽 카메라에 전체 보드가 보여야 합니다.
- 카메라 고정 위치, 초점, 해상도를 유지하고 원본 이미지의 임의 확대·축소·미러링을 피합니다.
  방향 보정은 아래의 공통 회전 설정으로만 적용하고, 같은 세션에서는 설정을 유지하세요.
- 같은 보드 자세에서 좌우 화면을 확인하고, **보드를 멈춘 상태로 양쪽 최신 영상이 갱신될 때까지 기다린 뒤** 캡처합니다.
  좌우 PC 수신 간격은 노출 동기화를 증명하지 않습니다.
- 서로 다른 자세로 15∼25쌍 이상을 권장합니다. 화면 중앙·가장자리, 여러 거리, 가로·세로 기울기를 다양하게 수집하세요.
  거의 정면인 사진만 반복해서 수집하면 낮은 재투영 오차에도 파라미터가 불안정할 수 있습니다.
- 대칭 체커보드의 코너 순서가 좌우에서 뒤집힐 수 있습니다. 양쪽 보드 방향과 첫 코너 대응을 확인하고,
  양쪽이 같은 보드면을 보도록 촬영하세요. 낮은 오차만으로 대응이 맞았다고 확정할 수 없습니다.
- `camera_id=0`을 LEFT, `1`을 RIGHT로 표시하지만 **실제 물리적 대응은 미확인**입니다.
  카메라 하나씩 가리거나 별도 표식을 사용해 확인하세요. 확인된 사실은 촬영 세션 기록에 별도로 남기세요.

## USB 수동 이미지 수집

ESP32-S3의 native USB 포트를 사용합니다. 같은 포트의 기존 viewer, 시리얼 모니터를 종료하세요.
포트 조회는 기존 뷰어 명령을 사용할 수 있습니다.

```powershell
python proxy/viewer.py --list-ports
python -m app.stereo.capture live --port COM5 --output app/stereo/data/session_01
```

키 동작:

1. LIVE 화면에서 정지한 체커보드가 양쪽 모두 선명한지 확인합니다.
2. `F`: 당시 좌우 프레임을 고정합니다. 한쪽 누락, 해상도 불일치, 오래된 프레임은 거부합니다.
3. 고정된 두 이미지를 확인한 뒤 `S`: **화면에 고정된 바로 그 쌍**을 저장합니다.
4. `R`: LIVE로 돌아가 보드 자세를 바꿉니다. 같은 고정 쌍을 두 번 저장하는 것은 거부합니다.
5. `Q`, `Esc`, 창 닫기: 종료합니다.

`A`는 LEFT, `D`는 RIGHT의 회전을 독립적으로 순환합니다.
화면에 `rot`, 원본 `raw` 해상도, 회전 후 해상도를 표시합니다.
저장 후 설정을 바꿨다면 새 세션 폴더를 사용하세요. 한 세션에 서로 다른 회전 설정이 섞이면 보정을 중단합니다.

`--max-age`는 고정 시 허용하는 PC 수신 후 경과 시간이며 기본 2초입니다.
`--baudrate` 기본값은 115200이며 native USB 실제 속도와는 별개입니다.
연결 오류 시 재접속하고 LIVE 캐시를 비워 서로 다른 연결의 최신 프레임이 섞이는 것을 방지합니다.
JPEG 마커·디코딩·헤더 해상도를 검사하고 손상된 프레임은 저장하지 않습니다.
전송 중 완전히 누락되거나 파서가 거부한 프레임의 개수는 현재 프로토콜만으로 모두 알 수 없습니다.

## 장치 없이 기존 JPEG 가져오기

파일 두 장을 사용자가 명시적으로 선택합니다. 기본값은 양쪽 미리보기 후 `S` 저장이며 `Q`로 취소합니다.

```powershell
python -m app.stereo.capture import --left C:/images/left01.jpg --right C:/images/right01.jpg --output app/stereo/data/session_01
```

이미 검토한 파일이나 자동화 테스트에서는 `--save-without-preview`를 명시해 GUI 없이 저장할 수 있습니다.
이 옵션은 보드 검출이나 동기화를 확인했다는 뜻이 아닙니다.

```powershell
python -m app.stereo.capture import --left C:/images/left01.jpg --right C:/images/right01.jpg --output app/stereo/data/session_01 --save-without-preview
```

파일 가져오기의 `frame_id`, `timestamp_us`는 알 수 없으므로 `null`입니다.
PC 시각은 파일을 읽어 가져온 시각이며 실제 촬영 시각이 아닙니다.

## LEFT/RIGHT 이미지 방향 보정

지원 값은 `0`(회전 없음), `cw90`(90° 시계 방향), `ccw90`(90° 반시계 방향), `180`입니다.
각 값은 **원본 JPEG를 디코딩한 픽셀 기준**입니다. 자동으로 방향을 추정하지 않습니다.
원본 JPEG와 ESP32 헤더의 `width`, `height`는 그대로 유지합니다.
EXIF 방향은 자동 적용하지 않으며, 선택한 회전만 메모리의 이미지에 한 번 적용합니다.
미리보기용 축소는 화면 표시만을 위한 것이고 캘리브레이션에는 원본 크기의 회전 영상이 입력됩니다.
90° 회전은 가로·세로를 교환하며, [OpenCV rotate API](https://docs.opencv.org/4.x/d2/de8/group__core__array.html)
방식으로 픽셀을 재배치합니다. JPEG 재인코딩은 하지 않습니다.

실제 사진으로 방향을 선택하고 프로필을 저장합니다.

```powershell
python -m app.stereo.capture preview --left C:/images/left01.jpg --right C:/images/right01.jpg --output app/stereo/results/orientation_01.json
```

- `A`: LEFT 회전 순환, `D`: RIGHT 회전 순환.
- `S`: 두 영상의 회전 후 크기가 일치하는지 검사하고 설정 JSON 저장.
- `Q`/`Esc`: 취소. 원본 이미지와 기존 프로필은 덮어쓰지 않습니다.

프로필 예시는 다음과 같습니다. **이 예시가 실제 ESP32의 올바른 방향이라는 뜻은 아닙니다.**

```json
{
  "schema_version": 1,
  "left_rotation": "cw90",
  "right_rotation": "ccw90"
}
```

GUI가 없는 환경에서는 카메라별 네 방향을 나란히 보여 주는 8칸 PNG를 생성할 수 있습니다.
이 명령은 비교 그림만 생성하고 회전 설정을 선택하거나 프로필을 저장하지 않습니다.

```powershell
python -m app.stereo.capture preview --left C:/images/left01.jpg --right C:/images/right01.jpg --comparison-output app/stereo/results/rotation_choices.png --no-gui
```

같은 프로필로 수집·보정·정렬을 실행합니다.

```powershell
python -m app.stereo.capture live --port COM5 --orientation app/stereo/results/orientation_01.json --output app/stereo/data/rotated_session_01
python -m app.stereo.capture import --left C:/images/left01.jpg --right C:/images/right01.jpg --orientation app/stereo/results/orientation_01.json --output app/stereo/data/rotated_session_01
python -m app.stereo.calibration --dataset app/stereo/data/rotated_session_01 --columns 9 --rows 6 --square-size 24 --unit mm --orientation app/stereo/results/orientation_01.json --output app/stereo/results/rotated_calibration_01.json
```

프로필 대신 `--left-rotation cw90 --right-rotation ccw90`처럼 값을 직접 지정할 수도 있습니다.
수집에서 미지정한 카메라는 `0`입니다. 캘리브레이션에서 회전 옵션을 모두 생략하면 저장된 쌍의 설정을 사용하며,
값을 직접 지정할 때는 양쪽 값을 함께 지정하는 것을 권장합니다. 프로필과 직접 지정한 값이 충돌하면 거부합니다.
GUI에서 설정을 바꾸면 저장 쌍에는 변경된 값이 기록되며, 시작할 때 읽었던 외부 프로필 파일 자체는 수정하지 않습니다.

새 저장 쌍과 보정 결과는 `schema_version: 2`에 `orientation`을 포함합니다.
서로 다른 선언이 섞인 데이터셋, 수집 메타데이터와 다른 보정 옵션은 **오류로 중단**하며 조용히 제외하지 않습니다.
잘못 선택한 새 쌍의 설정을 고치려면 원본 JPEG를 올바른 프로필로 **새 세션에 다시 import**하세요.
기존 JPEG를 직접 회전해 덮어쓸 필요가 없습니다.

이전 `schema_version: 1` 쌍에는 명시된 회전이 없으므로, 기존 사진은 보정 실행 시
`--orientation` 또는 양쪽 회전 옵션으로 방향을 지정할 수 있습니다. 옵션이 없으면 `0/0`입니다.
이전 보정 결과는 `0/0`으로 불러옵니다. 그 결과에 새 회전을 붙여 재사용하지 말고 사진으로 다시 보정하세요.
새 결과 파일에서 회전 설정이 누락되거나 잘못되면 불러오기를 거부합니다.

실제 사진에서는 보드·글자의 위아래, 좌우 물리적 대응, 미러링 유무와 같은 코너의 대응을 확인하세요.
회전만으로 미러링을 교정하지 않으며, 방향 선택이 노출 동기화나 거리 정확도를 검증하지 않습니다.

## 저장 형식 및 인터페이스

한 쌍마다 UTC 시각과 임의 ID를 포함하는 새 폴더를 생성합니다.

```text
app/stereo/data/session_01/
  pair_<UTC시각>_<ID>/
    left.jpg
    right.jpg
    pair.json
```

JPEG는 원본 바이트를 그대로 저장하며 디코딩된 이미지를 재인코딩하지 않습니다.
파일과 메타데이터를 `.pending_*`에 먼저 작성하고 완료된 폴더만 `pair_*`로 공개합니다.
작성 실패 시 해당 임시 쌍을 정리하며 기존 쌍을 덮어쓰지 않습니다.
강제 종료로 남은 `.pending_*`는 캘리브레이션 대상에서 제외됩니다.

`pair.json`의 좌우 항목에는 다음 정보가 있습니다.

| 필드 | 의미 |
|---|---|
| `camera_id`, `side` | 0/LEFT 또는 1/RIGHT; 물리적 대응 미검증 |
| `frame_id`, `timestamp_us` | CAM2에서 받은 원본 값. 파일 가져오기는 `null` |
| `width`, `height` | 원본 JPEG 해상도 |
| `processed_width`, `processed_height` | 설정된 회전을 적용한 해상도; 새 저장 형식 |
| `pc_received_at_utc` | ISO 8601 UTC 수신/가져오기 시각 |
| `pc_received_monotonic_ns` | 해당 PC 프로세스의 monotonic 시각. 촬영 시각 아님 |
| `saved_path` | `pair.json`이 있는 폴더 기준 상대 경로 |
| `source`, `source_path` | USB 또는 파일 가져오기, 파일 원본 경로 |

`pair_id`는 수동 선택 쌍의 식별자이며 공통 촬영 트리거 ID가 아닙니다.
`synchronized: null`, `synchronization_verified: false`로 동기화 여부가 미확인임을 기록합니다.
카메라 타임스탬프는 README상 카메라별 로컬 시각이며 실제 펌웨어 클록은 미확인입니다.
두 `timestamp_us`가 같거나 `frame_id`가 같다는 이유로 자동으로 쌍을 구성하지 않습니다.
`pc_receive_delta_ms`도 USB 도착/파일 가져오기 간격만 나타냅니다.

상위 `orientation`에 좌우 회전 설정을 기록합니다. 원본 크기가 서로 달라도 회전 후 크기가 같으면 저장할 수 있습니다.
원본 크기가 같더라도 회전 후 좌우 크기가 다르면 저장·보정을 거부합니다.

Python 인터페이스는 `CapturedFrame.from_wire/from_file`, `save_pair`, `load_pair`입니다.
캘리브레이션은 이 저장 형식만 읽으며 USB 포트·GUI·ESP32 연결에 의존하지 않습니다.

## 오프라인 캘리브레이션

```powershell
python -m app.stereo.calibration --dataset app/stereo/data/session_01 --columns 9 --rows 6 --square-size 24 --unit mm --expected-baseline-cm 9 --output app/stereo/results/calibration_01.json
```

처리 순서:

1. `pair_*` 폴더의 메타데이터·JPEG·해상도를 검사합니다.
2. 저장된/명시된 회전을 적용한 뒤 `findChessboardCornersSB`를 우선 사용하고,
   실패하면 `findChessboardCorners`와 `cornerSubPix`를 사용합니다.
3. 양쪽 코너가 모두 검출된 쌍만 사용합니다. **회전 후** 해상도는 첫 유효 쌍 기준으로 통일하며 자동 리사이즈하지 않습니다.
4. `calibrateCamera`로 좌우 내부 파라미터를 각각 계산합니다.
5. `stereoCalibrate(CALIB_FIX_INTRINSIC)`로 좌우 상대 회전·이동을 계산합니다.
6. `stereoRectify`와 `undistortPoints`로 정렬 파라미터 및 대응 코너 수직 오차를 계산합니다.

API 의미는 [OpenCV 공식 Camera Calibration 문서](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)를 참고했습니다.
렌즈 모델은 일반 pinhole 모델이며 fisheye 전용 모델은 포함하지 않습니다.

원본 바이트가 동일한 반복 쌍, 파일 누락, 손상 JPEG, 해상도 불일치, 코너 검출 실패는 제외 사유와 함께 기록합니다.
오차가 높은 쌍은 몰래 제거하지 않고 개별 오차와 경고로 남깁니다. 사용자가 검토해 새 데이터셋으로 다시 보정하세요.
**유효 쌍 3개 미만이면 계산을 중단**하며, 기본 권장 최소값 10개 미만이면 계산 결과에 경고를 남깁니다.
3개는 계산을 시도할 최소 조건이며 신뢰할 수 있는 품질 보장이 아닙니다.

## 파라미터·단위·품질 보고서

결과 JSON에는 다음 정보가 포함됩니다.

- 좌우 `K`, 왜곡 계수 `D`, 상대 `R`, `T`, `E`, `F`.
- 정렬 `R1`, `R2`, `P1`, `P2`, `Q`, 좌우 유효 ROI.
- 좌우 `orientation`, `input_space: rotated_raw_pixels`; `image_size`는 **회전 후** 보정 좌표계의 크기.
- 체커보드 설정과 `length_unit`, OpenCV 버전, 품질 설정, 입력 데이터셋 경로.
- 좌우 재투영 RMS, Stereo RMS, 쌍별 재투영·수직 오차.
- 정렬 후 대응 코너 수직 오차의 평균 절댓값, RMS, p95, 최대값.
- 사용·전체·제외·검출 실패 쌍 개수와 제외 사유. 검출 실패 개수는 전체 제외 개수의 일부입니다.
- `norm(T)`로 계산한 추정 카메라 간격과 약 9cm 기준의 상대 차이.

보드 좌표와 `T` 및 향후 `Q`를 사용하는 복원 좌표의 길이 단위는 `--unit`으로 유지됩니다.
9cm는 근사 비교값이며 고정 제약으로 넣지 않습니다. 실제 실측값이 다르면 `--expected-baseline-cm`에 입력하세요.
결과 JSON을 불러올 때 버전, 행렬 크기, 유한값, 입력 해상도를 검사합니다. 기존 결과 파일은 덮어쓰지 않습니다.
독립 사용 API: `calibrate_dataset`, `fit_calibration`, `save_result`, `load_result`.

초기 검토 기준은 다음과 같으며 프로젝트 실증으로 정한 합격 기준은 아닙니다.

| 항목 | 기본 경고 기준 | 옵션 |
|---|---|---|
| 유효 쌍 수 | 10개 미만 | `--min-pairs` |
| 좌우 재투영 RMS | 1.0px 초과 | `--max-reprojection-px` |
| 개별 쌍 재투영 RMS | 위 기준의 2배 초과 | 같은 옵션 |
| Stereo RMS | 1.5px 초과 | `--max-stereo-rms-px` |
| 정렬 수직 RMS | 1.0px 초과 | `--max-vertical-rms-px` |
| 정렬 수직 p95 | 수직 RMS 기준의 2배 초과 | 같은 옵션 |
| 기준 간격과 상대 차이 | 20% 초과 | `--baseline-relative-tolerance` |
| 보드 기울기 다양성 | 추정 보드 법선 최대 각도차 10도 미만 | 현재 고정 경고 |

수직 스테레오 배치, 정사각 내부 코너 격자도 추가 경고 대상입니다.
임계값을 완화해서 경고를 없애도 측정 품질이 검증된 것은 아닙니다.

종료 코드:

- `0`: 계산·저장 완료, 설정한 **수치 검토 기준** 통과.
- `2`: 계산·저장 완료, `REVIEW REQUIRED`. 부족한 사진 또는 품질 경고가 있음.
- `1`: 입력 오류, 계산 불가, 파일 저장 실패 등. 완료로 취급하지 않음.

모든 결과는 `physical_measurement_unverified`입니다. 보고 오차는 **보정에 사용한 데이터에 대한 오차**이며
독립 검증 사진·실제 기준 거리로 평가한 성능이 아닙니다. 실제 L/R 대응과 노출 동기화도 미검증입니다.
수치 기준을 통과해도 실제 거리 측정에 성공했다는 메시지를 출력하지 않습니다.

## Rectification 시각화

아래 `<pair_ID>`를 실제 저장 폴더명으로 바꿉니다. 입력 해상도가 보정 당시와 다르면 거부합니다.
입력은 **원본 JPEG**이며 보정 결과의 회전을 자동 적용합니다. 이미 회전한 이미지를 넘기면 안 됩니다.
검사용 `--orientation` 또는 회전 옵션을 지정하면 보정 결과와 반드시 일치해야 합니다.
같은 해상도를 만드는 반대 방향 회전이나 0°/180°도 설정값으로 비교해 거부합니다.
JPEG와 같은 폴더에 `pair.json`이 있으면 파일의 L/R 대응과 수집 당시 설정도 자동 검사합니다.
서로 다른 저장 쌍의 파일을 섞는 것은 거부합니다. 별도 JPEG에 메타데이터가 없으면 내용만으로 회전 설정의
옳고 그름을 알 수 없으므로 실제 사진을 확인하고 `--orientation`으로 현재 입력의 설정을 명시하세요.

```powershell
python -m app.stereo.rectify --parameters app/stereo/results/calibration_01.json --left app/stereo/data/session_01/<pair_ID>/left.jpg --right app/stereo/data/session_01/<pair_ID>/right.jpg --output app/stereo/results/preview_01 --show
```

`initUndistortRectifyMap`과 `remap`을 사용해 다음 PNG를 생성합니다.

- `left_rectified.png`, `right_rectified.png`: 정렬 영상.
- `before.png`, `rectified.png`: 좌우 나란히 보기와 녹색 수평선.

`before.png`는 방향 보정 후·렌즈/스테레오 정렬 전 영상입니다.
`rectify_pair`는 원본 배열을 받아 저장된 회전을 적용하고 정렬합니다.
향후 StereoSGBM에서도 이 함수를 재사용하면 같은 설정이 적용됩니다.
필요하면 `prepare_calibration_inputs`를 사용해 회전 설정·회전 후 크기를 먼저 검증할 수 있습니다.
두 함수의 입력 계약은 원본 픽셀 배열이므로 외부에서 회전을 먼저 적용하지 마세요.

같은 코너가 수평선상에 놓이는지 확인하세요. `--show`를 생략하면 GUI 없이 PNG만 생성합니다.
`alpha=0`으로 유효 영역 중심의 정렬 영상을 생성합니다. 결과 폴더도 기존 폴더를 덮어쓰지 않습니다.
가능하면 **보정에 사용하지 않은** 별도 사진도 시각화해 확인하세요.

## 하드웨어 없는 테스트

레포지토리 루트에서 실행합니다.

```powershell
python -B -m unittest discover -s app/stereo/tests -t . -v
```

테스트는 `app/stereo/results/test_*` 안에 임시 데이터를 생성하고 종료 시 정리합니다.
테스트는 실제 기존 FrameParser의 분할 수신·재동기화, JPEG 바이트 보존, 메타데이터,
누락·손상·해상도·오래된 프레임, 저장 실패, 경로 검사 등을 확인합니다.
알려진 90mm 가상 카메라의 투영 코너로 파라미터·단위·오차를 검증하고,
렌더링된 체커보드 JPEG 12쌍으로 검출부터 보정·제외 집계·JSON·CLI·PNG 출력까지 실행합니다.
합성 테스트 통과는 실제 ESP32 카메라 정확도를 보장하지 않습니다.

이번 개발 환경의 실행 결과: Python 3.12.10 / OpenCV 5.0.0 / NumPy 2.5.3 / pyserial 3.5,
총 22개 테스트를 통과했습니다. 기존 13개 테스트에 방향·프로필·크기·기존 형식 호환·미리보기 선택·회전 JPEG 통합 테스트를 추가했습니다.
기존 합성 JPEG 12쌍 사용·4쌍 제외, 추정 간격 약 9.0309cm, 정렬 수직 RMS 약 0.2433px.
회전된 JPEG 12쌍에서는 추정 간격 약 9.0314cm, 정렬 수직 RMS 약 0.2419px입니다.
GUI 선택 로직은 키 입력을 모의하여 검사하며 실제 사진의 방향과 실제 GUI 조작·라이브 장치는 미검증입니다.

## 다음 단계와 협업 경계

실제 실패 데이터의 재현·코너 번호·검출기 비교·별도 제외 데이터셋은 다음 진단 명령으로 생성합니다.
`--output`은 원본 데이터셋 밖의 **새 폴더**여야 합니다.

```powershell
python -m app.stereo.diagnose --dataset app/stereo/data/session_02 --reference app/stereo/results/calibration_02.json --output app/stereo/results/diagnosis_new
```

`corners/`에 모든 사진의 검출 코너 번호를 표시한 PNG와 좌우 비교 PNG를 생성합니다.
SB 실패 시 기존 검출기의 초기 좌표와 `cornerSubPix` 창 크기별 좌표를 비교한 PNG·JSON도 생성합니다.
`pair_errors.csv`, `diagnosis.json`, `reproduced_original.json`, `calibration_filtered.json`,
원본 바이트를 복사한 `dataset_filtered/`를 함께 생성합니다.
제외 규칙은 기존 결과와 같은 개별 카메라 재투영 기준입니다. 코너 순서 반전·미러링·기준 완화·간격 고정은 하지 않습니다.
원본 파일과 기존 결과의 SHA-256을 전후 비교합니다. 진단 수치가 좋아져도 실제 측정 품질은 미검증입니다.

`session_02`에서 확인된 문제는 SB/기존 검출기의 서로 다른 코너 시작 방향과,
작은 격자에 과도하게 큰 `(11,11)` 서브픽셀 창을 적용한 것입니다. 이 값은 실제 **23×23픽셀** 창입니다.
기본 `legacy` 경로는 기존 좌표를 유지합니다. 아래 `consistent` 옵션에서만 창 크기와 코너 대응을 개선합니다.
격자 오차가 낮아도 180° 대칭성 때문에 좌우 코너 대응은 별도로 확인해야 합니다.

수신·수동 저장은 `capture.py`/`dataset.py`, 알고리즘은 `calibration.py`/`rectify.py`로 분리했습니다.
`proxy/viewer.py` 내부 파서에 의존하므로 팀에서 CAM2 헤더·ID·해상도·클록 규칙을 바꿀 때 함께 확인해야 합니다.
`data/`, `results/`, `.venv/`는 이 폴더의 `.gitignore`로 제외합니다. 원본 사진·결과 JSON은 별도 보관하세요.

StereoSGBM 구현 전에는 실제 송신 펌웨어 확인, 물리적 L/R 확인, 충분한 실제 보드 데이터 수집,
독립 데이터의 정렬 품질 검증, 렌즈 모델 적합성·장착 안정성 확인이 필요합니다.
움직이는 대상을 측정하려면 노출 동기화·프레임 쌍 정책도 필요합니다.
거리 측정을 추가한 뒤에는 여러 실제 기준 거리에서 오차를 검증해야 합니다.

## 실측 Baseline 제약 보정과 비교

기본 보정은 자유 Baseline 추정이며 기존 동작을 유지합니다. `--expected-baseline-cm`은 비교 기준이고,
실제 제약은 별도 옵션 `--fixed-baseline-mm`으로 선택합니다. 길이 입력은 항상 mm이며 보드 단위로 변환합니다.

```powershell
python -m app.stereo.calibration --dataset app/stereo/data/session_02 --columns 9 --rows 6 --square-size 22 --unit mm --expected-baseline-cm 9.5 --fixed-baseline-mm 95 --corner-policy consistent --output app/stereo/results/calibration_fixed95_new.json
python -m app.stereo.compare_baseline --previous app/stereo/results/calibration_02.json --baseline-mm 95 --holdout --output app/stereo/results/baseline95_new
```

`consistent`는 작은 격자에서 기존 검출기 fallback의 서브픽셀 반경을 인접 코너 간격의 35% 이하
(최대 5픽셀)로 제한합니다. 양쪽 행·열 방향이 모두 반대로 검출되면 RIGHT 코너 배열을 180° 반전하고
그 사실을 각 쌍의 `corner_consistency`에 기록합니다. 원본 JPEG나 `pair.json`은 변경하지 않습니다.
이 대응 규칙은 **올바르게 회전한 영상에서 평행 카메라가 같은 보드면을 보는 경우**에만 적용합니다.
표식 없는 체커보드의 실제 코너 정체성·미러링·물리적 L/R 위치를 자동 인증하는 기능이 아닙니다.
격자 일관성·방향·재투영·정렬 오차를 함께 확인해야 합니다.

제약 보정은 독립 단안 보정의 K/D를 고정합니다. [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)로
상대 회전 3자유도, 이동 방향 2자유도, 쌍별 보드 자세 6자유도를 함께 최적화합니다.
T를 구면 방향과 실측 길이의 곱으로 표현하므로 모든 반복에서 길이 제약을 만족합니다.
R은 단위행렬로 고정하지 않습니다. 자유 Stereo 결과와 단안 상대 자세의 중앙값을 두 초기값으로 사용하고,
최적화의 수렴·카메라 앞쪽 깊이·오차·ROI·실제 remap 유효 영역을 검사합니다.
최적화 실패나 정렬 실패는 95mm 일치와 관계없이 REVIEW REQUIRED입니다.

비교 명령은 같은 원본 관측과 같은 개선 관측 각각에 자유 추정 / T만 스케일링 / 길이 제약 최적화를 실행합니다.
사용·제외 쌍, 단안 RMS와 공통 보드 자세의 좌우 재투영 RMS, Stereo RMS, 수직 RMS/p95/max,
ROI, 원본을 벗어나지 않는 remap 픽셀 비율, 코너 방향·격자 일관성, 최적화 진단을 저장합니다.
T만 스케일링한 결과는 진단 전용이며 재보정 성공으로 판정하지 않습니다.
카메라 뒤쪽 보드 또는 미수렴이 나온 공통 자세 재투영 결과는 INVALID로 표시하고 원시 수치는 JSON에 남깁니다.
단안 RMS는 독립 자세로 계산하기 때문에 Baseline 제약 전후 동일합니다. Stereo RMS는 2D 오차제곱 합을
좌우 전체 코너 수로 나눈 제곱근입니다. 잘못된 기하에서 별도 자세 재평가가 다른 국소해를 찾을 수 있어
기존/최적화 solver RMS와 재평가 RMS를 구분해 기록합니다.
`alpha=0`의 출력 유효 비율 100%는 원본 화각 전체를 보존한다는 뜻이 아니므로 초점거리·PNG도 확인하세요.

`--holdout`은 개선 관측을 순서 기준 매 3번째 검증 쌍으로 나누며 K/D/R/T 추정에 해당 사진을 사용하지 않습니다.
검증 사진에서도 같은 보드의 자세만 피팅합니다. 동일 촬영 세션 내부 검증이므로 별도 촬영 실증을 대체하지 않습니다.
새 출력 폴더만 허용하고 입력 JPEG·메타데이터·기존 JSON의 SHA-256을 전후 비교합니다.

`session_02`의 19쌍을 그대로 유지한 개선 자유 추정은 Baseline 약 96.975mm, Stereo 0.4155px,
수직 0.5420px입니다. 개선 95mm 제약은 Stereo 0.4185px, 수직 0.5468px이며 ROI는 양쪽 240×320입니다.
12쌍 보정/7쌍 검증의 수직 RMS는 자유 0.6758px, 제약 0.6880px입니다.
큰 개선의 원인은 코너 검출·대응 수정이며 95mm 제약 자체가 정렬을 더 좋게 만들었다고 해석하지 않습니다.
제약 결과의 수직 최대 오차 4.86px, 실제 L/R 미확인, 동기화 미확인, 거리 실증 미완료를 함께 고려하세요.

추가 8개 테스트는 알려진 95mm 기하의 R/T·단위 변환, analytic Jacobian, 잘못된 대응의 실패,
스케일링 진단의 정렬 불변성, 명시적 순서 반전, 작은 격자 fallback 창을 검증합니다.
기존/진단 테스트를 포함한 실행 결과는 33개 통과입니다.

## 14쌍 자유 보정 결과의 최종 검증

`validate.py`는 보정 행렬을 유지한 채 시차·깊이 부호, 선별 기록, 미사용 사진 및
교차검증을 평가합니다. 이 명령은 고정 Baseline 최적화나 코너 순서 반전을 사용하지 않습니다.

```powershell
python -m app.stereo.validate `
  --dataset app/stereo/data/session_02 `
  --result app/stereo/results/diagnosis_session_02/audit/calibration_filtered.json `
  --selection app/stereo/results/diagnosis_session_02/audit/diagnosis.json `
  --expected-baseline-mm 95 `
  --reviewed-holdouts app/stereo/results/validation_session02_95mm/reviewed_holdouts.json `
  --output app/stereo/results/validation_new
```

출력은 새 폴더만 허용합니다. `calibration_reference95.json`은 검증 기준만 95mm로 변경한
복사본으로 K/D/R/T/P/Q는 기존 결과와 동일합니다. 기존 결과 파일은 변경하지 않습니다.
`validation.json`, `corner_geometry.csv`, `experiment.jsonl`에 결과와 실험 조건을 남깁니다.
14개 leave-one-out 및 시간 순서 4개 블록의 재보정 결과도 별도로 보존합니다.

미사용 사진은 SB의 EXHAUSTIVE/ACCURACY 옵션으로 검출하며 fallback을 사용하지 않습니다.
좌우 코너를 반전하거나 낮은 정렬 오차를 보고 순서를 고르지 않습니다. 번호 이미지를 직접 확인한
JPEG·코너 해시가 일치하는 검토 manifest가 있어야 미사용 사진 평가에 포함합니다.
`--reviewed-holdouts`가 없으면 후보 이미지와 감사 정보만 만들고 미사용 사진을 평가에서 제외합니다.
다른 사진에 기존 manifest를 재사용하지 말고 새 검출 번호 이미지를 검토해야 합니다.

현재 14쌍 결과의 시차 정의는 `d = x_LEFT - x_RIGHT`입니다. 실제 대응점의 시차는 음수이고
저장된 Q가 이를 양의 깊이로 변환합니다. 양의 시차나 `abs(d)`를 이 Q에 넣으면 음의 깊이가 됩니다.
`T_x`는 첫 카메라에서 두 번째 카메라로의 좌표 변환 성분이며 카메라 위치는 `-R.T @ T`로 확인합니다.
물리적 L/R 및 미러링은 카메라 가리기와 비대칭 표식 촬영으로 별도 확인해야 합니다.

검증 결과 `run_02`: 미사용 6쌍 수직 RMS 0.324px, leave-one-out 0.607px,
시간 블록 0.775px입니다. 4쌍은 한쪽 검출 실패로 평가할 수 없습니다.
이는 같은 촬영 세션을 재사용한 후향적 검증이며 실제 거리 오차를 인증하지 않습니다.
LEFT 왜곡 모델의 센서 가장자리 외삽과 일부 사진의 1px 초과 오차를 추가 검토해야 합니다.
프로그램 종료 코드 0은 감사 실행 완료를 뜻하며 거리 측정 품질 PASS를 뜻하지 않습니다.
