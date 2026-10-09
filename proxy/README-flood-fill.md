# VIDEX Flood Fill 장애물 감지 모듈

정상 시연한 사다리꼴 ROI + Flood Fill WARNING 알고리즘을 재사용 가능한 Python 모듈로 분리했다.
기본 ROI, 커널, 판단 임계값, 신규 프레임 확인/해제 규칙을 보존한다.

## 설치

Python **3.10 이상**. 아래 명령은 저장소 루트(`videx`)에서 PowerShell로 실행한다.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r proxy/requirements-viewer.txt
```

USB가 필요 없는 이미지 분석만 사용한다면 `proxy/requirements-flood-fill.txt`만 설치하면 된다.
NumPy와 OpenCV가 필요하며, USB 실행에만 pyserial이 추가된다. 별도 모델 파일은 필요 없다.
GUI 없는 서버에서는 새 가상환경에 `numpy>=1.24`와 `opencv-python-headless>=4.8`을 설치해
핵심 모듈을 사용할 수 있다. `opencv-python`과 headless 패키지는 같은 환경에 함께 설치하지 않는다.

## 빠른 실행

```powershell
# 실제 USB: LEFT(camera_id=0), 기존 시계방향 90도 보정
& .\.venv\Scripts\python.exe -m proxy.flood_fill_mvp --port COM5

# USB 없이 기존 합성 시연: 빈 바닥 → 중앙 물체 → 빈 바닥
& .\.venv\Scripts\python.exe -m proxy.flood_fill_mvp --demo

# GUI 없이 제한 시간 실행 및 결과 저장
& .\.venv\Scripts\python.exe -m proxy.flood_fill_mvp --demo --headless --seconds 11 --output-dir flood-results/demo
```

기존 `python proxy/flood_fill_mvp.py --port COM5` 명령도 유지된다.
Q/Esc는 종료, S는 현재 화면 저장. `--output-dir`은 상태 전환 PNG, `last.png`, `report.json`을 저장한다.
원본 영상 / Canny / 채움 결과의 세 패널과 전환 이유, 비율, 프레임 ID를 표시한다.
`--headless`는 창만 숨긴다. 결과 저장을 위한 화면 합성 동작은 기존과 같다.
실시간 모드에서 COM5는 다른 뷰어나 시리얼 모니터가 사용하고 있지 않아야 한다.

## 다른 프로그램에서 import

```python
import cv2
from proxy.flood_fill import Detector

detector = Detector()  # 스트림별로 하나를 생성하고 계속 재사용
frame = cv2.imread("frame.jpg")  # 이미 올바른 방향인 uint8 BGR 영상
result = detector.process(frame)  # 호출 시각은 time.monotonic() 사용
print(result.state, result.unfilled_ratio, result.central_ratio, result.reason)
if result.valid and result.analysis is not None:
    roi = result.analysis["roi"]
    reachable = result.analysis["filled"]
    blocked = result.analysis["unfilled"]
```

WARNING은 연속 신규 프레임 3개가 필요하므로 동일 인스턴스에 프레임을 차례로 전달한다.
핵심 모듈은 회전/리사이즈를 하지 않는다. ESP32 LEFT의 원본 가로 영상을 직접 전달할 경우
먼저 `cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)`를 적용한다. USB CLI는 이를 자동 수행한다.
Android/PC 서버 연동에서는 BGR 배열로 디코딩한 뒤 이 API를 사용한다. 네이티브 Android 패키지는 아니다.

### 프레임 ID와 시각

```python
# 프레임에 고유 ID가 있는 수신 루프
result = detector.process(frame, received_at, frame_id=frame_id, now=now)
# 영상이 들어오지 않을 때도 호출해 2초 타임아웃을 반영
result = detector.poll(now)
```

- 시각 단위는 **초**. `received_at`, `now`, `poll()`은 같은 시계 기준을 사용한다.
- `process(frame)`는 현재 monotonic 시각을 사용한다.
- `process(frame, timestamp)`는 기본적으로 그 시각에서 판단한다. 녹화/합성 입력 재생에 사용 가능하다.
- 실제 수신 지연 검사에는 같은 시계의 `now`도 전달한다. 오래된 큐 프레임은 UNKNOWN으로 거절한다.
- `frame_id`가 있으면 직전과 같은 ID를 중복으로 취급한다. 없으면 직전과 같은 timestamp가 중복이다.
- 같은 픽셀이라도 새로운 ID/시각이면 새 관측이다. 픽셀 비교로 중복 판정하지 않는다.
- 중복은 분석/확인 횟수/마지막 수신 시각을 갱신하지 않는다. ID 역행은 연속 확인 횟수를 초기화한다.
- 영상 크기 변경 시 ROI/마스크를 새 크기로 재생성하며, **기존처럼 확인 횟수는 유지**한다.
- 한 Detector는 하나의 순차 스트림용이며 동시 호출용이 아니다.

### 반환값 `DetectionResult`

| 필드 | 의미 |
| --- | --- |
| `state` | `WARNING`, `MONITORING`, `UNKNOWN` |
| `unfilled_ratio` | 전체 ROI 중 미충전 비율, 0~1 |
| `central_ratio` | 중앙 영역 중 미충전 비율, 0~1 |
| `reason` | 상태 판단 근거 문자열 |
| `valid` | 현재 상태가 UNKNOWN이 아닌지 |
| `updated` | 이번 호출에서 신규 프레임을 분석했는지 |
| `frame_id`, `timestamp` | 마지막 처리 프레임과 수신 시각 |
| `consecutive_hits`, `consecutive_clears` | 연속 경고/해제 조건 횟수 |
| `analysis` | 영상, 경계선, 마스크, 좌표를 담은 dict |

`analysis`에는 `image`, `edges`, `barriers`, `roi`, `center`, `polygon`, `filled`,
`unfilled`, `seed`, `total`, `central`, `edge_density`, `invalid`가 있다.
마스크는 입력 크기의 **bool H×W**, `edges`는 uint8 H×W, `polygon`은 int32 꼭짓점 좌표,
`seed`는 `(x, y)` 또는 None이다. `image`는 입력 BGR 배열을 참조한다.
배열은 복사 비용을 피하기 위해 공유하므로 호출자가 입력/반환 배열을 수정하지 말고 필요시 `.copy()`한다.

처리된 영상이 없으면 비율과 analysis는 None이다. UNKNOWN으로 바뀌었을 때는
마지막 프레임의 비율/마스크가 남을 수 있으므로 **현재 판단에는 `valid`를 먼저 확인**한다.
상태/카운터는 반환 시점의 스냅샷이다. JSON 전송 시 배열을 제외한 필요한 필드만 직렬화한다.
잘못된 입력 형식(None, 빈 영상, grayscale/RGBA, float 배열)은 TypeError/ValueError를 발생시킨다.
정상 uint8 BGR이지만 너무 작거나 암전 등 판단 불가 영상은 UNKNOWN을 반환한다.

## 알고리즘과 보존한 기본값

1. 팀원 `outline.extract_outline()`으로 전체 영상에 GaussianBlur(5×5) + Canny(50,150).
2. Closing 5×5 한 번 + Dilate 3×3 한 번. 입력 해상도 기준 픽셀 크기.
3. 사다리꼴 ROI: (36%,42%), (64%,42%), (92%,97%), (8%,97%).
4. 하단 (50%,94%)에 가장 가까운 시작점을 x=46~54%, y=90~96%에서 찾는다.
   경계로부터 2px 이상 떨어진 점을 사용하며, ROI 내부에서 4방향 Flood Fill을 한다.
5. 미충전 = ROI - 충전 영역. 경계 픽셀도 포함한다. 중앙은 x=42~58%, y=45~85%와 ROI의 교집합.
6. 전체 미충전 **>=18% AND 중앙 >=30%**, 신규 프레임 **3회 연속**이면 WARNING.
7. WARNING 해제는 전체 **<12.6% OR 중앙 <21%**, 신규 프레임 **3회 연속**일 때.
8. 신규 영상 2초 부재, 시작점 불가, 작은 영상(한 변 <32px), ROI의 98% 초과 암전/과노출,
   경계 밀도 >45%는 UNKNOWN. 복구 후 다시 3프레임 확인한다.

`Config`에 모든 판단 기본값과 정규화된 좌표를 모았다. 값 변경은 선택 사항이며 기본값 변경은 없다.
기존 CLI의 `--low`, `--high`, `--kernel`, `--total-threshold`, `--center-threshold`,
`--confirm`, `--stale`도 그대로 지원한다. 비율은 0~1 단위이다.

미충전 영역은 장애물 후보이다. 바닥 무늬/그림자/ROI를 가로지르는 선은 오경고,
끊긴 윤곽선/저대비/물체 내부의 시작점은 미검출을 일으킬 수 있다.
MONITORING은 통행 안전의 확정을 뜻하지 않으며 거리·깊이·접근 속도는 계산하지 않는다.

## 파일 구조와 기존 코드 연결

- `proxy/flood_fill.py`: Config, ROI/채움 함수, 비율 계산, Detector, DetectionResult. GUI/USB 실행 없음.
- `proxy/flood_fill_mvp.py`: 기존 USB/합성 시연 CLI, 회전 보정, 세 패널 시각화.
- `proxy/outline.py`: 기존 팀원 코드. 핵심은 `extract_outline()`, CLI는 `receive_frames()`를 import.
  기존 `FrameParser`, CAM2 검증/JPEG 디코딩, 재연결 흐름을 그대로 이용한다.
- `proxy/test_flood_fill_mvp.py`, `proxy/test_flood_fill_api.py`: 동작 및 API 회귀 테스트.
- `proxy/fixtures/flood_fill_baseline.json`: 리팩터링 전 23단계 입력의 기대 결과. 임시 출력물이 아닌 테스트 데이터.

기존 팀원 소스 확인 기준:
https://github.com/kimssammwu/videx/blob/5c4547de7eb8a5a6157991f194e1f1218c426c1c/proxy/outline.py

이번 모듈화에서 `outline.py`는 수정하지 않았다. 현재 작업 폴더의 기존 미커밋 CAM2 버튼
메타데이터 허용 변경은 별도로 보존했다. 새 버튼 메타데이터를 보내는 펌웨어에서는 그 파서 변경도
필요하며, 이 변경의 포함 여부는 팀원 작업과 별도로 검토해야 한다. 기존 CAM2 헤더와는 호환된다.

## 검증

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s proxy -p "test_flood_fill*.py" -v
```

WARNING 입력, 낮은 미충전, ROI 외부 경계, 중복 ID/시각, 경고 확인/해제/타임아웃,
프레임 ID 재시작, 입력 크기 변경, 작은 틈 보완/큰 틈 누수, 시작점, 암전/과노출,
잘못된 입력, 기존 CAM2 수신기를 통한 합성 JPEG 분할 패킷 복원을 검증한다.

리팩터링 전 원본을 별도 보관하고 동일 입력 23단계에서 상태·근거·비율·카운터와
모든 마스크 픽셀의 완전 일치를 확인했다. 회귀 기준은 OpenCV 5.0.0에서 생성했다.
2026-10-09 모듈화 검증: **28개 테스트 통과**. 업로드 대상 파일과 Git HEAD의 기존
`outline.py`만 복사한 별도 폴더에서도 동일 테스트와 합성 CLI 시연이 통과했다.
저장된 실제 카메라 JPEG 29장에서는 판단·마스크·시각화 픽셀까지 기존과 같았다.
같은 저장 영상을 번갈아 처리한 분석 시간 중앙값은 기존 1.423ms, 모듈화 후 1.430ms였다
(이 PC에서의 참고 측정이며 실시간 USB 전송 성능 측정은 아니다).
이번 검증 시 COM5 장치가 없어 **실제 카메라 수신은 미검증**이다.
합성 테스트/저장 영상 재생 성공을 실시간 카메라 검증으로 간주하지 않는다.

## GitHub 업로드 준비

저장소 루트 `.gitignore`는 가상환경/캐시와 `flood-results/`, `mvp-results/`,
`flood_snapshot.png`를 제외한다. 출력은 이 디렉터리를 사용한다.
기존 다른 작업자의 실험 파일은 삭제하지 않았고 아래 추가 목록에도 포함하지 않는다.
**Commit/Push는 이 작업에서 실행하지 않는다.** 준비된 파일만 추가하려면:

```powershell
git add -- .gitignore proxy/flood_fill.py proxy/flood_fill_mvp.py proxy/test_flood_fill_mvp.py proxy/test_flood_fill_api.py proxy/fixtures/flood_fill_baseline.json proxy/requirements-flood-fill.txt proxy/requirements-viewer.txt proxy/README-flood-fill.md
git diff --cached --stat
git diff --cached --check
```

기존에 이미 staged된 파일이 있다면 먼저 `git diff --cached --name-only`로 확인한다.
권장 커밋 메시지: `refactor: modularize VIDEX flood-fill warning detector without behavior changes`
