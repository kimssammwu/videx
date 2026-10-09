"""Build the final Korean report from the saved experiments, without refitting."""
import csv
import json
from pathlib import Path

import numpy as np

from .experiment_session02 import DATA, ROOT, hashes, write_json

OUTPUT=ROOT/'app/stereo/results/autonomous_session02_20261009'


def load(name):
    return json.loads((OUTPUT/name).read_text(encoding='utf-8'))


def report():
    best=load('calibration_session02_best.json')
    audit=load('audit.json')
    original=json.loads((ROOT/'app/stereo/results/calibration_02.json').read_text())
    original_errors={r['pair_id']:r for r in original['report']['per_pair']}
    best_errors={p['pair_id']:p for e in ('training_evaluation','held_out_evaluation','excluded_outlier_evaluation') for p in best[e]['per_pair']}
    notes={
        1:'화면에 가로 밝기 띠와 모아레. LEFT 기본 검출 실패; CLAHE/3배 SB로 복구. 검증 전용.',
        2:'중앙 정면, 보드 전체 표시. 양안 노출 차이는 있으나 대응 정상.',
        3:'화면 우측으로 이동, RIGHT 여백이 작지만 54개 내부 코너 확인.',
        4:'화면 좌측, 회전과 기울기 변화. 전체 격자 정상.',
        5:'보드 상단 위치, 원근 기울기 존재. 전체 격자 정상.',
        6:'보드 하단 위치, RIGHT 흐림과 약한 대비. 화면 잘림이라는 초기 분류를 확대검사 후 수정. LEFT 감마 보정, RIGHT 수동 초기 호모그래피로 영상만 전개한 뒤 SB가 실제 54코너 재검출. 검증 전용.',
        7:'RIGHT에서 보드 오른쪽 여러 열이 영상 밖으로 잘림. LEFT 검출 가능; 전체 좌우 54대응 복구 불가.',
        8:'큰 격자, 중앙. 강화된 SB로 일부 초기 좌표 오차 감소.',
        9:'LEFT classic fallback의 순서 반전과 23×23 창의 코너 이동. 재검출로 복구.',
        10:'보드 대각 회전. RIGHT 기본 검출 실패, 강화된 SB로 복구. 보정 후 일부 RIGHT 코너는 alpha=0 잘림 영역.',
        11:'LEFT의 녹색/밝기 띠 및 양안 화면 밝기 차이. 코너 순서는 정상이나 일관된 수직 오프셋; 재검출 후에도 이상치.',
        12:'가장 가까운 보드 위치. LEFT fallback 순서 반전, 일부 코너 이동. 강화된 SB로 복구.',
        13:'가까운 보드, 화면 상부. RIGHT 여백과 대조가 작지만 내부 코너 정상.',
        14:'가까운 보드, 면내 회전과 원근 변화. 내부 코너 정상.',
        15:'LEFT fallback 순서 반전 및 대규모 격자 훼손. 강화된 SB로 복구.',
        16:'가까운 보드, 약한 면내 회전. 내부 코너 정상.',
        17:'LEFT fallback 순서 반전 및 코너 중복/이동. 강화된 SB로 복구.',
        18:'보드 하단 위치. 내부 코너 정상.',
        19:'RIGHT fallback 순서 반전과 코너 중복/이동. 강화된 SB로 복구.',
        20:'보드 대각 회전. 재검출 후 정상; 작은 잔여 수직 오프셋.',
        21:'작고 세로에 가까운 보드 방향. 코너 간격이 가장 작아 고정 11px 반경 부적절. SB 대응 정상.',
        22:'보드 작음, 손을 머리 위로 든 구도, 일부 흐림. 단독 재투영은 작지만 좌우 공동 포즈 오차가 큼; 이상치.',
        23:'작은 보드, 원근 기울기. 재검출 후 정상.',
        24:'앞뒤 기울기로 격자 높이가 압축되고 화면에 띠가 있음. ROI 검색으로 복구. 검증 전용.',
    }
    rows=[]
    for a in audit:
        n=a['n'];p=best_errors.get(a['pair_id']);old=original_errors.get(a['pair_id'])
        group='학습' if n not in (1,6,7,11,22,24) else ('미사용 검증' if n in (1,6,24) else ('이상치 제외' if n in (11,22) else '전체 격자 불가'))
        rows.append({'n':n,'pair_id':a['pair_id'],'group':group,
                     'original_left_detector':a['left']['original_info']['detector'],
                     'original_right_detector':a['right']['original_info']['detector'],
                     'original_left_rms_px':None if old is None else old['left_reprojection_rms_px'],
                     'original_right_rms_px':None if old is None else old['right_reprojection_rms_px'],
                     'final_left_rms_px':None if p is None else p['left_rms_px'],
                     'final_right_rms_px':None if p is None else p['right_rms_px'],
                     'final_stereo_joint_rms_px':None if p is None else p['stereo_joint_pose_rms_px'],
                     'final_vertical_rms_px':None if p is None else p['vertical_rms_px'],
                     'pc_arrival_delta_ms':a['metadata']['pc_receive_delta_ms'],
                     'camera_local_timestamp_difference_ms':(a['metadata']['right']['timestamp_us']-a['metadata']['left']['timestamp_us'])/1000,
                     'notes':notes[n]})
    with (OUTPUT/'all24_photo_audit.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(OUTPUT/'all24_photo_audit.json',rows)
    paths=[('저장된 기존 결과',None),('원본 코너 순서만 수정','02_original_order_corrected'),
           ('원본 정상 14쌍','03_original_clean14'),('강화된 SB 20쌍','06_accurate_corrected20'),
           ('k1/k2, 내부 파라미터 고정, 20쌍','09_k1k2_fixed20_recovered_holdout'),
           ('k1/k2, 정상 18쌍 — 선택','10_k1k2_fixed18_recovered_holdout'),
           ('왜곡 없음, 정상 18쌍','10_zero_fixed18_recovered_holdout'),
           ('95mm 제약, 내부 파라미터 고정, 18쌍','14_bundle18_baseline95_fixedK'),
           ('95mm 제약, 내부 파라미터도 최적화, 18쌍','14_bundle18_baseline95_freeK'),
           ('복구 2쌍까지 포함, 22쌍','12_k1k2_fixed22_including_recovered')]
    table=['| 방법 | 학습 쌍 | LEFT RMS px | RIGHT RMS px | Stereo RMS px | 수직 RMS px | Baseline mm |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,name in paths:
        d=original if name is None else load('experiments/'+name+'.json')
        r=d['report']
        table.append(f'| {label} | {r["used_pairs"]} | {r["left_reprojection_rms_px"]:.3f} | {r["right_reprojection_rms_px"]:.3f} | {r["stereo_rms_px"]:.3f} | {r["vertical_rms_px"]:.3f} | {r["baseline"]:.3f} |')
    photo_table=['| 번호 / UTC 촬영 ID 앞부분 | 원본 검출 L / R | 최종 역할 | 최종 수직 RMS px | 사진에서 확인한 내용 |',
                 '|---|---|---|---:|---|']
    for r in rows:
        value='—' if r['final_vertical_rms_px'] is None else f'{r["final_vertical_rms_px"]:.3f}'
        photo_table.append(f'| {r["n"]:02} / {r["pair_id"][:22]} | {r["original_left_detector"]} / {r["original_right_detector"]} | {r["group"]} | {value} | {r["notes"]} |')
    boot=load('bootstrap100.json');b=best['report'];h=best['held_out_evaluation']
    cv=[r for r in load('cross_validation_clean18_summary.json') if r['model']=='k1k2']
    filelink=lambda name:f'[{name}]({(OUTPUT/name).as_posix()})'
    codepath=(ROOT/'app/stereo/experiment_session02.py').as_posix()
    imagepath=(OUTPUT/'corners/09_corrected_pair.png').as_posix()
    constrained=load('experiments/14_bundle20_baseline95_freeK.json')
    text=f'''# VIDEX session_02 자율 Stereo Calibration 결과

2026-10-09, Asia/Seoul. 입력: 저장된 24쌍/48 JPEG, 9×6 내부 코너, 한 칸 22mm, LEFT cw90 / RIGHT ccw90, 보정 좌표 240×320.

**기존 사진으로 정상적인 내부 기하 보정을 복구했다. 선택 결과는 Baseline {b['baseline']:.3f}mm, Stereo RMS {b['stereo_rms_px']:.3f}px, 수직 RMS {b['vertical_rms_px']:.3f}px이다. 95mm를 강제로 넣지 않았다. 실제 거리의 절대 정확도는 기준 거리 데이터가 없어 미검증이다.**

선택 파일: {filelink('calibration_session02_best.json')}. 비교 전체: {filelink('experiment_summary.csv')}. 사진별 기록: {filelink('all24_photo_audit.csv')}.

## 1. 가장 유력한 기존 실패 원인

실제 원인은 두 가지가 동시에 발생한 것이다. 9·12·15·17번 LEFT와 19번 RIGHT는 classic fallback을 사용했고, 상대 카메라 SB와 180도 다른 인덱스 방향이었다. 코너 0과 53이 서로 반대 물리적 끝에 있었는데 코드는 같은 번호를 그대로 대응시켰다. 평면 체커보드는 반대로 번호를 붙여도 각 카메라 단독 보정은 잘 맞을 수 있지만, 두 카메라가 같은 물리적 점을 관측한다는 stereo 조건은 깨진다.

또한 cornerSubPix의 (11,11)은 11px 반경, 즉 23×23px 주변 창이다. 실제 격자 간격은 대체로 7~14px였고, fallback에서 창이 다른 코너를 포함해 코너 이동과 중복을 만들었다. 9번 LEFT 격자 호모그래피 RMS는 6.253px, 작은 창/강화 SB에서는 약 0.17~0.24px였다. 17번 LEFT는 6.891px에서 강화 SB 약 0.173px로 복구됐다. 12번은 상대적으로 이동이 작아도 인덱스 불일치만으로 stereo 정렬이 크게 망가졌다. 창 정의는 [OpenCV cornerSubPix 공식 문서](https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html)에서 확인했다.

원본 코너 좌표를 유지하고 순서만 수정한 대조 실험은 Baseline 876.6→95.6mm, Stereo RMS 14.52→2.27px로 개선됐다. 따라서 인덱스 불일치가 비정상 Baseline의 직접 원인이라는 증거가 강하다. 남은 2.27px는 훼손된 좌표를 재검출해야 해결됐다. 원본 코너 일부를 제거하는 것만으로도 14쌍 결과가 좋아지지만, 그 결과의 LEFT 왜곡 모델은 영상 가장자리에서 비단조 변환을 만들므로 최종 선택하지 않았다.

원본 JPEG는 320×240이고 회전은 각 알고리즘 입구에서 한 번만 적용된다. 저장 JPEG 자체를 이중 회전하는 문제는 확인되지 않았다. 현재 지정 회전 후 양쪽 장면은 같은 상하/좌우 방향이며, RIGHT만 수평 또는 수직 미러링하는 가설은 Stereo RMS 4.62/4.41px와 비정상 Baseline을 만들어 기각했다. 양쪽 공통 미러링과 물리적 L/R 이름 교환은 이 데이터만으로 구별할 수 없다.

## 2. 모든 사진에서 확인한 품질과 대응

48 JPEG를 전체 장면과 확대된 보드/번호 이미지로 직접 확인했다. 모아레와 밝기 띠가 있는 태블릿 화면 패턴, 약한 해상도, 좌우 노출/색 차이, 흐림, 작은 영상 가장자리 여백이 확인됐다. 손은 대체로 보드 외곽을 잡고 있으며, 확실히 보이지 않는 점을 생성해 채우지 않았다. 7번 RIGHT는 여러 열이 영상 밖으로 잘려 전체 54코너의 물리적 대응을 복구할 수 없다.

1번과 24번은 대비 또는 ROI 검색으로 자동 복구했다. 6번은 처음 잘림으로 분류했지만 확대검사에서 이를 수정했다. 네 극단 내부 코너의 대략적 위치로 영상을 전개하고 SB가 54개 실제 코너를 검출한 뒤 원래 픽셀 좌표로 역변환했다. 수동으로 예측한 54좌표 자체는 체커 교차 대비가 부족해 버렸다. 선택한 SB 좌표는 54/54 코너의 체커 교차 부호가 일치하고 호모그래피 RMS 0.185px이며, 검증용으로만 사용했다. 과정과 초기 좌표는 {filelink('salvage06_audit.json')}에 남겼다.

각 번호는 원본 pair 디렉터리의 시간순 정렬이며, ID는 UTC 이름이다. 코너 번호는 0~53이다.

{chr(10).join(photo_table)}

번호 9의 좌우 물리적 대응 복구 예시:

![9번 실제 픽셀의 수정된 0~53 코너]({imagepath})

원본 번호 이미지는 corners/NN_side_original.png, 복구 번호는 corners/NN_corrected_pair.png와 recovery/salvage 이름 이미지, 최종 확대 보드 시트는 board_detail_*.png에 있다. 초기 20쌍용 corrected 이미지는 그 시점 검출 상태를 보존하며, 1·6·24번의 최종 검출은 별도 recovery/salvage와 rectified 번호 이미지에서 확인한다.

## 3. 수행한 실험과 설정

독립된 Python/OpenCV 실험 모듈 [{Path(codepath).name}]({codepath})을 실행했다. 총 164개 보정/CV 결과 JSON과 100회 프레임 bootstrap 결과를 저장했다. 초기 SB, classic 초기 좌표, 반경 2/3/5/11px refinement, 강화된 SB, 2배 영상 SB를 비교했다. 실패한 사진에는 대비/감마/블러/ROI, 1/2/3배 탐색을 시도했고 마지막 6번은 영상 전개까지 수행했다.

왜곡은 full5, k1/k2+접선, k1/k2, k1, 0, 정사각 픽셀 제약을 비교했다. 각각 단독 K/D 추정 후 FIX_INTRINSIC 방식과 K/D 공동 최적화 방식을 비교했다. 원본 좌표 순서만 바꾸는 실험과, 코너를 실제로 재검출하는 실험을 분리했다. 평행 구도와 모든 사진에서 확인된 보드 긴 축 방향을 이용해 양 축을 함께 180도 뒤집었다. 이 방향 규칙은 이번 사진에 한정되며 임의의 촬영 자세에 일반 적용하는 코드로 간주하면 안 된다.

기존 max 재투영 1px, Stereo RMS 1.5px, 수직 RMS 1px 기준은 완화하지 않았다. 물리적 비교 기준은 사용자 제공 95mm를 사용했다. 각 실험은 설정, 사용/미사용 ID, K/D/R/T/Rectification, 이미지별 오차를 포함한다. 고정 내부 파라미터는 사진에서 추정한 K/D를 stereo 단계에서 고정한다는 뜻이며, 임의 숫자를 입력한 것이 아니다.

{chr(10).join(table)}

수직 RMS는 같은 240×320 출력, alpha=0에서 비교했다. 서로 다른 모델의 확대율이 차이를 만들 수 있어 가상 초점거리 240px로 정규화한 수직 오차도 각 평가에 기록했다. Stereo RMS는 각 2D 점의 유클리드 오차 RMS이며 좌우를 함께 집계한다. 미사용 사진에서는 카메라 파라미터를 고정하고 사진별 보드 포즈 6개 변수만 추정한다.

95mm 제약 실험은 T를 사후 스케일링하지 않았다. R, T 방향, 각 보드 포즈와 필요시 K/D를 재최적화했다. 18쌍 고정 K에서 오차는 오히려 0.208→0.211px로 증가했고, 자유 K에서도 0.205px로 이점이 작았다. 따라서 무제약 결과를 선택했다. 20쌍/95mm/자유 K 실행은 {constrained['optimizer']['evaluations']}회 제한까지 수렴하지 않았으며, 실패 상태를 그대로 기록하고 선택에서 제외했다.

## 4. 미사용 사진, 이상치, 안정성 검증

최종 학습은 18쌍이다. 1·6·24번은 최종 K/D/R/T 추정에 사용하지 않았다. 1·24번은 자동 복구, 6번은 시각적 초기화 후 SB 복구이다. 세 사진의 LEFT/RIGHT RMS는 {h['left_rms_px']:.3f}/{h['right_rms_px']:.3f}px, 공동 Stereo RMS {h['stereo_joint_pose_rms_px']:.3f}px, 수직 RMS **{h['vertical_rms_px']:.3f}px**이다. 수직 p95 {h['vertical_p95_px']:.3f}px, 최대 {h['vertical_max_px']:.3f}px이다. 자동 복구 2쌍만의 수직 RMS는 0.510px, 6번은 0.417px이다.

18쌍에서 일부를 매번 보정에 사용하지 않는 5-fold CV를 수행했다. 무작위 분할 수직 RMS {cv[0]['heldout_vertical_rms_px']:.3f}px, 시간 블록 분할 {cv[1]['heldout_vertical_rms_px']:.3f}px, 공동 Stereo RMS {cv[0]['heldout_joint_stereo_rms_px']:.3f}/{cv[1]['heldout_joint_stereo_rms_px']:.3f}px이다. 별도 13쌍 학습/5쌍 검증은 수직 RMS 0.287px였다. 최종 18쌍 파라미터가 이 5쌍도 다시 학습한 사실과, 검증 모델이 13쌍 모델이라는 사실을 구분했다.

전체 20쌍 CV와 18쌍 CV를 모두 저장했다. 복잡한 full5와 공동 최적화 모델 일부는 접지 않은 fold의 낮은 학습 오차에도 가장자리에서 비단조 왜곡을 만들었다. 선택한 k1/k2 고정 K 모델은 두 분할 방식의 모든 fold와 100회 bootstrap에서 이 문제가 없었다. 공식 문서도 비단조 렌즈 왜곡을 실패로 취급하도록 설명한다. [OpenCV 보정 모델과 한계](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).

100회 bootstrap Baseline 2.5~97.5 분위수는 {boot['baseline_percentiles_mm']['2.5']:.2f}~{boot['baseline_percentiles_mm']['97.5']:.2f}mm, 표준편차 {boot['baseline_std_mm']:.2f}mm였다. 이는 사진 구성에 대한 경험적 안정성이며 절대 거리 오차의 신뢰구간이 아니다.

11·22번은 재검출 후에도 최종 수직 RMS 1.334/1.181px로 남았다. 11번의 평균 수직 오프셋은 1.325px, 분산 성분의 표준편차는 0.149px로, 180도 인덱스 불일치처럼 큰 격자 전체 반전과는 다르다. 22번은 포즈 불일치와 잔여 변형이 함께 있다. 시간차/보드 움직임/화면 주사 영향이 유력하지만 이 JPEG만으로 원인을 단정할 수는 없다. 단독 재투영이 작다는 이유로 정상 pair로 처리하지 않았다.

메타데이터의 PC 수신 차이는 0~78ms이다. PC 시간이 같은 것은 같은 USB read 묶음으로 처리된 결과일 수도 있으므로 노출 동기화를 뜻하지 않는다. camera-local timestamp 차이는 약 -89.66~90.41ms이나, 공통 시계 검증이 없어 실제 노출 차이로 해석하지 않았다. frame_id도 서로 다른 스트림의 카운터이다.

이 검증은 하나의 기존 세션을 이용한 후향적 검증이다. 모델 선택과 이상치 판단에 같은 세션 정보를 이용했으므로 완전히 새로운 장면의 외부 검증이나 미래 성능 보증으로 표현하지 않는다.

## 5. 선택 파라미터와 Rectification

길이 단위는 mm, 변환은 X_right = R X_left + T이다. Baseline과 약 95mm의 차이는 **{95-b['baseline']:.3f}mm ({100*b['baseline_relative_difference']:.2f}%)**이다.

```text
K_left = {np.asarray(best['matrices']['K_left']).round(6)}
K_right = {np.asarray(best['matrices']['K_right']).round(6)}
D_left [k1,k2,p1,p2,k3] = {np.asarray(best['matrices']['D_left']).round(6)}
D_right = {np.asarray(best['matrices']['D_right']).round(6)}
R = {np.asarray(best['matrices']['R']).round(8)}
T_mm = {np.asarray(best['matrices']['T']).ravel().round(6)}
```

상대 회전은 {b['relative_rotation_degrees']:.3f}도이다. R의 det와 직교성을 확인했고, 저장 P/Q로 투영-복원한 정확한 3D 좌표의 최대 수치 오차는 {best['geometric_checks']['Q_exact_projection_roundtrip_max_mm']:.2e}mm이다. 실제 코너에서의 직접 삼각측량과 Q 복원 차이는 잔여 수직 오차의 영향까지 포함해 q_validation.json에 별도 기록했다.

alpha=0에서 두 ROI는 [0,0,240,320], remap 공통 유효 출력 면적은 99.996%이다. 하지만 출력 전체가 유효하다는 사실은 원래 시야를 모두 보존한다는 뜻이 아니다. 원본 픽셀 중 출력 안에 남는 비율은 LEFT 78.7%, RIGHT 65.4%이다. alpha=0.5는 공통 유효 출력 약 96.1%, 원본 시야 약 96.8%/84.3%를 보존한다. alpha=1은 원본을 거의 전부 보존하지만 공통 유효 출력은 약 74.3%이다. alpha=0.5나 1을 사용할 때는 R1/R2/P1/P2/Q를 함께 다시 생성해야 한다.

현재 입력 순서에서 T_x > 0이고 실제 대응의 rectified disparity는 음수이다. Q는 이 부호와 일관되고 삼각측량 깊이는 양수였다. 기본 양수 disparity만 검색하는 SGBM에 그대로 연결하면 실패할 수 있다. 물리적 좌우를 가림 실험으로 확인하고, 실제 데이터의 부호에 맞는 disparity 검색 범위 또는 스트림 순서를 선택해야 한다. 실험용 좌우 교환 결과는 같은 93.696mm와 같은 오차를 만들지만, 이를 물리적 정답이라고 임의로 채택하지 않았다.

## 6. 거리 측정에 사용할 수 있는가

**정렬과 거리 계산의 실험용 파라미터로 사용할 수 있다. 절대 거리 정확도가 검증된 실용 보정이라고 인증할 수는 없다.** 기준 깊이 측정값과 새로운 장면이 없고, 노출 동기화/실제 카메라 이름이 미검증이다. 실험 입력은 주로 약 0.5~0.8m의 체커보드이며, 먼 거리나 화면 가장자리의 일반 장면은 충분히 평가하지 않았다. 22mm는 사용자 제공 실측값으로 사용했으며 JPEG만으로 그 물리 길이를 검증할 수 없다.

보드 변 길이의 삼각측량 오차도 JSON에 저장했다. 이는 기하 모양의 자기 일관성 검사이다. 22mm가 이미 학습 스케일을 정했으므로 절대 거리의 독립 실측 검증으로 취급하지 않는다. 특히 낮은 해상도에서 작은 disparity 오차의 깊이 영향이 커진다. 선택 f/B에 대한 1차 근사로 1m에서 disparity 1px 오차는 약 29mm, 2m에서는 약 116mm 깊이 오차가 된다. 이 계산은 실제 SGBM 오차 측정값이 아니다.

## 7. 추가 촬영이 반드시 필요한가

현재 실패를 해결하고 정상적인 stereo 기하를 얻기 위해 전체 촬영을 다시 할 필요는 없다. 기존 데이터로 원인 분리, 좌표 복구, 유효한 R/T 및 내부 정렬 검증을 달성했다. **실제 거리 측정에 사용할지를 판정하려면 추가 실측 검증 사진은 필요하다.** 다음 조건이 구체적이다.

1. 먼저 각 렌즈를 따로 가려 물리적 LEFT/RIGHT 이름과 영상 미러링을 확인한다. 회전 설정은 유지하고 입력 원본에서 정확히 한 번 적용한다.
2. 보드를 고정한 채 각 자세를 최소 1초 이상 유지한 뒤 좌우 프레임을 선택한다. 가능하면 동기 트리거/공통 시각을 추가하고, 현재 타임스탬프 차이만으로 동기라고 판단하지 않는다.
3. 거리 검증은 실제 사용 범위를 포함한다. 예: 0.5/0.75/1.0/1.5/2.0m에서 각각 여러 정지 pair를 촬영하고, 렌즈 중심 기준의 측정 거리와 중앙/좌/우/상/하 표적 위치를 기록한다. 보고할 값이 광축 방향 Z인지 유클리드 거리인지도 일치시킨다. 이 사진은 재보정 학습에 넣기 전에 먼저 동결된 현재 JSON으로 평가한다.
4. 향후 보정 데이터를 늘릴 경우 roll만 바꾸지 말고 yaw/pitch를 각각 약 ±15~30도로 바꾸고, 두 카메라 모두 전체 9×6 격자와 외곽 여백이 보이게 한다. 기존에 부족했던 영상 가장자리 위치도 추가한다.
5. 인쇄한 무광 평판이나 고정된 표시 화면을 사용한다. 태블릿은 표시 배율을 고정하고 실제 가로/세로 한 칸이 22mm인지 확인하며, 반사/화면 주사 띠와 과노출을 줄인다. 가능한 경우 해상도를 높여 한 칸이 최소 약 15~25px로 보이게 하되, 해상도를 바꾸면 그 해상도에 맞는 파라미터와 검증을 별도로 수행한다.

## 재현과 보존

기존 JPEG 48개, manifest 24개, calibration_02.json의 SHA-256은 작업 전후 동일하다. {filelink('source_preservation.json')}의 preserved=true를 확인했다. Commit/Push는 수행하지 않았다. 실험 파일은 모두 이 별도 결과 디렉터리에 저장했다. 기존 calibration.py에는 실행 중 외부 변경이 생겼지만 이 작업은 해당 파일을 수정하지 않았으며 독립 fit 구현을 사용했다. 실행 버전과 현재 코드 해시는 runtime_manifest.json에 기록했다.

새 디렉터리에서 전체 재현하는 PowerShell 명령은 다음 순서이다. 감사 단계는 기존 출력 디렉터리를 덮어쓰지 않는다. 이후 단계는 새 디렉터리에 있는 해당 실험 파일만 생성/갱신한다.

```powershell
Set-Location C:\\videx_app\\videx
$pythonPath = '.\\app\\stereo\\.venv\\Scripts\\python.exe'
$experimentOutput = '.\\app\\stereo\\results\\session02_rerun_unique'
& $pythonPath -B -m app.stereo.experiment_session02 audit --output $experimentOutput
foreach ($stageName in @('suite','recover','cv','diagnostics','validation','constraints','cv18','stability','hypotheses','salvage06','finalize')) {{
    & $pythonPath -B -m app.stereo.experiment_session02 $stageName --output $experimentOutput
}}
```

既存のRAW JPEGを選択JSONで整列する使用例:

```powershell
& $pythonPath -B -m app.stereo.rectify --parameters .\\app\\stereo\\results\\autonomous_session02_20261009\\calibration_session02_best.json --left .\\app\\stereo\\data\\session_02\\pair_20261009T034257_267220Z_dd7f9658\\left.jpg --right .\\app\\stereo\\data\\session_02\\pair_20261009T034257_267220Z_dd7f9658\\right.jpg --output .\\app\\stereo\\results\\rectification_best_new
```

검증 예시: {filelink('rectified/01_rectified.png')}, {filelink('rectified/06_rectified.png')}, {filelink('rectified/24_rectified.png')}, 이상치 {filelink('rectified/22_rectified_numbered.png')}.
'''
    text=text.replace('既存のRAW JPEGを選択JSONで整列する使用例:', '기존 RAW JPEG를 선택한 JSON으로 정렬하는 사용 예:')
    text+='\n최종 검증: 기존 unittest 33개 통과, 선택 JSON의 공식 loader/print_report 통과, 164개 실험 행렬 모두 유한값, 원본 73개 파일 해시 동일. verification.json에 기록했다.\n'
    (OUTPUT/'report_ko.md').write_text(text,encoding='utf-8')
    preserved=hashes()==load('source_hashes_before.json')
    if not preserved: raise RuntimeError('Original data changed')
    print('Report saved:',OUTPUT/'report_ko.md','originals preserved:',preserved)


if __name__=='__main__':
    report()
