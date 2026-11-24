# Robot Action PBL

휴머노이드 로봇의 pick-and-place 동작 로그를 분석하는 머신러닝 팀 프로젝트입니다.

## 프로젝트 목표

- 에피소드의 성공·실패 분류
- 해석 가능한 동작 유형 군집화
- 수행 품질 점수 설계

데이터 로더, 단계 분할과 단계 특징, 분류(5차 모델까지), 군집(PR #3, v2), 품질 점수(PR #4, v2)와 이 셋을 한 번에 실행하는 데모 스크립트를 구현했습니다. 영상은 라벨 확인과 실패 유형 설명의 근거로만 씁니다.

## 환경 설정

Python 3.11 이상을 사용합니다. 아래 명령은 Windows PowerShell 기준입니다.

```powershell
git clone https://github.com/A1-devLab/robot-action-pbl.git
cd robot-action-pbl
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`py` 명령이 없는 환경에서는 설치한 Python 실행 파일 경로를 사용하세요. `requirements.txt`는 허용 버전 범위이며, 검증에 사용한 정확한 환경은 `requirements-lock.txt`에 기록합니다.

## 데이터 배치

제공받은 데이터의 `success`, `fail` 폴더를 아래와 같이 배치합니다. 압축 해제 과정에서 생긴 상위 폴더는 복사하지 않습니다.

```text
data/
├── success/
│   ├── episode_000002.csv
│   └── ...
└── fail/
    ├── episode_000001.csv
    └── ...
```

제공된 1차 데이터는 성공 27개, 실패 21개로 총 48개 에피소드입니다. 원본 CSV 48개를 `data/success`, `data/fail`에 포함합니다. 저장소를 복제하면 함께 내려받으며, 원본 데이터는 수정하지 않습니다.

## 폴더 구조

```text
robot-action-pbl/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # 원본 CSV 48개
├── notebooks/
│   └── 01_eda.ipynb       # 데이터 검사·탐색·분류 실행 노트북
├── src/
│   ├── data.py            # CSV 로딩·검사 (라벨 없는 폴더는 load_folder)
│   ├── eda.py             # 성공·실패 관절 시계열 비교
│   ├── features.py        # 1·2차 통계 특징
│   ├── train.py           # 1·2차 분류 학습·검증
│   ├── phases.py          # 잡기·옮기기·놓기 단계 분할
│   ├── features_phase.py  # 단계 특징 10개
│   ├── cluster.py         # 과제 3: 군집
│   ├── score.py           # 과제 4: 품질 점수 (PR #4, v1)
│   ├── classify.py        # 과제 2: 점수 기반 분류기 (4차)
│   ├── features_release.py # 놓는 순간 특징 (엄지 펴는 속도)
│   ├── score_v2.py        # 과제 4·2: 품질 점수 v2와 5차 분류기
│   ├── cluster_v2.py      # 과제 3: 군집 v2 (옮기기·매끄러움·놓기)
│   └── demo.py            # 데모데이: 분류·군집·점수 한 번에 실행
├── experiments/           # 모델 비교 실험·영상 근거 그림 스크립트
├── tests/
└── reports/               # 분석 결과·그림
```

## 실행

레포 루트에서 실행합니다. `--data-dir`를 생략하면 `data/`를 사용하며, 외부 데이터 폴더 경로를 지정할 수도 있습니다. 원본 CSV는 수정하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m src.data --data-dir data
.\.venv\Scripts\python.exe -m src.eda --data-dir data
.\.venv\Scripts\python.exe -m src.train --data-dir data --seed 42
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

탐색 노트북은 다음 명령으로 엽니다.

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks/01_eda.ipynb
```

### 로더와 결과 파일

`load_dataset(path)`는 각 에피소드의 ID, 라벨, 경로, 수치형 프레임, 검사 요약을 반환합니다. 빈 파일, 잘못된 열 구성, 빈 값·무한대·문자열, 시간 역전, 프레임 누락·중복, 파일명과 내부 ID 불일치, 에피소드 ID 중복을 검사합니다. 기본 동작은 오류가 있는 데이터를 거부합니다. CLI는 검사 가능한 오류를 요약표에 남기고 오류가 있으면 종료 코드 1을 반환합니다. 폴더가 없거나 CSV를 읽을 수 없으면 즉시 중단합니다.

- `reports/episode_summary.csv`: 파일별 길이·결측값·검사 결과 (로컬 경로 포함, Git 제외)
- `reports/eda/state_by_class.png`, `action_by_class.png`: 26개 관절의 클래스별 시계열
- `reports/baseline/features.csv`: ID·라벨과 208개 특징. 재사용 시 ID·라벨은 X에서 제외해야 합니다.
- `reports/baseline/oof_predictions.csv`: 각 에피소드의 평가 fold, 정답, 예측, 성공 확률
- `reports/baseline/metrics.json`, `confusion_matrix.png`: 교차검증 지표·설정·혼동행렬

`extract_features`는 모델 입력 X, 정답 y, 관리 정보 metadata를 별도로 반환합니다. 52개 state/action 채널마다 평균, 표준편차(ddof=0), 범위(max-min), 끝-시작 차이를 계산합니다. ID·경로·시간·수집 인덱스·라벨은 X에 들어가지 않습니다.

### 실험 보고서

사용 알고리즘, 특징 추출, 교차검증 설정, 성능, 혼동행렬과 한계는 [1차 baseline 실험 보고서](reports/baseline/README.md)에 기록합니다.

실행 환경 전체 버전은 `requirements-lock.txt`에 기록합니다. 동일한 Windows/Python 3.12 환경에서는 `pip install -r requirements-lock.txt`로 재현할 수 있습니다. 다른 OS에서는 우선 `requirements.txt`를 사용하세요.

검증 구현 참고: [scikit-learn 데이터 누출 방지](https://scikit-learn.org/stable/common_pitfalls.html), [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

### 품질 점수 (과제 4)

`src/score.py`는 에피소드 하나의 수행 품질을 0~100점으로 계산합니다. 공식과 기준 통계는 고정되어 있어 새 CSV에도 그대로 적용됩니다.

```text
z_i   = d_i × (x_i − mean_i) / std_i,  [-3, 3]으로 자름
s     = (z_1 + z_2 + z_3 + z_4) / 4          (동일 가중치)
score = 100 × Φ(s / std_s)                   (Φ: 표준정규 누적분포)
```

| i | 입력 | 방향 d | 의미 | 단일 AUC |
|---|---|---|---|---|
| 1 | `delta_hold_l_sh_r` | −1 | 옮기는 동안 어깨 roll이 안쪽으로 크게 돌아감 | 0.19 |
| 2 | `delta_hold_l_el_p` | +1 | 옮기는 동안 팔꿈치를 상자 쪽으로 폄 | 0.76 |
| 3 | `rev_elp_hold` | −1 | 옮기는 동안 팔꿈치 방향 전환 횟수 (망설임) | 0.26 |
| 4 | `t_retract` | −1 | 놓은 뒤 끝날 때까지 걸린 시간 (s) | 0.27 |

- mean, std, std_s는 1차 데이터 48개에서 계산해 `reports/score/params.json`에 저장합니다. 50점은 기준 데이터의 평균, 84점은 평균보다 1표준편차 좋은 동작입니다.
- 출력에는 총점과 함께 하위 점수 `sub_carry`(1, 2), `sub_smooth`(3), `sub_exit`(4)가 있어 점수의 이유를 설명할 수 있습니다.
- 손이 한 번도 닫히지 않은 에피소드는 0점(`no_grasp`), 읽을 수 없는 파일은 점수 없이 `invalid`로 표시합니다.
- 입력 선택 근거: 성공·실패 분리력이 큰 특징 중 서로 겹치지 않는 것만 골랐습니다. `pose_grasp_l_sh_r`은 1번과 상관이 −0.90이라 제외했습니다. 데이터가 48개뿐이라 학습한 로지스틱 가중치보다 동일 가중치가 교차검증에서 더 좋았습니다.

검증 결과 (`experiments/validate_score.py`, 5-fold × 시드 10개, 기준 통계는 학습 fold에서만 계산):

| 항목 | 동일 가중치 | 로지스틱 가중치 |
|---|---|---|
| 교차검증 AUC | 0.846 ± 0.006 | 0.799 ± 0.020 |
| AUC, 에피소드 1~24 | 0.733 | 0.667 |
| AUC, 에피소드 25~48 | 0.935 | 0.833 |

- 평균 점수는 성공 66.7점, 실패 32.4점입니다.
- 점수와 에피소드 번호의 순위 상관은 0.72입니다. 뒤로 갈수록 성공률이 오르므로(1~24번 38%, 25~48번 75%) 조작 숙련도가 실제 품질에 반영된 것으로 보입니다. 다만 에피소드 번호만으로 얻는 AUC는 0.74이고, 점수는 앞뒤 절반 안에서도 성공과 실패를 구분하므로 녹화 순서만 따라가는 지표는 아닙니다.
- 입력 4개는 48개 전체의 단일 AUC를 보고 골랐기 때문에 교차검증 수치는 약간 낙관적입니다.

새 데이터 채점 (데모):

```powershell
# 라벨 없는 CSV 폴더 또는 success/, fail/ 폴더 모두 가능, 파일 이름 제한 없음
.\.venv\Scripts\python.exe -m src.score --data-dir <새 데이터 폴더> --output reports/score/demo_scores.csv
# 기준 통계 재계산 (1차 데이터 기준, 보통 필요 없음)
.\.venv\Scripts\python.exe -m src.score --data-dir data --fit
.\.venv\Scripts\python.exe -m experiments.validate_score
```

### 분류 4차: 품질 점수 분류기 (과제 2)

`src/classify.py`는 품질 점수가 50점 이상이면 성공, 미만이면 실패로 판정합니다. 50점은 기준 데이터 48개의 평균이며, 임계값은 학습하지 않습니다. 분류와 품질 점수가 같은 공식이므로 판정 이유를 하위 점수로 설명할 수 있습니다.

같은 조건에서 비교한 결과입니다. A는 5-fold × 시드 10개 반복 교차검증, B는 에피소드 번호순 8개씩 6블록을 한 블록씩 빼고 학습한 결과입니다. 입력 선택과 임계값은 학습 fold에서만 정했습니다.

| 모델 | A 정확도 | A AUC | B 정확도 | B AUC |
|---|---:|---:|---:|---:|
| 3차: 단계 특징 10개 + LogReg | 0.667 | 0.721 | 0.688 | 0.695 |
| **4차: 점수 ≥ 50 (입력을 학습 fold에서 다시 선택)** | **0.696** | **0.810** | **0.708** | **0.787** |
| 궤적 8×40 + LogReg | 0.719 | 0.754 | 0.729 | 0.772 |
| ROCKET 2,000 커널 + 릿지 | 0.650 | 0.662 | 0.563 | 0.571 |

- 정확도 차이는 48개의 표준오차(약 0.066) 안이고, 분명한 개선은 AUC입니다.
- 임계값을 학습 fold에서 튜닝하면 A 정확도는 0.746으로 오르지만 B에서는 0.667로 떨어져 고정값을 씁니다.
- 자세한 설정·해석·한계: [4차 실험 보고서](reports/fourth_model/README.md)

```powershell
.\.venv\Scripts\python.exe -m experiments.fourth_model_score_classifier
```

### 영상 근거: 라벨 확인과 실패 유형

1차 데이터 영상 48개는 모두 CSV와 프레임 단위로 정렬되어 있어(CSV i번째 줄 = 영상 i번째 프레임), 로그로 찾은 순간의 장면을 바로 볼 수 있습니다. 과제 공지에 따라 영상은 모델 입력에 쓰지 않고 근거로만 씁니다.

- 라벨 48개 모두 마지막 장면과 일치합니다.
- 실패는 두 유형입니다.
  - **A형 (14개, 1~28번)**: 높은 곳에서 또는 기울여 놓아 넘어짐
  - **B형 (7개, 19~48번)**: 바닥에 세웠지만 손을 펴는 0.1~0.3초 사이에 손가락이 밀어 넘어뜨림
- 자세한 내용·그림·재생성 방법: [영상 근거](reports/video_evidence/README.md)

### 5차 모델: 점수 v2, 분류기 v2, 군집 v2 (데모 기본값)

B형 실패는 손을 펴는 순간에 일어나는데, 4차까지의 특징은 그 순간을 보지 못합니다. 그래서 그 순간을 재는 로그 특징을 추가했습니다.

- **새 특징** `thumb_open_02s` (`src/features_release.py`): 손가락이 펴지기 시작한 뒤 0.2초 동안 엄지가 벌어진 양
- **점수 v2** (`src/score_v2.py`): PR #4 점수의 `t_retract`(놓은 뒤 복귀 시간)를 `thumb_open_02s`로 교체. 하위 점수는 `sub_carry`, `sub_smooth`, `sub_release`
- **분류기 v2**: 점수 v2 ≥ 50 → 성공
- **군집 v2** (`src/cluster_v2.py`): 하위 점수 3개로 K-평균 k=3
  - C0 깔끔하게 옮기고 부드럽게 놓음 (19개, 성공률 84%)
  - C1 잘 옮기지만 손을 급하게 폄 (18개, 50%)
  - C2 머뭇거리며 불안정하게 내려놓음 (11개, 18%)

| 모델 | A 정확도 | A AUC | B 정확도 | B AUC |
|---|---:|---:|---:|---:|
| 4차 (입력을 학습 fold에서 다시 선택, 공정한 추정) | 0.696 | 0.810 | 0.708 | 0.787 |
| 5차 v2 (낙관적: 특징을 영상·라벨을 보고 고름) | 0.829 | 0.884 | 0.833 | 0.875 |

- v2 수치는 낙관적입니다. 영상 없이 데이터만으로 특징을 고르게 하면 개선이 없었습니다(정확도 0.660). 새 데이터에서 v1과 v2를 다시 비교해야 합니다.
- 군집 v2는 PR #3보다 안정적이고(부트스트랩 ARI 0.53 → 0.78), 녹화 순서에 덜 묶여 있습니다(에피소드 번호 η² 0.72 → 0.25).
- B형 실패 7개 중 4개는 v2도 놓칩니다.
- 자세한 비교: [5차 실험 보고서](reports/fifth_model/README.md)

```powershell
.\.venv\Scripts\python.exe -m experiments.fifth_model_release
```

### 데모 실행 (데모데이)

`src/demo.py`는 지정한 폴더 바로 아래의 모든 CSV(하위 폴더는 `success/`, `fail/`만 인식)에 분류(과제 2), 군집(과제 3), 품질 점수(과제 4)를 한 번에 적용합니다.

```powershell
# 라벨 없는 CSV 폴더 (파일 이름 제한 없음), 기본값 v2
.\.venv\Scripts\python.exe -m src.demo --data-dir <테스트 데이터 폴더>
# 4차 모델·PR #4 점수·PR #3 군집으로 실행
.\.venv\Scripts\python.exe -m src.demo --data-dir <테스트 데이터 폴더> --version v1
# success/, fail/ 폴더가 있으면 정확도·혼동행렬도 출력
.\.venv\Scripts\python.exe -m src.demo --data-dir data
```

- 군집 모델은 실행할 때 `data/`의 기준 에피소드 48개로 다시 학습합니다(결과 고정, 약 1초). 저장된 모델 파일이 필요 없습니다.
- 품질 점수는 고정 기준 통계를 씁니다: v2는 `reports/score_v2/params.json`, v1은 `reports/score/params.json`.
- 결과는 `reports/demo/results.csv`에 저장됩니다(`--output`으로 변경 가능, Git 제외).
  - 공통 열: `status`, `prediction`, `score`, `cluster`, `cluster_name`, 입력 특징
  - 하위 점수 열: v2는 `sub_carry`, `sub_smooth`, `sub_release`, v1은 `sub_carry`, `sub_smooth`, `sub_exit`
- `status`는 `ok`, `no_grasp`(손이 닫히지 않음 → 실패, 0점), `invalid`(파일 검사 실패), `error`(그 파일만 처리 실패)입니다. 파일 하나가 잘못돼도 나머지는 계속 처리합니다.
- `data/`를 쓸 수 없으면 경고를 출력하고 군집 없이 분류·점수만 계산합니다.

## 공통 실험 원칙

- 학습·평가 데이터는 에피소드 단위로 분리합니다.
- 에피소드 ID, 파일 경로, 폴더 이름을 모델 입력 특징에 포함하지 않습니다.
- 정규화와 특징 선택은 각 교차검증의 학습 구간에서만 학습합니다.
- 실험 시 데이터 버전, 특징 구성, 난수 시드, 검증 방식과 결과를 기록합니다.

## 다음 작업

1. 영상에서 얻은 통찰로 특징을 정한 것(`thumb_open_02s`)을 보고서에 어떻게 쓸지 교수님·TA에게 확인
2. 추가 데이터가 오면 학습에 넣기 전에 v1과 v2를 먼저 그 데이터로 비교
3. B형 실패(손을 펴는 순간 넘어뜨림) 중 v2도 놓치는 21, 25, 44, 48번 분석
4. 기구학 정보(URDF 또는 링크 길이)를 받으면 놓는 순간의 손 높이·기울기 특징 추가
5. 데모 리허설: 라벨 없는 폴더, 바뀐 파일 이름, 깨진 파일로 `src.demo` 실행 확인 (Windows)
