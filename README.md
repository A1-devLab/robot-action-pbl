# Robot Action PBL

휴머노이드 로봇의 pick-and-place 동작 로그를 분석하는 머신러닝 팀 프로젝트입니다.

## 프로젝트 목표

- 에피소드의 성공·실패 분류
- 해석 가능한 동작 유형 군집화
- 수행 품질 점수 설계

공통 데이터 로더, 탐색 그래프, 기본 특징 추출과 로지스틱 회귀 교차검증을 구현했습니다. 군집화와 품질 점수는 이후 단계입니다.

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
│   ├── data.py            # CSV 로딩·검사
│   ├── eda.py             # 성공·실패 관절 시계열 비교
│   ├── features.py        # 에피소드별 특징 추출
│   └── train.py           # 분류 학습·검증
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

## 공통 실험 원칙

- 학습·평가 데이터는 에피소드 단위로 분리합니다.
- 에피소드 ID, 파일 경로, 폴더 이름을 모델 입력 특징에 포함하지 않습니다.
- 정규화와 특징 선택은 각 교차검증의 학습 구간에서만 학습합니다.
- 실험 시 데이터 버전, 특징 구성, 난수 시드, 검증 방식과 결과를 기록합니다.

## 다음 작업

1. 오류 분류된 에피소드와 종료 구간의 움직임 분석
2. 특징 수를 줄이거나 구간별 특징을 추가해 기준 모델과 비교
3. 분류기, 군집, 품질 점수를 하나의 데모 스크립트로 묶기
