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

`py` 명령이 없는 환경에서는 설치한 Python 실행 파일 경로를 사용하세요. 의존성 버전은 초기 범위이며, 모델 구현과 실행 검증 후 팀 공통 버전을 확정합니다.

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

제공된 1차 데이터는 성공 27개, 실패 21개로 총 48개 에피소드입니다. 원본 데이터는 수정하지 않으며 Git에 올리지 않습니다. 데이터는 별도로 전달받아 배치해야 합니다.

## 폴더 구조

```text
robot-action-pbl/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # 원본 데이터 (Git 제외)
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

### 검증 방식과 첫 결과

에피소드당 한 행으로 5겹 층화 교차검증(shuffle=True, seed=42)을 수행합니다. 매 fold마다 상수 특징 제거 → 표준화 → 로지스틱 회귀(C=1, max_iter=5000)를 새로 학습합니다. 특징 제거와 표준화는 학습 데이터에만 fit합니다. 각 에피소드는 자신을 학습하지 않은 모델로 한 번씩 평가하며, 전체 out-of-fold 예측으로 지표를 계산합니다. 하이퍼파라미터 탐색과 최종 전체 데이터 모델 저장은 포함하지 않습니다.

1차 데이터 48개 기준 초기 결과:

| 지표 | 값 |
|---|---:|
| 정확도 | 0.5625 |
| 균형 정확도 | 0.5423 |
| 성공 클래스 F1 | 0.6441 |
| ROC AUC | 0.5785 |
| 다수 클래스 기준 정확도 | 0.5625 |

혼동행렬은 행=정답, 열=예측이며 순서는 fail, success입니다: `[[8, 13], [8, 19]]`. 다수 클래스 기준을 정확도에서 넘지 못했으므로 현재 모델은 성능 개선을 위한 기준점입니다. 48개 표본의 단일 5겹 분할 결과이며, 독립 테스트 성능을 보장하지 않습니다.

탐색 그래프는 각 에피소드를 0~1 상대 시간으로 보간해 표시합니다. 얇은 선은 개별 에피소드, 굵은 선은 클래스 평균입니다. 이 정렬은 실제 작업 단계의 일치를 보장하지 않으며 원본 단위도 별도 확인이 필요합니다. 전체 라벨을 본 탐색 결과를 바탕으로 특징을 반복 조정하면 같은 검증 점수가 낙관적으로 바뀔 수 있습니다. 이후 비교에서는 별도 테스트 또는 중첩 검증을 고려합니다.

실행 환경 전체 버전은 `requirements-lock.txt`에 기록합니다. 동일한 Windows/Python 3.12 환경에서는 `pip install -r requirements-lock.txt`로 재현할 수 있습니다. 다른 OS에서는 우선 `requirements.txt`를 사용하세요.

검증 구현 참고: [scikit-learn 데이터 누출 방지](https://scikit-learn.org/stable/common_pitfalls.html), [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

## 공통 실험 원칙

- 학습·평가 데이터는 에피소드 단위로 분리합니다.
- 에피소드 ID, 파일 경로, 폴더 이름을 모델 입력 특징에 포함하지 않습니다.
- 정규화와 특징 선택은 각 교차검증의 학습 구간에서만 학습합니다.
- 실험 시 데이터 버전, 특징 구성, 난수 시드, 검증 방식과 결과를 기록합니다.

## 다음 작업

1. 오류 분류된 에피소드와 종료 구간의 움직임 분석
2. 특징 수를 줄이거나 구간별 특징을 추가해 기준 모델과 비교
3. 해석 가능한 군집화와 수행 품질 점수 설계
