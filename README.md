# 안토시아닌 탈색 시뮬레이터

pH, 초기 흡광도, 겉보기 속도상수, H₂O₂ 설정 농도, 반응 모형을 바꾸며 시간-흡광도 곡선을 탐색하는 Streamlit 웹앱입니다. CSV 업로드를 통한 실험 데이터 시각화 기능도 포함합니다.

## 1. 로컬 실행

Python 3.10 이상을 권장합니다.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2. GitHub에 올리기

1. GitHub에서 새 저장소(repository)를 만듭니다.
2. 이 폴더의 `app.py`, `requirements.txt`, `README.md`를 저장소에 업로드합니다.
3. 저장소가 Public이면 Streamlit Community Cloud에서 배포할 수 있습니다.

## 3. Streamlit 배포

1. https://share.streamlit.io/ 에 접속해 GitHub 계정으로 로그인합니다.
2. **Create app**을 선택합니다.
3. 저장소, 브랜치(`main`), 메인 파일 경로(`app.py`)를 지정합니다.
4. **Deploy**를 누릅니다.

## CSV 형식

필수 열 이름은 다음과 같습니다.

- `time`: 측정 시간
- `pH`: 해당 시료의 pH
- `group`: 처리군 이름(예: `H2O`, `H2O2`)
- `absorbance`: 흡광도

예시:

```csv
time,pH,group,absorbance
0,4,H2O,0.82
1,4,H2O,0.81
0,4,H2O2,0.80
1,4,H2O2,0.74
```

## 과학적 한계

- 앱의 시뮬레이션은 단순한 0차·1차·2차 수학 모형을 사용합니다.
- pH 4, 7, 10의 겉보기 속도상수는 사용자가 지정하고, 중간 pH는 구간별 직선 보간합니다.
- H₂O₂ 농도와 속도상수의 관계는 사용자가 정한 지수식으로 표현한 가정입니다. 실제 속도법칙으로 검증된 식이 아닙니다.
- pH에 따라 안토시아닌의 화학종, 흡수 파장, 몰흡광계수가 달라질 수 있습니다. 따라서 흡광도 감소를 곧바로 농도 감소로 단정하지 않습니다.
- 보고서에는 시뮬레이션 결과와 실험 측정 결과를 명확히 구분해 제시해야 합니다.
