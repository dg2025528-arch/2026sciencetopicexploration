import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from bio_connection_tab import render_bio_connection_tab


APP_TITLE = (
    "pH 조건에 따른 안토시아닌의 분자 구조 변화와 "
    "과산화수소에 의한 산화 반응 속도 시뮬레이션"
)

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg, #071225, #0b1830 55%, #101f38
        );
        color: #e8f0fa;
    }

    [data-testid="stSidebar"] {
        background: #0a162b;
        border-right: 1px solid #294260;
    }

    h1, h2, h3 {
        color: #f1f6ff !important;
    }

    h1 {
        font-size: 2rem !important;
        line-height: 1.35 !important;
    }

    div[data-testid="stMetric"] {
        background: #10233e;
        border: 1px solid #294260;
        padding: 12px;
        border-radius: 10px;
        min-width: 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        flex-wrap: nowrap;
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #10233e;
        color: #CBD5E1 !important;
        border-radius: 8px 8px 0 0;
        white-space: nowrap;
        flex-shrink: 0;
        padding: 10px 14px;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        background: #234766 !important;
        font-weight: 700;
    }

    .small,
    [data-testid="stCaptionContainer"] {
        color: #C4D2E3 !important;
        font-size: .88rem;
        line-height: 1.6;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
    }

    [data-testid="stMarkdownContainer"] a {
        color: #8DCEFF;
        text-decoration: underline;
    }

    [data-testid="stDownloadButton"] button,
    [data-testid="stButton"] button {
        min-height: 44px;
    }

    div[role="radiogroup"] {
        flex-wrap: wrap;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.8rem 2rem;
        }

        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.4rem !important; }
        h3 { font-size: 1.15rem !important; }

        .stTabs [data-baseweb="tab"] {
            padding: 10px 12px;
            font-size: .9rem;
        }

        [data-testid="stDownloadButton"] button,
        [data-testid="stButton"] button {
            width: 100%;
            white-space: normal;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 제목 및 화면 옵션
# =========================================================
st.title(APP_TITLE)
st.caption(
    "pH-dependent anthocyanin stability · "
    "H₂O₂ treatment · absorbance kinetics"
)

mobile_view = st.checkbox(
    "📱 모바일 보기",
    value=False,
    key="mobile_view",
    help=(
        "그래프를 세로로 넓히고 요약 지표를 세로로 표시합니다. "
        "휴대폰에서는 체크하는 것을 권장합니다."
    ),
)

st.caption(
    "휴대폰에서는 왼쪽 위 사이드바 열기 버튼으로 실험 조건을 "
    "변경하세요. 탭 목록은 좌우로 밀어서 볼 수 있습니다."
)

st.info(
    "시뮬레이션은 실제 화학 반응을 예측하거나 검증한 모델이 아니라 "
    "분석 방법을 학습하기 위한 가정 모델입니다. "
    "실제 측정 데이터와 구분하세요."
)


# =========================================================
# 내장 실험 데이터
# =========================================================
DATA_DIR = Path(__file__).parent / "data"

BUILTIN_FILES = [
    "pH4withH2O2_filled_v2.csv",
    "pH7_H2O2_processed.csv",
    "pH10withH2O2_filled.csv",
]


# =========================================================
# 공통 함수
# =========================================================
def ph_color(pH, fade):
    anchors = [
        (4, (183, 55, 83)),
        (7, (61, 105, 170)),
        (10, (82, 142, 102)),
    ]

    a, b = (
        (anchors[0], anchors[1])
        if pH <= 7
        else (anchors[1], anchors[2])
    )

    x = np.clip((pH - a[0]) / (b[0] - a[0]), 0, 1)
    fade = float(np.clip(fade, 0, 1))

    color = [
        int(a[1][i] * (1 - x) + b[1][i] * x)
        for i in range(3)
    ]
    color = [
        int(component * (1 - fade) + 225 * fade)
        for component in color
    ]

    return "#{:02x}{:02x}{:02x}".format(*color)


def prepare(df):
    d = df.copy()

    for column in ["time", "pH", "absorbance"]:
        d[column] = pd.to_numeric(d[column], errors="coerce")
        d[column] = d[column].replace([np.inf, -np.inf], np.nan)

    d["group"] = d["group"].astype("string").str.strip()
    d["group"] = d["group"].replace("", pd.NA)

    d = (
        d.dropna(subset=["time", "pH", "group", "absorbance"])
        .sort_values(["pH", "group", "time"], kind="stable")
        .reset_index(drop=True)
    )

    if d.empty:
        raise ValueError("정리 후 유효한 데이터가 없습니다.")

    d["A_norm"] = d.groupby(["pH", "group"])["absorbance"].transform(
        lambda s: s / s.iloc[0] if s.iloc[0] != 0 else np.nan
    )

    return d


@st.cache_data
def load_builtin():
    """data 폴더의 실험 CSV를 읽어 하나로 합침."""
    frames = []

    for name in BUILTIN_FILES:
        path = DATA_DIR / name
        frames.append(pd.read_csv(path, encoding="utf-8-sig"))

    raw = pd.concat(frames, ignore_index=True)
    cleaned = prepare(raw)

    quality = {
        "original_rows": len(raw),
        "missing_rows": int(raw.isna().any(axis=1).sum()),
        "removed_rows": len(raw) - len(cleaned),
    }

    return cleaned, quality


def model_fit(t, a):
    """변환 선형회귀 방식의 모형 비교."""
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)

    if len(a) < 3 or len(np.unique(t)) < 3:
        return pd.DataFrame()

    models = []

    p = np.polyfit(t, a, 1)
    models.append(
        ("0차 선형", np.polyval(p, t), -p[0], "A/s", "A=b−kt")
    )

    if np.all(a > 0):
        p = np.polyfit(t, np.log(a), 1)

        with np.errstate(over="ignore", invalid="ignore"):
            predicted = np.exp(np.polyval(p, t))

        models.append(
            ("1차 지수형", predicted, -p[0], "s⁻¹", "A=B·exp(−kt)")
        )

        p = np.polyfit(t, 1 / a, 1)
        denominator = np.polyval(p, t)

        if np.all(denominator > 0):
            models.append(
                (
                    "2차 역수형",
                    1 / denominator,
                    p[0],
                    "A⁻¹·s⁻¹",
                    "1/A=b+kt",
                )
            )

    ss = np.sum((a - a.mean()) ** 2)
    rows = []

    for name, predicted, k, unit, equation in models:
        if not np.all(np.isfinite(predicted)):
            continue

        error = a - predicted
        rmse = np.sqrt(np.mean(error ** 2))
        r2 = 1 - np.sum(error ** 2) / ss if ss > 0 else np.nan

        rows.append(
            {
                "모형": name,
                "겉보기 k": float(k),
                "k 단위": unit,
                "R²": r2,
                "RMSE": rmse,
                "식": equation,
            }
        )

    return pd.DataFrame(rows)


def make_figure():
    size = (5.2, 4.5) if mobile_view else (8, 3.8)
    fig, ax = plt.subplots(figsize=size)

    font_size = 12 if mobile_view else 11
    ax.tick_params(labelsize=font_size)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    return fig, ax


def show_figure(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def add_legend(ax):
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            fontsize=9,
            loc="best",
            ncol=1 if mobile_view else 2,
        )


def show_metrics(items):
    if mobile_view:
        for label, value in items:
            st.metric(label, value)
    else:
        columns = st.columns(len(items))
        for column, (label, value) in zip(columns, items):
            column.metric(label, value)


def plot_data(d, normalized=False, reference=False, color_source=None):
    """d의 조건별 곡선을 그림. color_source는 pH별 색 고정용 전체 데이터."""
    fig, ax = make_figure()

    y_column = "A_norm" if normalized else "absorbance"
    source = color_source if color_source is not None else d
    ph_values = sorted(source["pH"].unique())
    mapping = []

    for index, ((p, group), series) in enumerate(
        d.groupby(["pH", "group"]), start=1
    ):
        curve_id = f"G{index}"
        color = f"C{ph_values.index(p) % 10}"

        ax.plot(
            series["time"],
            series[y_column],
            marker=".",
            markersize=3,
            linewidth=1.7,
            color=color,
            label=f"{curve_id} | pH {p:g}",
        )

        mapping.append(
            {"그래프 ID": curve_id, "pH": p, "group": group}
        )

    if reference:
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(
        "Relative absorbance, A/A0" if normalized else "Absorbance, A"
    )

    add_legend(ax)
    show_figure(fig)

    return pd.DataFrame(mapping)


# =========================================================
# 사이드바
# =========================================================
st.sidebar.header("실험 조건")

ph = st.sidebar.slider("pH", 4.0, 10.0, 7.0, 0.1)

buffer = st.sidebar.number_input(
    "완충용액 부피 (mL)", 1.0, 500.0, 45.0, 1.0
)

extract = st.sidebar.number_input(
    "안토시아닌 추출액 부피 (mL)", 0.1, 100.0, 5.0, 0.5
)

treated = st.sidebar.toggle("H₂O₂ 처리군", value=True)

duration = st.sidebar.slider("관찰 시간 (초)", 30, 3600, 300, 30)

st.sidebar.markdown("**온도: 25 °C 고정**")
st.sidebar.caption(
    "속도상수 직접 설정 기능은 없습니다. "
    "시뮬레이션의 가정값과 실험 데이터에서 추정한 값은 구분합니다."
)
st.sidebar.caption(
    "현재 시뮬레이션에서 용액 부피는 표시·기록용이며, "
    "희석 효과와 H₂O₂ 첨가 부피는 계산에 반영하지 않습니다."
)


# =========================================================
# 탭
# =========================================================
tabs = st.tabs(
    ["시뮬레이션", "분석", "비교", "관련 개념", "생명과학과 연결", "저장"]
)


# =========================================================
# 1. 시뮬레이션
# =========================================================
with tabs[0]:
    st.subheader("화학 실험 시뮬레이션")

    show_metrics(
        [
            ("전체 용액 부피", f"{buffer + extract:.1f} mL"),
            ("관찰 시간", f"{duration} s"),
            ("온도", "25 °C"),
        ]
    )

    st.caption(
        "비커는 고정된 높이의 도식입니다. "
        "색은 정성적으로 표현했고, 정량 분석은 흡광도 데이터로만 했습니다."
    )

    A0 = 0.65 * (0.75 + 0.25 * np.exp(-((ph - 7) / 2.4) ** 2))
    k = (0.0018 if treated else 0.00015) * (1 + 0.10 * abs(ph - 7))

    t = np.linspace(0, duration, 160)
    rel = np.exp(-k * t)
    A = A0 * rel

    st.markdown("#### 가정한 지수 감소 예시")
    st.caption(
        "1차 지수 감소를 가정해 계산한 곡선입니다. "
        "실측 결과가 아닙니다."
    )

    fig, ax = make_figure()

    treatment_label = "H2O2-treated" if treated else "Untreated control"

    ax.plot(
        t,
        A,
        linewidth=2.4,
        color="#2563eb",
        label=f"pH {ph:.1f} | {treatment_label}",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absorbance, A")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, A0 * 1.15)

    add_legend(ax)
    show_figure(fig)

    if "view_time" not in st.session_state:
        st.session_state["view_time"] = 0

    st.session_state["view_time"] = min(
        st.session_state["view_time"], duration
    )

    now = st.slider(
        "색 변화 관찰 시점 (초)",
        min_value=0,
        max_value=duration,
        step=1,
        key="view_time",
    )

    current_rel = math.exp(-k * now)
    fade = (1 - current_rel) * (1 if treated else 0.18)
    liquid = ph_color(ph, fade)

    st.markdown(
        f"""
<style>
.beaker-card {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 48px;
    padding: 32px 24px;
    margin: 16px 0 20px;
    border: 1px solid #CBD5E1;
    border-radius: 20px;
    background: #F8FAFC;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
}}

.beaker-scene {{
    width: 210px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.beaker-glass {{
    position: relative;
    width: 160px;
    height: 205px;
    box-sizing: border-box;
    border: 3px solid #64748B;
    border-top: none;
    border-radius: 0 0 22px 22px;
    background: rgba(100, 116, 139, 0.06);
}}

.beaker-rim {{
    position: absolute;
    top: -5px;
    left: -9px;
    right: -9px;
    height: 8px;
    border: 2px solid #64748B;
    border-radius: 50%;
    background: #E2E8F0;
    z-index: 4;
}}

.beaker-fluid {{
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 74%;
    border-radius: 0 0 18px 18px;
    background: {liquid};
    box-shadow: inset -12px 0 24px rgba(0, 0, 0, 0.12);
}}

.beaker-surface {{
    position: absolute;
    top: -5px;
    left: 0;
    right: 0;
    height: 11px;
    border-radius: 50%;
    background: {liquid};
    border-top: 1px solid rgba(255, 255, 255, 0.7);
    filter: brightness(1.12);
}}

.beaker-shine {{
    position: absolute;
    top: 16px;
    bottom: 18px;
    left: 12px;
    width: 8px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.7);
    z-index: 3;
}}

.beaker-mark {{
    position: absolute;
    right: 0;
    width: 23px;
    height: 1px;
    background: rgba(71, 85, 105, 0.7);
    z-index: 3;
}}

.beaker-platform {{
    width: 195px;
    height: 10px;
    margin-top: 10px;
    border-radius: 50%;
    background: rgba(15, 23, 42, 0.15);
}}

.beaker-details {{
    width: 100%;
    max-width: 340px;
    min-width: 0;
}}

.beaker-badge {{
    display: inline-block;
    padding: 5px 11px;
    border: 1px solid #93C5FD;
    border-radius: 20px;
    background: #DBEAFE;
    font-size: 0.82rem;
    margin-bottom: 14px;
}}

.beaker-heading {{
    margin-bottom: 16px;
    font-size: 1.25rem;
    font-weight: 700;
}}

.beaker-value-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid #E2E8F0;
}}

.beaker-value-label {{
    font-size: 0.93rem;
}}

.beaker-value {{
    font-size: 1.3rem;
    font-weight: 700;
    white-space: nowrap;
}}

.beaker-note {{
    margin-top: 16px;
    font-size: 0.83rem;
    line-height: 1.7;
}}

.beaker-label {{
    margin-top: 12px;
    font-size: 0.85rem;
    text-align: center;
}}

.beaker-card .beaker-heading,
.beaker-card .beaker-value {{
    color: #0F172A !important;
}}

.beaker-card .beaker-label,
.beaker-card .beaker-value-label,
.beaker-card .beaker-note {{
    color: #475569 !important;
}}

.beaker-card .beaker-badge {{
    color: #1E3A8A !important;
}}

@media (max-width: 768px) {{
    .beaker-card {{
        flex-direction: column;
        gap: 24px;
        padding: 26px 18px;
    }}

    .beaker-details {{
        max-width: 100%;
    }}

    .beaker-heading {{
        text-align: center;
    }}
}}
</style>

<div class="beaker-card">
    <div class="beaker-scene">
        <div class="beaker-glass">
            <div class="beaker-rim"></div>
            <div class="beaker-fluid">
                <div class="beaker-surface"></div>
            </div>
            <div class="beaker-shine"></div>
            <div class="beaker-mark" style="top:25%;"></div>
            <div class="beaker-mark" style="top:45%;"></div>
            <div class="beaker-mark" style="top:65%;"></div>
            <div class="beaker-mark" style="top:85%;"></div>
        </div>
        <div class="beaker-platform"></div>
        <div class="beaker-label">안토시아닌 용액 · 개념도</div>
    </div>
    <div class="beaker-details">
        <div class="beaker-badge">
            {'H₂O₂ 처리 조건' if treated else 'H₂O₂ 미처리 조건'}
        </div>
        <div class="beaker-heading">시간에 따른 색 변화</div>
        <div class="beaker-value-row">
            <span class="beaker-value-label">관찰 시점</span>
            <span class="beaker-value">{now} s</span>
        </div>
        <div class="beaker-value-row">
            <span class="beaker-value-label">설정 pH</span>
            <span class="beaker-value">{ph:.1f}</span>
        </div>
        <div class="beaker-value-row">
            <span class="beaker-value-label">상대 흡광도 · 가정값</span>
            <span class="beaker-value">{current_rel:.3f}</span>
        </div>
        <div class="beaker-note">
            위 슬라이더를 움직여 시점별 색상을 확인하세요.<br>
            색상은 정성적 표현이며 실제 분광 데이터로 보정되지 않았습니다.
            비커의 액면과 눈금은 부피 측정용이 아닙니다.
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "초기 흡광도와 속도 계수는 교육용 가정값입니다. "
        "실제 안토시아닌 조성, H₂O₂ 농도와 반응 경로를 반영한 "
        "검증 모델이 아닙니다."
    )
# =========================================================
# 2. 실험 데이터 분석
# =========================================================
data = None
quality = None
source_name = ""

with tabs[1]:
    st.subheader("실험 데이터 분석")

    source = st.radio(
        "데이터 선택",
        ["내 실험 데이터 (pH 4·7·10)", "CSV 직접 업로드"],
        horizontal=not mobile_view,
        key="data_source",
    )
    source_name = source

    if source == "내 실험 데이터 (pH 4·7·10)":
        try:
            missing_files = [
                name for name in BUILTIN_FILES
                if not (DATA_DIR / name).is_file()
            ]

            if missing_files:
                raise FileNotFoundError(
                    "data 폴더에서 찾을 수 없는 파일: "
                    + ", ".join(missing_files)
                )

            signatures = tuple(
                (name, (DATA_DIR / name).stat().st_mtime_ns)
                for name in BUILTIN_FILES
            )
            data, quality = load_builtin(signatures)
            st.success(
                f"저장된 실험 데이터 {len(data)}개 행을 불러왔습니다."
            )
        except Exception as error:
            st.error(f"내장 데이터 읽기 오류: {error}")

        st.caption(
            "H₂O₂ 처리 조건의 pH별 데이터입니다. "
            "pH 4는 672초까지, pH 7·10은 300초까지 기록되어 있습니다. "
            "filled·processed 파일에 보간이나 보정을 적용했다면 "
            "발표·보고서에 원자료와 처리 방법을 구분해 적어주세요."
        )

    else:
        st.write("필수 열: `time`, `pH`, `group`, `absorbance`")
        st.caption(
            "time은 초 단위입니다. 반복 실험은 group 이름으로 구분하세요. "
            "예: H2O2_rep1, H2O2_rep2"
        )

        uploaded_files = st.file_uploader(
            "CSV 업로드 — 여러 파일 선택 가능",
            type=["csv"],
            accept_multiple_files=True,
            key="measurement_csv",
        )

        if uploaded_files:
            try:
                frames = []
                for uploaded in uploaded_files:
                    frame = pd.read_csv(
                        uploaded, encoding="utf-8-sig"
                    )
                    frame.columns = [
                        str(c).replace("\ufeff", "").strip()
                        for c in frame.columns
                    ]
                    missing = REQUIRED_COLUMNS - set(frame.columns)
                    if missing:
                        raise ValueError(
                            f"{uploaded.name}: 필수 열 누락 — "
                            + ", ".join(sorted(missing))
                        )
                    frames.append(frame)

                raw = pd.concat(frames, ignore_index=True)
                data = prepare(raw)
                quality = quality_summary(raw, data)
                st.success(
                    f"{len(uploaded_files)}개 파일, "
                    f"{len(data)}개 유효 행을 불러왔습니다."
                )
            except Exception as error:
                data = None
                st.error(f"CSV 읽기 오류: {error}")

    if data is None:
        st.info("유효한 실험 데이터를 불러오면 분석 결과가 표시됩니다.")

    else:
        show_metrics(
            [
                ("유효 행", len(data)),
                ("원본 결측 포함 행", quality["missing_rows"]),
                ("제외 행", quality["removed_rows"]),
                (
                    "시간 범위",
                    f"{data.time.min():g}–{data.time.max():g} s",
                ),
            ]
        )

        with st.expander("데이터 품질 상세"):
            st.write(f"원본 행 수: {quality['original_rows']}")
            duplicate_count = int(
                data.duplicated(
                    subset=["time", "pH", "group", "absorbance"]
                ).sum()
            )
            st.write(f"동일 측정값 중복 행: {duplicate_count}")

            duplicate_times = int(
                data.duplicated(
                    subset=["pH", "group", "time"],
                    keep=False,
                ).sum()
            )
            st.write(
                f"같은 조건·시간에 여러 값이 있는 행: {duplicate_times}"
            )

            if duplicate_times:
                st.warning(
                    "서로 다른 반복 실험이 같은 group에 섞이지 않았는지 "
                    "확인하세요. 현재 각 pH·group을 하나의 시계열로 "
                    "처리하며 중복 시점을 자동 평균하지 않습니다."
                )

            if (data["absorbance"] < 0).any():
                st.warning(
                    "음수 흡광도가 있습니다. 블랭크 보정과 원자료를 확인하세요."
                )

            if data["A_norm"].isna().any():
                st.warning(
                    "초기 흡광도가 0인 조건은 정규화를 계산하지 않습니다."
                )

            st.caption(
                "원본 결측 포함 행은 CSV에 빈 값으로 남은 행만 셉니다. "
                "이미 보간된 값은 이 검사로 구분할 수 없습니다."
            )

        with st.expander("실험 데이터 표"):
            st.dataframe(
                data,
                use_container_width=True,
                height=280,
                hide_index=True,
            )

        st.markdown("#### 그래프 보기 선택")
        view = ph_selector(data, "analysis_ph")

        st.caption(
            "버튼을 누르면 아래 두 그래프와 조건별 요약·모형 결과가 "
            "함께 바뀝니다. 그래프의 색은 pH 구분용입니다."
        )

        st.markdown("#### 원래 흡광도")
        plot_data(
            view,
            key="analysis_absorbance",
            color_source=data,
        )

        st.markdown("#### 정규화 흡광도")
        plot_data(
            view,
            key="analysis_normalized",
            normalized=True,
            reference=True,
            color_source=data,
        )

        normalization_explanation()

        st.caption(
            "그래프 드래그: 확대 · 더블클릭: 범위 복원 · "
            "범례 클릭: 곡선 숨기기/표시 · 카메라 버튼: PNG 저장"
        )

        summaries = []
        fit_tables = []

        for (p, group), series in view.groupby(["pH", "group"]):
            series = series.sort_values("time")
            first = float(series["absorbance"].iloc[0])
            last = float(series["absorbance"].iloc[-1])
            start = float(series["time"].iloc[0])
            end = float(series["time"].iloc[-1])
            elapsed = end - start

            minimum_index = series["absorbance"].idxmin()

            summaries.append(
                {
                    "pH": p,
                    "group": group,
                    "시작 시간(s)": start,
                    "종료 시간(s)": end,
                    "초기 A": first,
                    "최종 A": last,
                    "최저 A": float(
                        series.loc[minimum_index, "absorbance"]
                    ),
                    "최저값 시간(s)": float(
                        series.loc[minimum_index, "time"]
                    ),
                    "최종 상대 감소율(%)": (
                        (first - last) / first * 100
                        if first != 0 else np.nan
                    ),
                    "구간 평균 감소율(A/s)": (
                        (first - last) / elapsed
                        if elapsed > 0 else np.nan
                    ),
                }
            )

            fitted = model_fit(
                series["time"].to_numpy() - start,
                series["absorbance"].to_numpy(),
            )
            if not fitted.empty:
                fitted.insert(0, "group", str(group))
                fitted.insert(0, "pH", p)
                fit_tables.append(fitted)

        st.markdown("#### 선택 조건 요약")
        summary_df = pd.DataFrame(summaries)
        st.dataframe(
            summary_df.round(6),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "구간 평균 감소율은 처음과 마지막 값으로 계산한 순변화율입니다. "
            "감소 후 증가 같은 중간 변화까지 설명하지는 못합니다. "
            "조건별 종료 시간이 다르면 이 값을 직접 비교할 때 주의하세요."
        )

        if fit_tables:
            st.markdown("#### 실측 데이터 모형 적합")
            fit_df = pd.concat(fit_tables, ignore_index=True)

            best_df = (
                fit_df.sort_values("RMSE")
                .groupby(["pH", "group"], sort=True)
                .head(1)
                .sort_values(["pH", "group"])
                .rename(columns={"모형": "후보 중 최소 RMSE 모형"})
            )

            st.dataframe(
                best_df.round(6),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("세 모형의 적합 결과 모두 보기"):
                st.dataframe(
                    fit_df.round(6),
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "0차는 A, 1차는 ln(A), 2차는 1/A를 시간에 대해 "
                "선형회귀합니다. RMSE와 R²는 원래 흡광도 척도에서 "
                "계산합니다. 이 방식은 원래 척도의 비선형 최소제곱 "
                "적합과 같지 않습니다."
            )
            st.warning(
                "후보 중 오차가 작다는 사실만으로 실제 반응 차수가 "
                "확정되지는 않습니다. 특히 pH 7의 감소 후 증가 형태는 "
                "단조 감소 모형으로 충분히 설명하기 어렵습니다. "
                "겉보기 k의 단위·의미는 모형마다 다르며, "
                "음수 k를 탈색 속도상수로 해석하지 않습니다."
            )

        with st.expander("실험 해석 시 주의점"):
            st.markdown(
                "- 흡광도 감소를 실제 분해량과 동일시하지 않습니다.\n"
                "- 동일 파장·광로 길이·블랭크 등 측정 조건을 확인하세요.\n"
                "- 현재 내장 데이터는 모두 H₂O₂ 처리 조건입니다. "
                "미처리 대조군과의 차이는 이 자료만으로 판단할 수 없습니다.\n"
                "- 반복 실험이 없으면 오차 범위나 통계적 유의성을 "
                "단정하지 않습니다.\n"
                "- 보간·보정값은 독립적인 추가 측정값이 아닙니다."
            )

        st.download_button(
            "전체 정리 데이터 CSV 다운로드",
            data.to_csv(index=False).encode("utf-8-sig"),
            file_name="anthocyanin_processed.csv",
            mime="text/csv",
            key="download_processed",
            use_container_width=True,
        )


# =========================================================
# 3. 조건별 비교
# =========================================================
with tabs[2]:
    st.subheader("조건별 비교")

    if data is None:
        st.info("분석 탭에서 읽을 수 있는 CSV를 선택해 주세요.")
    else:
        compare_view = ph_selector(data, "comparison_ph")

        common_only = st.checkbox(
            "선택 조건의 공통 시간 범위만 비교",
            value=True,
            key="common_time",
        )

        if common_only:
            time_ranges = compare_view.groupby(
                ["pH", "group"]
            )["time"].agg(["min", "max"])

            common_start = float(time_ranges["min"].max())
            common_end = float(time_ranges["max"].min())

            if common_start <= common_end:
                compare_view = compare_view[
                    compare_view["time"].between(
                        common_start, common_end
                    )
                ].copy()
                st.caption(
                    f"공통 시간 범위: {common_start:g}–{common_end:g}초. "
                    "정규화 기준은 각 조건의 원래 첫 측정값을 유지합니다."
                )
            else:
                compare_view = compare_view.iloc[0:0]
                st.warning("선택 조건에 공통 시간 범위가 없습니다.")

        if not compare_view.empty:
            st.markdown("#### 원래 흡광도 비교")
            plot_data(
                compare_view,
                key="comparison_absorbance",
                color_source=data,
            )

            st.markdown("#### 정규화 흡광도 비교")
            plot_data(
                compare_view,
                key="comparison_normalized",
                normalized=True,
                reference=True,
                color_source=data,
            )

            normalization_explanation(
                "비교 그래프의 정규화 기준과 해석"
            )

        st.markdown(
            "**비교할 때 확인할 점**\n"
            "- 먼저 동일 시간 범위에서 상대적 변화를 비교합니다.\n"
            "- pH별 초기 흡광도 차이는 농도 차이만을 뜻하지 않습니다.\n"
            "- pH 7의 후반 흡광도 증가는 데이터에 나타난 현상입니다. "
            "그 원인은 이 그래프만으로 확정할 수 없습니다.\n"
            "- 처리군·대조군 비교는 동일 pH의 두 데이터가 모두 "
            "있을 때 가능합니다."
        )


# =========================================================
# 4. 화학 개념
# =========================================================
with tabs[3]:
    st.subheader("화학 개념 학습")

    concepts = [
        (
            "1. 안토시아닌의 산·염기 반응",
            "안토시아닌은 pH에 따라 양성자 이동, 수화, 구조 전환 등의 "
            "영향을 받아 여러 화학종의 분포가 달라집니다. "
            "대표적인 형태로 플라빌륨 양이온, 퀴노이드 형태, "
            "헤미케탈, 칼콘 등이 있습니다. 종류와 용액 조건에 따라 "
            "평형과 흡광 특성이 달라집니다.",
        ),
        (
            "2. 산성 조건",
            "충분히 산성인 조건에서는 플라빌륨 양이온이 상대적으로 "
            "안정화되어 붉은색 계열이 나타날 수 있습니다. "
            "특정 pH의 실제 색은 색소 종류·농도·용액 조성에 따라 "
            "달라지므로 고정된 RGB 값으로 일반화할 수 없습니다.",
        ),
        (
            "3. pH 증가에 따른 변화",
            "pH가 증가하면 화학종의 분포와 전자 구조가 변하여 "
            "흡수 스펙트럼이 달라질 수 있습니다. "
            "보라색·청색 계열의 변화뿐 아니라 수화에 따른 탈색도 "
            "가능합니다. pH 7이면 반드시 파란색인 것은 아닙니다.",
        ),
        (
            "4. 염기성 조건과 안정성",
            "염기성에서는 여러 퀴노이드 형태와 칼콘 계열 등의 "
            "분포가 달라질 수 있습니다. 강한 알칼리성에서는 "
            "비가역적 분해 가능성도 커질 수 있습니다. "
            "pH 10의 실제 색을 초록색으로 단정할 수는 없습니다.",
        ),
        (
            "5. H₂O₂의 작용",
            "H₂O₂는 산화제로 작용하여 안토시아닌의 구조 변화나 "
            "분해에 관여할 수 있습니다. 발색에 관여하는 공액 전자계가 "
            "변하면 흡광 특성도 바뀝니다. 구체적인 경로와 생성물은 "
            "이 앱의 흡광도 자료만으로 확인하지 않습니다.",
        ),
        (
            "6. Beer–Lambert 법칙",
            "일정한 화학종·측정 조건에서 흡광도는 농도에 비례할 수 "
            "있습니다. pH가 바뀌어 화학종과 몰흡광계수가 달라지면 "
            "흡광도 차이를 곧바로 농도 차이로 해석하면 안 됩니다.",
        ),
        (
            "7. 화학 평형",
            "가역 반응에서 정반응과 역반응의 속도가 같아져 "
            "거시적 조성이 일정하게 유지되는 동적 상태입니다. "
            "H₂O₂에 의한 모든 변화가 가역 평형인 것은 아닙니다.",
        ),
        (
            "8. 르샤틀리에 원리",
            "평형계에 농도·온도 등의 변화가 가해지면 그 영향을 "
            "줄이는 방향으로 새로운 평형에 도달하는 경향입니다. "
            "pH에 따른 화학종 분포를 이해하는 데 활용할 수 있습니다.",
        ),
        (
            "9. 산화·환원",
            "산화는 전자를 잃거나 산화수가 증가하는 과정이고, "
            "환원은 전자를 얻거나 산화수가 감소하는 과정입니다. "
            "산화제는 다른 물질을 산화시키며 자신은 환원됩니다.",
        ),
        (
            "10. 반응 속도와 흡광도 변화율",
            "엄밀한 반응 속도는 농도의 시간 변화로 정의합니다. "
            "이 앱에서는 흡광도 시간 변화율을 겉보기 지표로 사용하며 "
            "농도 기반 반응 속도와 구분합니다.",
        ),
        (
            "11. 완충 용액과 공통 이온 효과",
            "완충 용액은 보통 약산과 짝염기 또는 약염기와 짝산으로 "
            "구성되어 소량의 산·염기 첨가에 따른 pH 변화를 줄입니다. "
            "공통 이온 효과는 평형계에 이미 있는 이온을 첨가했을 때 "
            "평형이 이동하는 현상입니다. 두 개념은 관련되지만 "
            "동일하지 않습니다.",
        ),
    ]

    for index, (title, body) in enumerate(concepts):
        with st.expander(title):
            st.write(body)
            if index == 5:
                st.latex(r"A=-\log_{10}(T),\qquad T=I/I_0")
                st.latex(r"A=\varepsilon l c")
                st.caption("흡광도 A는 무차원량입니다.")
            elif index == 9:
                st.latex(
                    r"\text{평균 흡광도 감소율}"
                    r"=-\frac{A(t_2)-A(t_1)}{t_2-t_1}"
                )
                st.latex(
                    r"\text{순간 흡광도 감소율}=-\frac{dA}{dt}"
                )


# =========================================================
# 5. 생명과학
# =========================================================
with tabs[4]:
    render_bio_connection_tab()


# =========================================================
# 6. 저장
# =========================================================
with tabs[5]:
    st.subheader("보고서 및 조건 저장")

    settings = {
        "pH": ph,
        "buffer_volume_mL": buffer,
        "anthocyanin_extract_volume_mL": extract,
        "H2O2_treatment": treated,
        "observation_time_s": duration,
        "temperature_C": 25,
        "note": "교육용 가정 시뮬레이션 조건이며 CSV 실험 조건과 별개",
    }

    with st.expander("현재 시뮬레이션 조건", expanded=True):
        st.json(settings)

    st.download_button(
        "시뮬레이션 조건 JSON 저장",
        json.dumps(
            settings, ensure_ascii=False, indent=2
        ).encode("utf-8"),
        file_name="anthocyanin_conditions.json",
        mime="application/json",
        use_container_width=True,
        key="download_settings",
    )

    saved_json = st.file_uploader(
        "저장한 조건 JSON 확인",
        type=["json"],
        key="conditions_json",
    )
    if saved_json is not None:
        try:
            loaded = json.loads(
                saved_json.getvalue().decode("utf-8-sig")
            )
            st.json(loaded)
            st.caption(
                "확인용으로 표시하며 사이드바에 자동 적용하지 않습니다."
            )
        except Exception as error:
            st.error(f"JSON 읽기 오류: {error}")

    lines = [
        APP_TITLE,
        f"생성 시각(서버 기준): {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "[가정 시뮬레이션 조건]",
        f"pH: {ph:g}",
        f"완충용액: {buffer:g} mL",
        f"추출액: {extract:g} mL",
        f"H₂O₂ 처리 설정: {treated}",
        f"관찰 시간: {duration} s",
        "온도 설정: 25 °C",
        "",
        "시뮬레이션의 초기 흡광도·계수·색은 교육용 설정입니다.",
        "부피에 따른 희석 효과는 계산하지 않습니다.",
        "비커 색은 실측 색이나 실제 색소 잔존율을 나타내지 않습니다.",
    ]

    if data is not None:
        lines.extend(
            [
                "",
                "[분석 데이터]",
                f"데이터 선택: {source_name}",
                f"유효 행: {len(data)}",
                (
                    f"시간 범위: {data.time.min():g}"
                    f"–{data.time.max():g} s"
                ),
                "정규화: 각 pH·group의 첫 측정값으로 A/A₀ 계산",
                "",
                "[조건별 초기·최종 값]",
            ]
        )

        for (p, group), series in data.groupby(["pH", "group"]):
            series = series.sort_values("time")
            first = float(series["absorbance"].iloc[0])
            last = float(series["absorbance"].iloc[-1])
            lines.append(
                f"pH {p:g}, {group}: "
                f"{series.time.iloc[0]:g}–{series.time.iloc[-1]:g} s, "
                f"A 초기={first:.6f}, A 최종={last:.6f}"
            )

        lines.extend(
            [
                "",
                "[해석상 한계]",
                "조건별 종료 시간이 다를 수 있습니다.",
                "정규화는 pH별 화학종·몰흡광계수 차이를 제거하지 않습니다.",
                "특정 파장 흡광도 감소를 실제 분해량으로 단정하지 않습니다.",
                "모형 적합만으로 실제 반응 차수를 확정하지 않습니다.",
                "반복 실험 없이 통계적 유의성을 주장하지 않습니다.",
                "보간·보정값이 있으면 처리 방법을 별도로 기록해야 합니다.",
            ]
        )

    st.download_button(
        "TXT 요약 보고서 다운로드",
        "\n".join(lines).encode("utf-8-sig"),
        file_name="anthocyanin_report.txt",
        mime="text/plain",
        use_container_width=True,
        key="download_report",
    )

st.markdown("---")
st.caption(
    "교육용 분석 도구 · 가정 시뮬레이션과 실험 측정 결과를 구분하세요."
)

# =========================================================
# 2. 실험 데이터 분석
# =========================================================
data = None
quality = None
source_name = ""

with tabs[1]:
    st.subheader("실험 데이터 분석")

    source = st.radio(
        "데이터 선택",
        ["내 실험 데이터 (pH 4·7·10)", "CSV 직접 업로드"],
        horizontal=not mobile_view,
        key="data_source",
    )
    source_name = source

    if source == "내 실험 데이터 (pH 4·7·10)":
        try:
            missing_files = [
                name for name in BUILTIN_FILES
                if not (DATA_DIR / name).is_file()
            ]

            if missing_files:
                raise FileNotFoundError(
                    "data 폴더에서 찾을 수 없는 파일: "
                    + ", ".join(missing_files)
                )

            signatures = tuple(
                (name, (DATA_DIR / name).stat().st_mtime_ns)
                for name in BUILTIN_FILES
            )

            data, quality = load_builtin(signatures)

            st.success(
                f"저장된 실험 데이터 {len(data)}개 행을 불러왔습니다."
            )

        except Exception as error:
            st.error(f"내장 데이터 읽기 오류: {error}")

        st.caption(
            "H₂O₂ 처리 조건의 pH별 데이터입니다. "
            "pH 4는 672초까지, pH 7·10은 300초까지 기록되어 있습니다. "
            "filled·processed 파일에 보간이나 보정을 적용했다면 "
            "발표·보고서에 원자료와 처리 방법을 구분해 적어주세요."
        )

    else:
        st.write("필수 열: `time`, `pH`, `group`, `absorbance`")

        st.caption(
            "time은 초 단위입니다. 반복 실험은 group 이름으로 구분하세요. "
            "예: H2O2_rep1, H2O2_rep2"
        )

        uploaded_files = st.file_uploader(
            "CSV 업로드 — 여러 파일 선택 가능",
            type=["csv"],
            accept_multiple_files=True,
            key="measurement_csv",
        )

        if uploaded_files:
            try:
                frames = []

                for uploaded in uploaded_files:
                    frame = pd.read_csv(
                        uploaded,
                        encoding="utf-8-sig",
                    )

                    frame.columns = [
                        str(c).replace("\ufeff", "").strip()
                        for c in frame.columns
                    ]

                    missing = REQUIRED_COLUMNS - set(frame.columns)

                    if missing:
                        raise ValueError(
                            f"{uploaded.name}: 필수 열 누락 — "
                            + ", ".join(sorted(missing))
                        )

                    frames.append(frame)

                raw = pd.concat(frames, ignore_index=True)
                data = prepare(raw)
                quality = quality_summary(raw, data)

                st.success(
                    f"{len(uploaded_files)}개 파일, "
                    f"{len(data)}개 유효 행을 불러왔습니다."
                )

            except Exception as error:
                data = None
                st.error(f"CSV 읽기 오류: {error}")

    if data is None:
        st.info(
            "유효한 실험 데이터를 불러오면 분석 결과가 표시됩니다."
        )

    else:
        show_metrics(
            [
                ("유효 행", len(data)),
                ("원본 결측 포함 행", quality["missing_rows"]),
                ("제외 행", quality["removed_rows"]),
                (
                    "시간 범위",
                    f"{data.time.min():g}–{data.time.max():g} s",
                ),
            ]
        )

        with st.expander("데이터 품질 상세"):
            st.write(f"원본 행 수: {quality['original_rows']}")

            duplicate_count = int(
                data.duplicated(
                    subset=["time", "pH", "group", "absorbance"]
                ).sum()
            )

            st.write(f"동일 측정값 중복 행: {duplicate_count}")

            duplicate_times = int(
                data.duplicated(
                    subset=["pH", "group", "time"],
                    keep=False,
                ).sum()
            )

            st.write(
                f"같은 조건·시간에 여러 값이 있는 행: {duplicate_times}"
            )

            if duplicate_times:
                st.warning(
                    "서로 다른 반복 실험이 같은 group에 섞이지 않았는지 "
                    "확인하세요. 현재 각 pH·group을 하나의 시계열로 "
                    "처리하며 중복 시점을 자동 평균하지 않습니다."
                )

            if (data["absorbance"] < 0).any():
                st.warning(
                    "음수 흡광도가 있습니다. "
                    "블랭크 보정과 원자료를 확인하세요."
                )

            if data["A_norm"].isna().any():
                st.warning(
                    "초기 흡광도가 0인 조건은 정규화를 계산하지 않습니다."
                )

            st.caption(
                "원본 결측 포함 행은 CSV에 빈 값으로 남은 행만 셉니다. "
                "이미 보간된 값은 이 검사로 구분할 수 없습니다."
            )

        with st.expander("실험 데이터 표"):
            st.dataframe(
                data,
                use_container_width=True,
                height=280,
                hide_index=True,
            )

        st.markdown("#### 그래프 보기 선택")

        view = ph_selector(data, "analysis_ph")

        st.caption(
            "버튼을 누르면 아래 두 그래프와 조건별 요약·모형 결과가 "
            "함께 바뀝니다. 그래프의 색은 pH 구분용입니다."
        )

        st.markdown("#### 원래 흡광도")

        plot_data(
            view,
            key="analysis_absorbance",
            color_source=data,
        )

        st.markdown("#### 정규화 흡광도")

        plot_data(
            view,
            key="analysis_normalized",
            normalized=True,
            reference=True,
            color_source=data,
        )

        normalization_explanation()

        st.caption(
            "그래프 드래그: 확대 · 더블클릭: 범위 복원 · "
            "범례 클릭: 곡선 숨기기/표시 · 카메라 버튼: PNG 저장"
        )

        summaries = []
        fit_tables = []

        for (p, group), series in view.groupby(["pH", "group"]):
            series = series.sort_values("time")

            first = float(series["absorbance"].iloc[0])
            last = float(series["absorbance"].iloc[-1])
            start = float(series["time"].iloc[0])
            end = float(series["time"].iloc[-1])
            elapsed = end - start

            minimum_index = series["absorbance"].idxmin()

            summaries.append(
                {
                    "pH": p,
                    "group": group,
                    "시작 시간(s)": start,
                    "종료 시간(s)": end,
                    "초기 A": first,
                    "최종 A": last,
                    "최저 A": float(
                        series.loc[minimum_index, "absorbance"]
                    ),
                    "최저값 시간(s)": float(
                        series.loc[minimum_index, "time"]
                    ),
                    "최종 상대 감소율(%)": (
                        (first - last) / first * 100
                        if first != 0 else np.nan
                    ),
                    "구간 평균 감소율(A/s)": (
                        (first - last) / elapsed
                        if elapsed > 0 else np.nan
                    ),
                }
            )

            fitted = model_fit(
                series["time"].to_numpy() - start,
                series["absorbance"].to_numpy(),
            )

            if not fitted.empty:
                fitted.insert(0, "group", str(group))
                fitted.insert(0, "pH", p)
                fit_tables.append(fitted)

        st.markdown("#### 선택 조건 요약")

        summary_df = pd.DataFrame(summaries)

        st.dataframe(
            summary_df.round(6),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "구간 평균 감소율은 처음과 마지막 값으로 계산한 순변화율입니다. "
            "감소 후 증가 같은 중간 변화까지 설명하지는 못합니다. "
            "조건별 종료 시간이 다르면 이 값을 직접 비교할 때 주의하세요."
        )

        if fit_tables:
            st.markdown("#### 실측 데이터 모형 적합")

            fit_df = pd.concat(fit_tables, ignore_index=True)

            best_df = (
                fit_df.sort_values("RMSE")
                .groupby(["pH", "group"], sort=True)
                .head(1)
                .sort_values(["pH", "group"])
                .rename(
                    columns={"모형": "후보 중 최소 RMSE 모형"}
                )
            )

            st.dataframe(
                best_df.round(6),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("세 모형의 적합 결과 모두 보기"):
                st.dataframe(
                    fit_df.round(6),
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "0차는 A, 1차는 ln(A), 2차는 1/A를 시간에 대해 "
                "선형회귀합니다. RMSE와 R²는 원래 흡광도 척도에서 "
                "계산합니다. 이 방식은 원래 척도의 비선형 최소제곱 "
                "적합과 같지 않습니다."
            )

            st.warning(
                "후보 중 오차가 작다는 사실만으로 실제 반응 차수가 "
                "확정되지는 않습니다. 특히 pH 7의 감소 후 증가 형태는 "
                "단조 감소 모형으로 충분히 설명하기 어렵습니다. "
                "겉보기 k의 단위·의미는 모형마다 다르며, "
                "음수 k를 탈색 속도상수로 해석하지 않습니다."
            )

        with st.expander("실험 해석 시 주의점"):
            st.markdown(
                "- 흡광도 감소를 실제 분해량과 동일시하지 않습니다.\n"
                "- 동일 파장·광로 길이·블랭크 등 측정 조건을 확인하세요.\n"
                "- 현재 내장 데이터는 모두 H₂O₂ 처리 조건입니다. "
                "미처리 대조군과의 차이는 이 자료만으로 판단할 수 없습니다.\n"
                "- 반복 실험이 없으면 오차 범위나 통계적 유의성을 "
                "단정하지 않습니다.\n"
                "- 보간·보정값은 독립적인 추가 측정값이 아닙니다."
            )

        st.download_button(
            "전체 정리 데이터 CSV 다운로드",
            data.to_csv(index=False).encode("utf-8-sig"),
            file_name="anthocyanin_processed.csv",
            mime="text/csv",
            key="download_processed",
            use_container_width=True,
        )


# =========================================================
# 3. 조건별 비교
# =========================================================
with tabs[2]:
    st.subheader("조건별 비교")

    if data is None:
        st.info("분석 탭에서 읽을 수 있는 CSV를 선택해 주세요.")

    else:
        compare_view = ph_selector(data, "comparison_ph")

        common_only = st.checkbox(
            "선택 조건의 공통 시간 범위만 비교",
            value=True,
            key="common_time",
        )

        if common_only:
            time_ranges = compare_view.groupby(
                ["pH", "group"]
            )["time"].agg(["min", "max"])

            common_start = float(time_ranges["min"].max())
            common_end = float(time_ranges["max"].min())

            if common_start <= common_end:
                compare_view = compare_view[
                    compare_view["time"].between(
                        common_start,
                        common_end,
                    )
                ].copy()

                st.caption(
                    f"공통 시간 범위: "
                    f"{common_start:g}–{common_end:g}초. "
                    "정규화 기준은 각 조건의 원래 첫 측정값을 유지합니다."
                )

            else:
                compare_view = compare_view.iloc[0:0]
                st.warning("선택 조건에 공통 시간 범위가 없습니다.")

        if not compare_view.empty:
            st.markdown("#### 원래 흡광도 비교")

            plot_data(
                compare_view,
                key="comparison_absorbance",
                color_source=data,
            )

            st.markdown("#### 정규화 흡광도 비교")

            plot_data(
                compare_view,
                key="comparison_normalized",
                normalized=True,
                reference=True,
                color_source=data,
            )

            normalization_explanation(
                "비교 그래프의 정규화 기준과 해석"
            )

        st.markdown(
            "**비교할 때 확인할 점**\n"
            "- 먼저 동일 시간 범위에서 상대적 변화를 비교합니다.\n"
            "- pH별 초기 흡광도 차이는 농도 차이만을 뜻하지 않습니다.\n"
            "- pH 7의 후반 흡광도 증가는 데이터에 나타난 현상입니다. "
            "그 원인은 이 그래프만으로 확정할 수 없습니다.\n"
            "- 처리군·대조군 비교는 동일 pH의 두 데이터가 모두 "
            "있을 때 가능합니다."
        )


# =========================================================
# 4. 화학 개념
# =========================================================
with tabs[3]:
    st.subheader("화학 개념 학습")

    concepts = [
        (
            "1. 안토시아닌의 산·염기 반응",
            "안토시아닌은 pH에 따라 양성자 이동, 수화, 구조 전환 등의 "
            "영향을 받아 여러 화학종의 분포가 달라집니다. "
            "대표적인 형태로 플라빌륨 양이온, 퀴노이드 형태, "
            "헤미케탈, 칼콘 등이 있습니다. 종류와 용액 조건에 따라 "
            "평형과 흡광 특성이 달라집니다.",
        ),
        (
            "2. 산성 조건",
            "충분히 산성인 조건에서는 플라빌륨 양이온이 상대적으로 "
            "안정화되어 붉은색 계열이 나타날 수 있습니다. "
            "특정 pH의 실제 색은 색소 종류·농도·용액 조성에 따라 "
            "달라지므로 고정된 RGB 값으로 일반화할 수 없습니다.",
        ),
        (
            "3. pH 증가에 따른 변화",
            "pH가 증가하면 화학종의 분포와 전자 구조가 변하여 "
            "흡수 스펙트럼이 달라질 수 있습니다. "
            "보라색·청색 계열의 변화뿐 아니라 수화에 따른 탈색도 "
            "가능합니다. pH 7이면 반드시 파란색인 것은 아닙니다.",
        ),
        (
            "4. 염기성 조건과 안정성",
            "염기성에서는 여러 퀴노이드 형태와 칼콘 계열 등의 "
            "분포가 달라질 수 있습니다. 강한 알칼리성에서는 "
            "비가역적 분해 가능성도 커질 수 있습니다. "
            "pH 10의 실제 색을 초록색으로 단정할 수는 없습니다.",
        ),
        (
            "5. H₂O₂의 작용",
            "H₂O₂는 산화제로 작용하여 안토시아닌의 구조 변화나 "
            "분해에 관여할 수 있습니다. 발색에 관여하는 공액 전자계가 "
            "변하면 흡광 특성도 바뀝니다. 구체적인 경로와 생성물은 "
            "이 앱의 흡광도 자료만으로 확인하지 않습니다.",
        ),
        (
            "6. Beer–Lambert 법칙",
            "일정한 화학종·측정 조건에서 흡광도는 농도에 비례할 수 "
            "있습니다. pH가 바뀌어 화학종과 몰흡광계수가 달라지면 "
            "흡광도 차이를 곧바로 농도 차이로 해석하면 안 됩니다.",
        ),
        (
            "7. 화학 평형",
            "가역 반응에서 정반응과 역반응의 속도가 같아져 "
            "거시적 조성이 일정하게 유지되는 동적 상태입니다. "
            "H₂O₂에 의한 모든 변화가 가역 평형인 것은 아닙니다.",
        ),
        (
            "8. 르샤틀리에 원리",
            "평형계에 농도·온도 등의 변화가 가해지면 그 영향을 "
            "줄이는 방향으로 새로운 평형에 도달하는 경향입니다. "
            "pH에 따른 화학종 분포를 이해하는 데 활용할 수 있습니다.",
        ),
        (
            "9. 산화·환원",
            "산화는 전자를 잃거나 산화수가 증가하는 과정이고, "
            "환원은 전자를 얻거나 산화수가 감소하는 과정입니다. "
            "산화제는 다른 물질을 산화시키며 자신은 환원됩니다.",
        ),
        (
            "10. 반응 속도와 흡광도 변화율",
            "엄밀한 반응 속도는 농도의 시간 변화로 정의합니다. "
            "이 앱에서는 흡광도 시간 변화율을 겉보기 지표로 사용하며 "
            "농도 기반 반응 속도와 구분합니다.",
        ),
        (
            "11. 완충 용액과 공통 이온 효과",
            "완충 용액은 보통 약산과 짝염기 또는 약염기와 짝산으로 "
            "구성되어 소량의 산·염기 첨가에 따른 pH 변화를 줄입니다. "
            "공통 이온 효과는 평형계에 이미 있는 이온을 첨가했을 때 "
            "평형이 이동하는 현상입니다. 두 개념은 관련되지만 "
            "동일하지 않습니다.",
        ),
    ]

    for index, (title, body) in enumerate(concepts):
        with st.expander(title):
            st.write(body)

            if index == 5:
                st.latex(
                    r"A=-\log_{10}(T),\qquad T=I/I_0"
                )
                st.latex(r"A=\varepsilon l c")
                st.caption("흡광도 A는 무차원량입니다.")

            elif index == 9:
                st.markdown("평균 흡광도 감소율")
                st.latex(
                    r"-\frac{A(t_2)-A(t_1)}{t_2-t_1}"
                )
                st.markdown("순간 흡광도 감소율")
                st.latex(r"-\frac{dA}{dt}")


# =========================================================
# 5. 생명과학
# =========================================================
with tabs[4]:
    render_bio_connection_tab()


# =========================================================
# 6. 저장
# =========================================================
with tabs[5]:
    st.subheader("보고서 및 조건 저장")

    settings = {
        "pH": ph,
        "buffer_volume_mL": buffer,
        "anthocyanin_extract_volume_mL": extract,
        "H2O2_treatment": treated,
        "observation_time_s": duration,
        "temperature_C": 25,
        "note": (
            "교육용 가정 시뮬레이션 조건이며 "
            "CSV 실험 조건과 별개"
        ),
    }

    with st.expander("현재 시뮬레이션 조건", expanded=True):
        st.json(settings)

    st.download_button(
        "시뮬레이션 조건 JSON 저장",
        json.dumps(
            settings,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        file_name="anthocyanin_conditions.json",
        mime="application/json",
        use_container_width=True,
        key="download_settings",
    )

    saved_json = st.file_uploader(
        "저장한 조건 JSON 확인",
        type=["json"],
        key="conditions_json",
    )

    if saved_json is not None:
        try:
            loaded = json.loads(
                saved_json.getvalue().decode("utf-8-sig")
            )

            st.json(loaded)

            st.caption(
                "확인용으로 표시하며 사이드바에 자동 적용하지 않습니다."
            )

        except Exception as error:
            st.error(f"JSON 읽기 오류: {error}")

    lines = [
        APP_TITLE,
        (
            "생성 시각(서버 기준): "
            f"{datetime.now():%Y-%m-%d %H:%M}"
        ),
        "",
        "[가정 시뮬레이션 조건]",
        f"pH: {ph:g}",
        f"완충용액: {buffer:g} mL",
        f"추출액: {extract:g} mL",
        f"H₂O₂ 처리 설정: {treated}",
        f"관찰 시간: {duration} s",
        "온도 설정: 25 °C",
        "",
        "시뮬레이션의 초기 흡광도·계수·색은 교육용 설정입니다.",
        "부피에 따른 희석 효과는 계산하지 않습니다.",
        "비커 색은 실측 색이나 실제 색소 잔존율을 나타내지 않습니다.",
    ]

    if data is not None:
        lines.extend(
            [
                "",
                "[분석 데이터]",
                f"데이터 선택: {source_name}",
                f"유효 행: {len(data)}",
                (
                    f"시간 범위: {data.time.min():g}"
                    f"–{data.time.max():g} s"
                ),
                "정규화: 각 pH·group의 첫 측정값으로 A/A₀ 계산",
                "",
                "[조건별 초기·최종 값]",
            ]
        )

        for (p, group), series in data.groupby(["pH", "group"]):
            series = series.sort_values("time")
            first = float(series["absorbance"].iloc[0])
            last = float(series["absorbance"].iloc[-1])

            lines.append(
                f"pH {p:g}, {group}: "
                f"{series.time.iloc[0]:g}–"
                f"{series.time.iloc[-1]:g} s, "
                f"A 초기={first:.6f}, A 최종={last:.6f}"
            )

        lines.extend(
            [
                "",
                "[해석상 한계]",
                "조건별 종료 시간이 다를 수 있습니다.",
                (
                    "정규화는 pH별 화학종·몰흡광계수 차이를 "
                    "제거하지 않습니다."
                ),
                (
                    "특정 파장 흡광도 감소를 실제 분해량으로 "
                    "단정하지 않습니다."
                ),
                "모형 적합만으로 실제 반응 차수를 확정하지 않습니다.",
                "반복 실험 없이 통계적 유의성을 주장하지 않습니다.",
                (
                    "보간·보정값이 있으면 처리 방법을 "
                    "별도로 기록해야 합니다."
                ),
            ]
        )

    st.download_button(
        "TXT 요약 보고서 다운로드",
        "\n".join(lines).encode("utf-8-sig"),
        file_name="anthocyanin_report.txt",
        mime="text/plain",
        use_container_width=True,
        key="download_report",
    )


# =========================================================
# 하단 안내
# =========================================================
st.markdown("---")
st.caption(
    "교육용 분석 도구 · "
    "가정 시뮬레이션과 실험 측정 결과를 구분하세요."
)
