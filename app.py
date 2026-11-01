import json
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from bio_connection_tab import render_bio_connection_tab


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="Anthocyanin Lab",
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
        border-radius: 8px 8px 0 0;
        white-space: nowrap;
        flex-shrink: 0;
        padding: 10px 14px;
    }

    .small {
        color: #9fb5cc;
        font-size: .88rem;
        line-height: 1.6;
    }

    .liquid-display {
        display: flex;
        align-items: center;
        gap: 24px;
        flex-wrap: wrap;
        padding: 16px 0;
    }

    .liquid-beaker {
        width: 160px;
        height: 200px;
        flex-shrink: 0;
        border: 3px solid #a9bfd4;
        border-top: 0;
        border-radius: 8px 8px 28px 28px;
        position: relative;
    }

    .liquid-description {
        flex: 1;
        min-width: 0;
        overflow-wrap: anywhere;
    }

    [data-testid="stDownloadButton"] button,
    [data-testid="stButton"] button {
        min-height: 44px;
    }

    /* 좁은 화면에서도 라디오 선택지가 잘리지 않게 */
    div[role="radiogroup"] {
        flex-wrap: wrap;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.8rem 2rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }

        h3 {
            font-size: 1.15rem !important;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 10px 12px;
            font-size: .9rem;
        }

        .liquid-display {
            flex-direction: column;
            align-items: center;
            gap: 12px;
        }

        .liquid-description {
            width: 100%;
            text-align: center;
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
st.title("pH 조건에 따른 안토시아닌의 분자 구조 변화와 과산화수소에 의한 산화 반응 속도 시뮬레이션")
st.caption(
    "pH-dependent anthocyanin stability · "
    "H₂O₂ treatment · absorbance kinetics"
)
st.caption("made by 홍의찬, 조윤찬")

mobile_view = st.checkbox(
    "📱 모바일 버전",
    value=False,
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
        d[column] = d[column].replace(
            [np.inf, -np.inf], np.nan
        )

    d["group"] = d["group"].astype("string").str.strip()
    d["group"] = d["group"].replace("", pd.NA)

    d = (
        d.dropna(subset=["time", "pH", "group", "absorbance"])
        .sort_values(["pH", "group", "time"], kind="stable")
        .reset_index(drop=True)
    )

    if d.empty:
        raise ValueError("정리 후 유효한 데이터가 없습니다.")

    d["A_norm"] = (
        d.groupby(["pH", "group"])["absorbance"]
        .transform(
            lambda s: (
                s / s.iloc[0]
                if s.iloc[0] != 0
                else np.nan
            )
        )
    )

    return d


def model_fit(t, a):
    """기존 변환 선형회귀 방식의 모형 비교."""
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)

    if len(a) < 3 or len(np.unique(t)) < 3:
        return pd.DataFrame()

    models = []

    # 감소 제약을 적용하지 않으므로 k가 음수일 수 있음.
    # 음수 k는 증가 추세를 의미하며 탈색 속도상수로 해석하지 않음.
    p = np.polyfit(t, a, 1)
    models.append(
        (
            "0차 선형",
            np.polyval(p, t),
            -p[0],
            "A/s",
            "A=b−kt",
        )
    )

    if np.all(a > 0):
        p = np.polyfit(t, np.log(a), 1)

        with np.errstate(over="ignore", invalid="ignore"):
            predicted = np.exp(np.polyval(p, t))

        models.append(
            (
                "1차 지수형",
                predicted,
                -p[0],
                "s⁻¹",
                "A=B·exp(−kt)",
            )
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
        r2 = (
            1 - np.sum(error ** 2) / ss
            if ss > 0
            else np.nan
        )

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
    """모바일 보기에서는 그래프 높이와 글자 크기를 확대."""
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

    # 기존 requirements.txt의 Streamlit 버전과 호환
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
    """모바일 보기에서는 지표를 세로로 표시."""
    if mobile_view:
        for label, value in items:
            st.metric(label, value)
    else:
        columns = st.columns(len(items))
        for column, (label, value) in zip(columns, items):
            column.metric(label, value)


def plot_data(d, normalized=False, reference=False):
    fig, ax = make_figure()

    y_column = "A_norm" if normalized else "absorbance"
    mapping = []

    for index, ((p, group), series) in enumerate(
        d.groupby(["pH", "group"]), start=1
    ):
        # 한글 group 이름도 깨지지 않게 범례에는 숫자 ID 사용
        curve_id = f"G{index}"

        ax.plot(
            series["time"],
            series[y_column],
            marker=".",
            markersize=3,
            linewidth=1.7,
            label=f"{curve_id} | pH {p:g}",
        )

        mapping.append(
            {
                "그래프 ID": curve_id,
                "pH": p,
                "group": group,
            }
        )

    if reference:
        ax.axhline(
            1,
            color="gray",
            linestyle="--",
            linewidth=0.8,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(
        "Relative absorbance, A/A0"
        if normalized
        else "Absorbance, A"
    )

    add_legend(ax)
    show_figure(fig)

    return pd.DataFrame(mapping)


# =========================================================
# 사이드바
# =========================================================
st.sidebar.header("실험 조건")

ph = st.sidebar.slider(
    "pH",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.1,
)

buffer = st.sidebar.number_input(
    "완충용액 부피 (mL)",
    min_value=1.0,
    max_value=500.0,
    value=45.0,
    step=1.0,
)

extract = st.sidebar.number_input(
    "안토시아닌 추출액 부피 (mL)",
    min_value=0.1,
    max_value=100.0,
    value=5.0,
    step=0.5,
)

treated = st.sidebar.toggle(
    "H₂O₂ 처리군",
    value=True,
)

duration = st.sidebar.slider(
    "관찰 시간 (초)",
    min_value=30,
    max_value=3600,
    value=300,
    step=30,
)

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
    [
        "시뮬레이션",
        "분석",
        "비교",
        "화학 개념",
        "생명 과학적 응용",
        "저장",
    ]
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
        "색상은 정성적 교육용 근사 표현입니다."
    )

    # 교육용 가정값: 실제 화학 반응에서 검증된 값이 아님
    A0 = 0.65 * (
        0.75 + 0.25 * np.exp(-((ph - 7) / 2.4) ** 2)
    )
    k = (0.0018 if treated else 0.00015) * (
        1 + 0.10 * abs(ph - 7)
    )

    t = np.linspace(0, duration, 160)
    rel = np.exp(-k * t)
    A = A0 * rel

    st.markdown("#### 기본 시뮬레이션")
    st.caption("1차 지수 감소를 가정한 곡선입니다.")

    fig, ax = make_figure()

    treatment_label = (
        "H2O2-treated" if treated else "Untreated control"
    )

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

    # 관찰 시간을 줄일 때 현재 시점이 범위를 넘지 않도록 보정
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

    # 기존 색상 표현 방식 유지
    fade = (1 - current_rel) * (
        1 if treated else 0.18
    )
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
    border: 1px solid #35516d;
    border-radius: 20px;
    background: linear-gradient(135deg, #10233e, #0b192e);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
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
    border: 3px solid #b7cfe3;
    border-top: none;
    border-radius: 0 0 22px 22px;
    background: rgba(220, 239, 255, 0.035);
    box-shadow: inset 8px 0 14px rgba(255, 255, 255, 0.04);
}}

.beaker-rim {{
    position: absolute;
    top: -5px;
    left: -9px;
    right: -9px;
    height: 8px;
    border: 2px solid #c9dfef;
    border-radius: 50%;
    background: #142a43;
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
    border-top: 1px solid rgba(255, 255, 255, 0.65);
    filter: brightness(1.12);
}}

.beaker-shine {{
    position: absolute;
    top: 16px;
    bottom: 18px;
    left: 12px;
    width: 8px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.16);
    z-index: 3;
}}

.beaker-mark {{
    position: absolute;
    right: 0;
    width: 23px;
    height: 1px;
    background: rgba(235, 246, 255, 0.75);
    z-index: 3;
}}

.beaker-platform {{
    width: 195px;
    height: 10px;
    margin-top: 10px;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.25);
}}

.beaker-label {{
    margin-top: 12px;
    color: #c4d2e3;
    font-size: 0.85rem;
    text-align: center;
}}

.beaker-details {{
    width: 100%;
    max-width: 340px;
    min-width: 0;
}}

.beaker-badge {{
    display: inline-block;
    padding: 5px 11px;
    border: 1px solid #52708d;
    border-radius: 20px;
    background: #193450;
    color: #d9edff;
    font-size: 0.82rem;
    margin-bottom: 14px;
}}

.beaker-heading {{
    margin-bottom: 16px;
    color: #f1f5f9;
    font-size: 1.25rem;
    font-weight: 700;
}}

.beaker-value-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid #30465f;
}}

.beaker-value-label {{
    color: #c4d2e3;
    font-size: 0.93rem;
}}

.beaker-value {{
    color: #f1f5f9;
    font-size: 1.3rem;
    font-weight: 700;
    white-space: nowrap;
}}

.beaker-note {{
    margin-top: 16px;
    color: #c4d2e3;
    font-size: 0.83rem;
    line-height: 1.7;
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
# 2. 데이터 분석
# =========================================================
with tabs[1]:
    st.subheader("실험 데이터 분석")

    st.write(
        "CSV 필수 열: `time`, `pH`, `group`, `absorbance`"
    )
    st.caption(
        "time은 초 단위입니다. 서로 다른 반복 실험은 group 이름을 "
        "구분해 올리세요. 예: H2O2_rep1, H2O2_rep2"
    )

    file = st.file_uploader(
        "측정 CSV 업로드",
        type=["csv"],
        key="measurement_csv",
    )

    # 파일을 지우거나 잘못된 파일로 바꿨을 때 이전 데이터가 남지 않게 함
    st.session_state.pop("data", None)
    st.session_state.pop("data_quality", None)

    if file is not None:
        try:
            raw = pd.read_csv(file)
            required = {"time", "pH", "group", "absorbance"}
            missing = required - set(raw.columns)

            if missing:
                st.error(
                    "필수 열 누락: " + ", ".join(sorted(missing))
                )
            else:
                cleaned = prepare(raw)

                quality = {
                    "original_rows": len(raw),
                    "missing_rows": int(
                        raw.isna().any(axis=1).sum()
                    ),
                    "removed_rows": len(raw) - len(cleaned),
                }

                st.session_state["data"] = cleaned
                st.session_state["data_quality"] = quality

                st.success(
                    f"{len(cleaned)}개 유효 행을 불러왔습니다."
                )

        except Exception as error:
            st.error(f"CSV 읽기 오류: {error}")

    if "data" in st.session_state:
        d = st.session_state["data"]
        quality = st.session_state["data_quality"]

        show_metrics(
            [
                ("유효 행", len(d)),
                ("원본 결측 포함 행", quality["missing_rows"]),
                ("제외 행", quality["removed_rows"]),
                (
                    "시간 범위",
                    f"{d.time.min():g}–{d.time.max():g} s",
                ),
            ]
        )

        with st.expander("데이터 품질 상세"):
            st.write(f"원본 행 수: {quality['original_rows']}")
            st.write(f"완전히 같은 중복 행: {int(d.duplicated().sum())}")

            duplicate_times = int(
                d.duplicated(
                    subset=["pH", "group", "time"],
                    keep=False,
                ).sum()
            )

            st.write(
                f"같은 조건·시간에 여러 값이 있는 행: {duplicate_times}"
            )

            if duplicate_times:
                st.warning(
                    "별도 반복 실험이 같은 group에 섞였는지 확인하세요. "
                    "현재 앱은 각 pH·group을 하나의 시계열로 처리합니다."
                )

            if (d["absorbance"] < 0).any():
                st.warning(
                    "음수 흡광도가 있습니다. 블랭크 보정과 측정값을 확인하세요."
                )

            if d["A_norm"].isna().any():
                st.warning(
                    "초기 흡광도가 0인 조건은 정규화값을 계산하지 않습니다."
                )

        st.caption("휴대폰에서는 표를 좌우로 밀어 전체 열을 볼 수 있습니다.")
        st.dataframe(
            d,
            use_container_width=True,
            height=220,
        )

        st.markdown("#### 원래 흡광도")
        mapping = plot_data(d)

        with st.expander("그래프 ID와 실험 조건"):
            st.dataframe(
                mapping,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### 정규화 흡광도")

        plot_data(d, normalized=True)

        st.caption(
            "각 pH·group의 첫 측정값을 A₀로 사용합니다. "
            "A/A₀ = 0.8은 초기 흡광도 대비 20% 감소를 뜻합니다. "
            "초기값보다 증가하면 1을 넘을 수 있습니다."
        )

        summaries = []
        fits = []

        for (p, group), series in d.groupby(["pH", "group"]):
            series = series.sort_values("time")

            if len(series) < 2:
                continue

            a0 = float(series["absorbance"].iloc[0])
            af = float(series["absorbance"].iloc[-1])
            dt = float(
                series["time"].iloc[-1]
                - series["time"].iloc[0]
            )

            summaries.append(
                {
                    "pH": p,
                    "group": group,
                    "초기 A": a0,
                    "최종 A": af,
                    "상대 감소율(%)": (
                        (a0 - af) / a0 * 100
                        if a0 != 0
                        else np.nan
                    ),
                    "평균 감소 속도(A/s)": (
                        (a0 - af) / dt
                        if dt > 0
                        else np.nan
                    ),
                }
            )

            mf = model_fit(
                series["time"].to_numpy()
                - series["time"].iloc[0],
                series["absorbance"].to_numpy(),
            )

            if not mf.empty:
                best = (
                    mf.sort_values(
                        ["RMSE", "R²"],
                        ascending=[True, False],
                    )
                    .iloc[0]
                )

                fits.append(
                    {
                        "pH": p,
                        "group": group,
                        "적합도 기준 모형": best["모형"],
                        "겉보기 k": best["겉보기 k"],
                        "k 단위": best["k 단위"],
                        "R²": best["R²"],
                        "RMSE": best["RMSE"],
                    }
                )

        st.markdown("#### 조건별 요약")
        st.dataframe(
            pd.DataFrame(summaries).round(6),
            use_container_width=True,
        )

        if fits:
            st.markdown("#### 모형 비교 결과")
            st.dataframe(
                pd.DataFrame(fits).round(6),
                use_container_width=True,
            )
            st.caption(
                "기존 변환 선형회귀 방식으로 적합합니다. "
                "0차는 A, 1차는 ln(A), 2차는 1/A를 회귀하며 "
                "RMSE는 원래 흡광도 척도에서 계산합니다. "
                "음수 k는 해당 모형에서 증가 추세를 뜻하므로 "
                "탈색 속도상수로 해석하지 않습니다."
            )

        st.caption(
            "적합도가 높은 모형이 실제 반응 차수임을 증명하지는 않습니다. "
            "잔차 패턴과 반복 실험도 검토해야 합니다."
        )

        st.download_button(
            "정리된 데이터 CSV 다운로드",
            d.to_csv(index=False).encode("utf-8-sig"),
            file_name="anthocyanin_processed.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("해석상 주의점", expanded=True):
            st.markdown(
                """
- 정규화는 초기값 차이를 고려하지만 pH에 따른 화학종과 몰흡광계수 차이를 제거하지 않습니다.
- 특정 파장의 흡광도 감소를 실제 분해량과 동일시하지 않습니다.
- 반복 실험이 없다면 통계적 유의성을 주장하지 않습니다.
- 첫 측정값의 오차는 그 값을 기준으로 한 정규화 결과 전체에 영향을 줄 수 있습니다.
"""
            )

    else:
        st.info(
            "CSV를 업로드하면 데이터 품질 검사, 원본·정규화 그래프와 "
            "속도 분석이 표시됩니다."
        )


# =========================================================
# 3. 조건별 비교
# =========================================================
with tabs[2]:
    st.subheader("조건별 비교")

    if "data" in st.session_state:
        d = st.session_state["data"]

        mapping = plot_data(
            d,
            normalized=True,
            reference=True,
        )

        with st.expander("그래프 ID와 실험 조건"):
            st.dataframe(
                mapping,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("분석 탭에서 CSV를 먼저 업로드하세요.")

    st.markdown(
        """
1. 동일 pH에서 H₂O₂ 처리군과 대조군의 상대 흡광도 변화를 비교합니다.
2. 각 조건의 첫 흡광도를 기준으로 정규화합니다.
3. pH별 차이는 화학종 분포와 몰흡광계수 차이를 고려해 해석합니다.
4. 반복 실험이 없다면 통계적 유의성을 단정하지 않습니다.
"""
    )


# =========================================================
# 4. 화학 개념
# =========================================================
with tabs[3]:
    st.subheader("화학 개념 학습")

    concepts = [
        (
            "1. 안토시아닌의 산·염기 반응",
            "안토시아닌은 pH에 따라 양성자 이동, 수화 반응 및 "
            "구조 전환 등의 영향을 받아 여러 화학종 사이의 평형이 "
            "변화한다. 대표적인 구조는 플라빌륨 양이온, 퀴노이드 형태, "
            "헤미케탈, 칼콘이며 pH와 용액 환경에 따라 서로 다른 "
            "비율로 분포한다. 각 화학종은 빛 흡수 특성도 다르다.",
        ),
        (
            "2. 산성 조건에서의 안토시아닌",
            "충분히 산성인 조건에서는 플라빌륨 양이온이 상대적으로 "
            "안정화되어 붉은색 계열을 나타낼 수 있다. "
            "실제 색상과 화학종 분포는 안토시아닌 종류, 추출액 조성, "
            "농도 및 실험 조건에 따라 달라진다.",
        ),
        (
            "3. pH 4~7에서의 안토시아닌",
            "pH가 증가하면 플라빌륨 양이온의 비율이 감소하고 "
            "퀴노이드 형태 등 여러 화학종의 비율이 변화한다. "
            "보라색·청색 변화뿐 아니라 수화에 따른 탈색도 가능하며, "
            "특정 pH에서 특정 색이 반드시 나타나는 것은 아니다.",
        ),
        (
            "4. pH 7 이상에서의 안토시아닌",
            "염기성 조건에서는 퀴노이드 형태와 칼콘 계열 등 "
            "여러 화학종의 분포가 달라질 수 있다. "
            "강한 알칼리성 환경에서는 구조 안정성이 낮아지고 "
            "비가역적 분해가 일어날 가능성도 있다.",
        ),
        (
            "5. H₂O₂에 의한 안토시아닌 변화",
            "H₂O₂는 산화제로 작용할 수 있으며 안토시아닌의 "
            "구조 변화나 분해를 유발할 수 있다. 공액 전자계가 "
            "변화하면 가시광선 흡수 특성도 달라진다. 구체적인 "
            "반응 경로와 생성물은 종류와 조건에 따라 달라지며 "
            "본 앱에서 직접 확인하지 않는다.",
        ),
        (
            "6. 흡광 특성과 Beer–Lambert 법칙",
            "흡광도와 투과도는 A = −log₁₀(T), T = I/I₀의 관계를 "
            "가지며 Beer–Lambert 법칙은 A = εlc로 표현된다. "
            "흡광도는 무차원량이다. 일정한 화학종과 조건에서 "
            "흡광도는 농도에 비례하지만, 서로 다른 pH의 흡광도를 "
            "곧바로 농도 차이로 해석해서는 안 된다.",
        ),
        (
            "7. 화학 평형",
            "화학 평형은 가역 반응에서 정반응과 역반응의 속도가 "
            "같아 물질의 농도가 거시적으로 일정하게 유지되는 "
            "동적 평형이다. H₂O₂에 의한 모든 변화가 "
            "가역 평형인 것은 아니다.",
        ),
        (
            "8. 르샤틀리에 원리",
            "평형 상태의 화학계에 농도나 온도 등의 변화가 "
            "가해지면 그 변화의 영향을 줄이는 방향으로 "
            "새로운 평형에 도달하는 경향이 있다. "
            "pH 변화에 따른 안토시아닌 화학종 분포를 "
            "이해하는 데 활용한다.",
        ),
        (
            "9. 산화·환원 반응",
            "산화·환원 반응은 전자 이동과 산화수 변화로 설명할 수 "
            "있다. 산화는 전자를 잃거나 산화수가 증가하는 과정이며 "
            "환원은 전자를 얻거나 산화수가 감소하는 과정이다. "
            "H₂O₂는 본 실험 조건에서 산화제로 작용할 수 있다.",
        ),
        (
            "10. 반응 속도",
            "반응 속도는 단위 시간당 반응물의 농도 감소량 또는 "
            "생성물의 농도 증가량으로 나타낸다. 본 앱에서는 "
            "흡광도 변화율로 겉보기 탈색 속도를 비교한다. "
            "평균 탈색 속도는 −(A₂−A₁)/(t₂−t₁), "
            "순간 탈색 속도는 −dA/dt로 표현할 수 있다.",
        ),
        (
            "11. 완충 용액과 공통 이온 효과",
            "완충 용액은 약산과 짝염기 또는 약염기와 짝산으로 "
            "구성되어 산·염기 첨가에 따른 pH 변화를 억제한다. "
            "HA/A⁻ 완충계에서는 A⁻ + H⁺ → HA, "
            "HA + OH⁻ → A⁻ + H₂O 반응이 일어난다. "
            "공통 이온 효과는 이미 존재하는 이온과 같은 이온을 "
            "첨가했을 때 평형이 이동하는 현상이며, "
            "완충 작용과 관련되지만 동일한 개념은 아니다.",
        ),
    ]

    for title, body in concepts:
        with st.expander(title):
            st.write(body)


# =========================================================
# 5. 생명과학 연결
# =========================================================
with tabs[4]:
    render_bio_connection_tab()


# =========================================================
# 6. 보고서·조건 저장
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
    }

    with st.expander("현재 실험 조건", expanded=True):
        st.json(settings)

    st.download_button(
        "실험 조건 JSON 저장",
        json.dumps(
            settings,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        file_name="anthocyanin_conditions.json",
        mime="application/json",
        use_container_width=True,
    )

    jf = st.file_uploader(
        "저장한 조건 JSON 불러오기",
        type=["json"],
        key="conditions_json",
    )

    if jf is not None:
        try:
            loaded = json.loads(jf.getvalue())
            st.write("불러온 조건")
            st.json(loaded)
            st.caption(
                "불러온 값은 확인용이며 사이드바에 자동 적용되지는 않습니다."
            )
        except Exception as error:
            st.error(f"JSON 읽기 오류: {error}")

    st.markdown("---")

    lines = [
        "pH 조건에 따른 안토시아닌의 분자 구조 변화와 과산화수소에 의한 산화 반응 속도 시뮬레이션",
        f"생성 시각: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "[사이드바 시뮬레이션 조건]",
        (
            f"pH {ph:g}, 완충용액 {buffer:g} mL, "
            f"추출액 {extract:g} mL"
        ),
        (
            f"H₂O₂ 처리: {'처리군' if treated else '미처리 조건'}, "
            f"관찰 시간 {duration} s, 온도 25 °C"
        ),
        "",
        "시뮬레이션은 교육용 가정 모델이며 실제 반응을 검증한 모델이 아닙니다.",
        "용액 부피는 표시·기록용이며 현재 흡광도 계산에 반영되지 않습니다.",
        "흡광도 기반 속도는 겉보기 탈색 속도로 해석해야 합니다.",
        "사이드바 조건은 업로드 CSV의 실험 조건을 대신하지 않습니다.",
    ]

    if "data" in st.session_state:
        d = st.session_state["data"]
        lines.extend(
            [
                "",
                "[업로드 데이터]",
                f"유효 행 수: {len(d)}",
                (
                    f"시간 범위: {d.time.min():g}"
                    f"–{d.time.max():g} s"
                ),
                "정규화 방식: 각 pH·group의 첫 흡광도를 기준으로 A/A₀ 계산",
            ]
        )

    st.download_button(
        "TXT 요약 보고서 다운로드",
        "\n".join(lines).encode("utf-8-sig"),
        file_name="anthocyanin_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.caption(
        "요약 보고서는 TXT 파일로 제공합니다. "
        "PDF가 필요하면 내려받은 문서를 편집기에서 열어 PDF로 저장하세요."
    )


# =========================================================
# 하단 안내
# =========================================================
st.markdown("---")
st.markdown(
    '<p class="small">'
    "pH 조건에 따른 안토시아닌의 분자 구조 변화와 과산화수소에 의한 산화 반응 속도 시뮬레이션")
st.markdown(
    "*시뮬레이션과 실험 결과는 항상 같지 않습니다.*"
    "</p>",)
unsafe_allow_html=True,
