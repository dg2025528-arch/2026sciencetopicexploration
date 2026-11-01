import io
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="Anthocyanin Lab V4",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #071225, #0b1830 55%, #101f38);
        color: #e8f0fa;
    }
    [data-testid="stSidebar"] {
        background: #0a162b;
        border-right: 1px solid #294260;
    }
    h1, h2, h3 {color: #f1f6ff !important;}
    div[data-testid="stMetric"] {
        background: #10233e;
        border: 1px solid #294260;
        padding: 12px;
        border-radius: 10px;
    }
    .small {color: #9fb5cc; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULTS = {
    "ph": 7.0,
    "buffer": 45.0,
    "extract": 5.0,
    "treated": True,
    "duration": 300,
    "model": "1차 지수형",
    "view_time": 0,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

MODEL_NAMES = ["0차 선형", "1차 지수형", "2차 역수형"]
GROUP_KEYS = ["pH", "group", "replicate"]


# =========================================================
# 공통 함수
# =========================================================
def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def figure_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    return buffer.getvalue()


def make_figure(height=3.8):
    fig, ax = plt.subplots(figsize=(9, height))
    fig.patch.set_facecolor("#10233e")
    ax.set_facecolor("#10233e")
    ax.tick_params(colors="#dce7f5")
    ax.xaxis.label.set_color("#dce7f5")
    ax.yaxis.label.set_color("#dce7f5")
    ax.title.set_color("#f1f6ff")

    for spine in ax.spines.values():
        spine.set_color("#66809e")

    ax.grid(alpha=0.18, color="#b7c8db")
    ax.set_axisbelow(True)
    return fig, ax


def add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    legend = ax.legend(
        fontsize=8,
        facecolor="#e8eef7",
        edgecolor="#8298b2",
    )
    for text in legend.get_texts():
        text.set_color("#13243a")


def ph_color(ph, fade):
    anchors = [
        (4, (183, 55, 83)),
        (7, (61, 105, 170)),
        (10, (82, 142, 102)),
    ]
    left, right = (
        (anchors[0], anchors[1])
        if ph <= 7
        else (anchors[1], anchors[2])
    )

    fraction = np.clip(
        (ph - left[0]) / (right[0] - left[0]), 0, 1
    )
    base = [
        left[1][i] * (1 - fraction) + right[1][i] * fraction
        for i in range(3)
    ]
    fade = float(np.clip(fade, 0, 1))
    rgb = [int(c * (1 - fade) + 225 * fade) for c in base]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def decay_curve(name, t, a0, k):
    t = np.asarray(t, dtype=float)

    if name == "0차 선형":
        return np.maximum(a0 - k * t, 0)
    if name == "1차 지수형":
        return a0 * np.exp(-k * t)
    return a0 / (1 + k * a0 * t)


MODEL_INFO = {
    "0차 선형": {
        "unit": "A/s",
        "equation": "A = max(A₀ − k·t, 0)",
    },
    "1차 지수형": {
        "unit": "s⁻¹",
        "equation": "A = A₀ exp(−k·t)",
    },
    "2차 역수형": {
        "unit": "A⁻¹·s⁻¹",
        "equation": "A = A₀ / (1 + k·A₀·t)",
    },
}


# =========================================================
# 合成 예시 데이터
# =========================================================
@st.cache_data
def sample_data():
    rng = np.random.default_rng(42)
    rows = []
    times = np.arange(0, 601, 30)

    for ph in [4.0, 7.0, 10.0]:
        for group in ["control", "H2O2"]:
            for replicate in [1, 2, 3]:
                a0 = 0.62 * rng.normal(1, 0.025)
                k = (
                    0.0018 if group == "H2O2" else 0.00015
                ) * (1 + 0.1 * abs(ph - 7))
                k *= rng.normal(1, 0.06)

                absorbance = (
                    a0 * np.exp(-k * times)
                    + rng.normal(0, 0.006, len(times))
                )

                for time, value in zip(times, absorbance):
                    rows.append(
                        {
                            "time": time,
                            "pH": ph,
                            "group": group,
                            "replicate": replicate,
                            "absorbance": round(float(value), 6),
                        }
                    )

    return pd.DataFrame(rows)


# =========================================================
# CSV 정리 및 품질 검사
# =========================================================
def prepare_data(raw):
    required = {"time", "pH", "group", "absorbance"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            "필수 열 누락: " + ", ".join(sorted(missing))
        )

    d = raw.copy()
    original_count = len(d)
    original_missing = int(d.isna().any(axis=1).sum())

    if "replicate" not in d.columns:
        d["replicate"] = "1"

    for column in ["time", "pH", "absorbance"]:
        d[column] = pd.to_numeric(d[column], errors="coerce")
        d[column] = d[column].replace([np.inf, -np.inf], np.nan)

    for column in ["group", "replicate"]:
        d[column] = d[column].astype("string").str.strip()
        d[column] = d[column].replace("", pd.NA)

    valid = d[
        ["time", "pH", "group", "replicate", "absorbance"]
    ].notna().all(axis=1)

    d = d.loc[valid].copy()
    if d.empty:
        raise ValueError("정리 후 유효한 데이터가 없습니다.")

    d = d.sort_values(GROUP_KEYS + ["time"], kind="stable")
    d["A_norm"] = np.nan

    ambiguous = 0
    nonpositive_baseline = 0

    for _, series in d.groupby(GROUP_KEYS, sort=False):
        initial_time = series["time"].min()
        baseline_rows = series.loc[series["time"] == initial_time]

        # 첫 시점의 중복 측정은 평균값을 정규화 기준으로 사용
        baseline = float(baseline_rows["absorbance"].mean())

        if len(baseline_rows) > 1:
            ambiguous += 1

        if baseline > 0:
            d.loc[series.index, "A_norm"] = (
                series["absorbance"] / baseline
            )
        else:
            nonpositive_baseline += 1

    quality = {
        "원본 행": original_count,
        "원본 결측 포함 행": original_missing,
        "유효 행": len(d),
        "제외 행": original_count - len(d),
        "동일 실험·시간 중복 행": int(
            d.duplicated(GROUP_KEYS + ["time"], keep=False).sum()
        ),
        "음수 시간 행": int((d["time"] < 0).sum()),
        "음수 흡광도 행": int((d["absorbance"] < 0).sum()),
        "비양수 초기값 실험": nonpositive_baseline,
        "첫 시점 중복 실험": ambiguous,
    }

    return d, quality


@st.cache_data
def read_csv_data(content):
    raw = pd.read_csv(io.BytesIO(content))
    return prepare_data(raw)


# =========================================================
# 모형 적합
# =========================================================
def fit_models(series):
    """
    모든 모형을 원래 흡광도 공간에서 적합.
    A0와 k를 함께 추정하며 A0 > 0, k >= 0으로 제한.
    """
    time = series["time"].to_numpy(dtype=float)
    values = series["absorbance"].to_numpy(dtype=float)
    elapsed = time - time.min()

    if len(np.unique(elapsed)) < 3:
        return [], {}, "서로 다른 시간점이 3개 미만입니다."

    if np.any(values < 0):
        return [], {}, "음수 흡광도가 있어 감소 모형 적합을 건너뜁니다."

    if np.max(values) <= 0:
        return [], {}, "양의 흡광도가 없어 모형 적합을 건너뜁니다."

    slope = float(np.polyfit(elapsed, values, 1)[0])
    span = float(elapsed.max())
    a0_guess = max(float(values[elapsed == 0].mean()), 1e-6)
    rate_guess = max(-slope, a0_guess / max(span, 1) * 0.01)

    rows = []
    curves = {}

    for name in MODEL_NAMES:
        if name == "0차 선형":
            k_guess = rate_guess
        elif name == "1차 지수형":
            k_guess = rate_guess / a0_guess
        else:
            k_guess = rate_guess / a0_guess**2

        def function(t, a0, k):
            return decay_curve(name, t, a0, k)

        try:
            parameters, _ = curve_fit(
                function,
                elapsed,
                values,
                p0=[a0_guess, max(k_guess, 1e-10)],
                bounds=([1e-12, 0], [np.inf, np.inf]),
                maxfev=20000,
            )

            fitted_a0, fitted_k = map(float, parameters)
            predicted = function(elapsed, fitted_a0, fitted_k)
            residual = values - predicted
            sse = float(np.sum(residual**2))
            sst = float(np.sum((values - values.mean())**2))

            half_life = np.nan
            if name == "1차 지수형" and fitted_k > 1e-12:
                half_life = np.log(2) / fitted_k

            rows.append(
                {
                    "모형": name,
                    "추정 A₀": fitted_a0,
                    "겉보기 k": fitted_k,
                    "k 단위": MODEL_INFO[name]["unit"],
                    "R²": 1 - sse / sst if sst > 0 else np.nan,
                    "RMSE": np.sqrt(sse / len(values)),
                    "1차 반감기(s)": half_life,
                    "원자료 선형 추세": (
                        "증가" if slope > 0 else "감소 또는 일정"
                    ),
                }
            )

            curves[name] = {
                "a0": fitted_a0,
                "k": fitted_k,
                "predicted": predicted,
                "residual": residual,
            }

        except (RuntimeError, ValueError, FloatingPointError):
            continue

    message = ""
    if slope > 0:
        message = (
            "원자료에 증가 추세가 있습니다. 감소 제약 모형의 "
            "속도상수와 반감기를 탈색 특성으로 해석하지 마세요."
        )
    if not rows:
        message = "모든 모형의 적합에 실패했습니다."

    return rows, curves, message


@st.cache_data
def analyze_data(d):
    summaries = []
    all_fits = []
    details = {}

    for key, series in d.groupby(GROUP_KEYS, sort=True):
        series = series.sort_values("time")
        first_time = float(series["time"].min())
        last_time = float(series["time"].max())

        a0 = float(
            series.loc[series["time"] == first_time, "absorbance"].mean()
        )
        af = float(
            series.loc[series["time"] == last_time, "absorbance"].mean()
        )
        elapsed = last_time - first_time
        identifiers = dict(zip(GROUP_KEYS, key))

        summaries.append(
            {
                **identifiers,
                "측정 행": len(series),
                "서로 다른 시간점": series["time"].nunique(),
                "첫 측정 시각(s)": first_time,
                "마지막 측정 시각(s)": last_time,
                "초기 A": a0,
                "최종 A": af,
                "상대 감소율(%)": (
                    (a0 - af) / a0 * 100 if a0 > 0 else np.nan
                ),
                "평균 감소 속도(A/s)": (
                    (a0 - af) / elapsed if elapsed > 0 else np.nan
                ),
            }
        )

        rows, curves, message = fit_models(series)
        for row in rows:
            all_fits.append({**identifiers, **row})

        details[key] = {
            "data": series,
            "curves": curves,
            "message": message,
        }

    return pd.DataFrame(summaries), pd.DataFrame(all_fits), details


# =========================================================
# 조건 저장 및 적용
# =========================================================
def current_settings():
    return {
        "pH": st.session_state["ph"],
        "buffer_volume_mL": st.session_state["buffer"],
        "anthocyanin_extract_volume_mL": st.session_state["extract"],
        "H2O2_treatment": st.session_state["treated"],
        "observation_time_s": st.session_state["duration"],
        "temperature_C": 25,
        "model": st.session_state["model"],
    }


def apply_settings(content):
    try:
        obj = json.loads(content)
        if not isinstance(obj, dict):
            raise ValueError("JSON 최상위 값은 객체여야 합니다.")

        mapping = {
            "pH": ("ph", 4.0, 10.0),
            "buffer_volume_mL": ("buffer", 1.0, 500.0),
            "anthocyanin_extract_volume_mL": ("extract", 0.1, 100.0),
            "observation_time_s": ("duration", 30, 3600),
        }

        updates = {}

        for source, (target, lower, upper) in mapping.items():
            if source not in obj:
                continue
            if isinstance(obj[source], bool):
                raise ValueError(f"{source}: 숫자가 필요합니다.")

            value = float(obj[source])
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(
                    f"{source}: {lower}~{upper} 범위여야 합니다."
                )

            if target == "duration":
                if not value.is_integer():
                    raise ValueError("관찰 시간은 정수여야 합니다.")
                value = int(value)

            updates[target] = value

        if "H2O2_treatment" in obj:
            if not isinstance(obj["H2O2_treatment"], bool):
                raise ValueError("H2O2_treatment는 true/false여야 합니다.")
            updates["treated"] = obj["H2O2_treatment"]

        if "model" in obj:
            if obj["model"] not in MODEL_NAMES:
                raise ValueError("지원하지 않는 모형입니다.")
            updates["model"] = obj["model"]

        if "temperature_C" in obj and obj["temperature_C"] != 25:
            raise ValueError("이 앱은 온도 25 °C만 지원합니다.")

        st.session_state.update(updates)
        st.session_state["view_time"] = 0
        st.session_state["settings_notice"] = "조건을 적용했습니다."
        st.session_state.pop("settings_error", None)

    except (ValueError, TypeError, UnicodeDecodeError) as error:
        st.session_state["settings_error"] = str(error)


# =========================================================
# 화면 및 사이드바
# =========================================================
st.title("ANTHOCYANIN LAB")
st.caption(
    "pH-dependent anthocyanin stability · "
    "H₂O₂ treatment · absorbance kinetics"
)
st.info(
    "시뮬레이션은 교육용 가정 모델입니다. 실제 반응을 예측하거나 "
    "검증한 모델이 아니며, 측정 데이터 분석과 구분해야 합니다."
)

with st.sidebar:
    st.header("실험 조건")
    st.slider("설정 pH", 4.0, 10.0, step=0.1, key="ph")
    st.number_input(
        "완충용액 부피 (mL)",
        min_value=1.0,
        max_value=500.0,
        step=1.0,
        key="buffer",
    )
    st.number_input(
        "안토시아닌 추출액 부피 (mL)",
        min_value=0.1,
        max_value=100.0,
        step=0.5,
        key="extract",
    )
    st.toggle("H₂O₂ 처리 시나리오", key="treated")
    st.number_input(
        "관찰 시간 (초)",
        min_value=30,
        max_value=3600,
        step=30,
        key="duration",
    )
    st.markdown("**온도: 25 °C 고정**")
    st.caption(
        "설정 pH는 혼합 후에도 유지된다고 가정합니다. "
        "완충능과 H₂O₂ 첨가 부피는 계산하지 않습니다."
    )
    st.caption(
        "부피는 단순 희석에만 반영됩니다. "
        "H₂O₂ 처리 여부는 가정 시나리오이며 실제 농도를 뜻하지 않습니다."
    )

ph = st.session_state["ph"]
buffer_volume = st.session_state["buffer"]
extract_volume = st.session_state["extract"]
treated = st.session_state["treated"]
duration = st.session_state["duration"]

tabs = st.tabs(
    [
        "실험 시뮬레이션",
        "데이터 분석",
        "조건별 비교",
        "화학 개념",
        "보고서·조건 저장",
    ]
)

data = None
quality = {}
summaries = pd.DataFrame()
fits = pd.DataFrame()
details = {}


# =========================================================
# 탭 1: 시뮬레이션
# =========================================================
with tabs[0]:
    st.subheader("교육용 흡광도 시뮬레이션")

    st.radio(
        "수학적 모형",
        MODEL_NAMES,
        key="model",
        horizontal=True,
    )
    model = st.session_state["model"]

    total_volume = buffer_volume + extract_volume
    extract_fraction = extract_volume / total_volume

    # 기본 조건: 완충용액 45 mL + 추출액 5 mL
    reference_fraction = 0.1
    base_a0 = 0.65 * (
        0.75 + 0.25 * np.exp(-((ph - 7) / 2.4)**2)
    )
    a0 = base_a0 * extract_fraction / reference_fraction

    # 검증되지 않은 교육용 계수
    q = (0.0018 if treated else 0.00015) * (
        1 + 0.10 * abs(ph - 7)
    )

    # 모형별 단위는 다르지만 초기 상대 감소 속도는 q로 통일
    if model == "0차 선형":
        k = a0 * q
    elif model == "1차 지수형":
        k = q
    else:
        k = q / a0

    time = np.linspace(0, duration, 240)
    absorbance = decay_curve(model, time, a0, k)

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 용액 부피", f"{total_volume:.1f} mL")
    c2.metric("추출액 부피 비율", f"{extract_fraction:.1%}")
    c3.metric("초기 흡광도 · 가정값", f"{a0:.3f}")

    treatment_label = "H2O2-treated" if treated else "Untreated control"

    fig, ax = make_figure()
    ax.plot(
        time,
        absorbance,
        color="#63b3ff",
        linewidth=2.5,
        label=f"pH {ph:.1f} | {treatment_label} | assumed model",
    )
    ax.scatter([0], [a0], color="#63b3ff", zorder=3)
    ax.set(
        title="Educational simulation: absorbance decay",
        xlabel="Time (s)",
        ylabel="Absorbance, A (dimensionless)",
        xlim=(0, duration),
        ylim=(0, a0 * 1.15),
    )
    add_legend(ax)
    fig.tight_layout()
    simulation_png = figure_bytes(fig)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"{MODEL_INFO[model]['equation']} · "
        "모형별 초기 상대 감소 속도를 같게 맞춘 교육용 비교입니다. "
        "서로 다른 차수의 k는 단위가 다르므로 직접 비교하지 않습니다."
    )

    st.download_button(
        "시뮬레이션 그래프 PNG",
        simulation_png,
        "anthocyanin_simulation.png",
        "image/png",
    )

    if st.session_state["view_time"] > duration:
        st.session_state["view_time"] = duration

    now = st.slider(
        "색 변화 관찰 시점 (초)",
        min_value=0,
        max_value=duration,
        step=1,
        key="view_time",
    )

    current_a = float(decay_curve(model, now, a0, k))
    relative_a = current_a / a0
    liquid = ph_color(ph, 1 - relative_a)

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:24px">
          <div style="
            width:150px;height:200px;
            border:3px solid #a9bfd4;border-top:0;
            border-radius:8px 8px 28px 28px;
            background:linear-gradient(
                to bottom,transparent 0 24%,{liquid} 24% 100%
            );">
          </div>
          <div>
            <b>시점: {now} s</b>
            <p>가정 흡광도: {current_a:.3f}</p>
            <p>상대 흡광도 A/A₀: {relative_a:.3f}</p>
            <p class="small">
              용액 높이는 고정된 도식이며 실제 부피 눈금이 아닙니다.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if a0 > 2:
        st.warning(
            "가정 초기 흡광도가 2를 초과합니다. 실제 측정에서는 "
            "기기 선형 범위와 미광 영향을 확인하고 희석을 검토하세요."
        )

    st.warning(
        "pH별 초기값과 속도 계수는 임의의 교육용 가정값입니다. "
        "부피 효과는 원액 조성이 일정하다는 단순 희석 가정만 반영하며, "
        "화학종 평형·완충능·H₂O₂ 농도·실제 반응 경로는 계산하지 않습니다. "
        "화면 색상도 실제 스펙트럼으로 보정되지 않았습니다."
    )


# =========================================================
# 탭 2: 데이터 분석
# =========================================================
with tabs[1]:
    st.subheader("측정 데이터 분석")
    st.write(
        "필수 열: `time`, `pH`, `group`, `absorbance`  \n"
        "선택 열: `replicate` · 시간 단위: 초"
    )

    st.download_button(
        "합성 예시 CSV 다운로드",
        csv_bytes(sample_data()),
        "anthocyanin_synthetic_example.csv",
        "text/csv",
    )
    st.caption(
        "예시 CSV는 난수로 생성한 합성 데이터이며 실제 측정값이 아닙니다."
    )

    uploaded = st.file_uploader("측정 CSV 업로드", type=["csv"])
    use_sample = st.checkbox("합성 예시 데이터로 분석 체험")

    try:
        if uploaded is not None:
            data, quality = read_csv_data(uploaded.getvalue())
            st.caption(f"분석 대상: 업로드 파일 {uploaded.name}")
        elif use_sample:
            data, quality = prepare_data(sample_data())
            st.warning("현재 분석 대상은 합성 예시 데이터입니다.")

    except Exception as error:
        st.error(f"CSV 처리 오류: {error}")

    if data is None:
        st.info("CSV를 업로드하거나 합성 예시 데이터를 선택하세요.")

    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("원본 행", quality["원본 행"])
        c2.metric("유효 행", quality["유효 행"])
        c3.metric("제외 행", quality["제외 행"])
        c4.metric(
            "실험 수",
            data.groupby(GROUP_KEYS).ngroups,
        )

        with st.expander("데이터 품질 검사", expanded=True):
            st.json(quality)
            if quality["동일 실험·시간 중복 행"]:
                st.warning(
                    "같은 실험의 동일 시간에 여러 행이 있습니다. "
                    "별도 반복 실험이면 replicate 값을 분리하세요. "
                    "현재 적합에서는 각 행을 개별 관측값으로 사용합니다."
                )
            if quality["첫 시점 중복 실험"]:
                st.caption(
                    "동일 실험의 첫 시점에 여러 값이 있으면 "
                    "그 평균을 정규화 기준으로 사용합니다."
                )
            if quality["음수 시간 행"]:
                st.warning(
                    "음수 시간이 있습니다. 주입·혼합 시점 등 시간 기준을 확인하세요."
                )
            if quality["음수 흡광도 행"]:
                st.warning(
                    "음수 흡광도가 있습니다. 블랭크 보정을 확인하세요. "
                    "해당 실험의 감소 모형 적합은 건너뜁니다."
                )
            if quality["비양수 초기값 실험"]:
                st.warning(
                    "초기 흡광도가 0 이하인 실험은 A/A₀를 계산하지 않습니다."
                )

        st.dataframe(data, use_container_width=True, height=220)

        summaries, fits, details = analyze_data(data)

        # 숫자 ID로 표시해 업로드된 한글 그룹명의 폰트 문제 방지
        experiment_keys = list(details)
        experiment_ids = {
            key: f"E{index + 1:02d}"
            for index, key in enumerate(experiment_keys)
        }

        mapping_table = pd.DataFrame(
            [
                {
                    "그래프 ID": experiment_ids[key],
                    **dict(zip(GROUP_KEYS, key)),
                }
                for key in experiment_keys
            ]
        )

        with st.expander("그래프 ID와 실험 조건"):
            st.dataframe(mapping_table, use_container_width=True)

        for column, ylabel, title in [
            (
                "absorbance",
                "Absorbance, A (dimensionless)",
                "Measured absorbance",
            ),
            (
                "A_norm",
                "Relative absorbance, A / A0",
                "Normalized absorbance",
            ),
        ]:
            fig, ax = make_figure()
            for key, item in details.items():
                series = item["data"]
                ax.plot(
                    series["time"],
                    series[column],
                    marker=".",
                    markersize=4,
                    linewidth=1.3,
                    label=experiment_ids[key],
                )
            ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
            if len(details) <= 15:
                add_legend(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### 実験別の要約".replace("実験別の要約", "실험별 요약"))
        st.dataframe(
            summaries.round(6),
            use_container_width=True,
        )

        st.markdown("### 0차·1차·2차 전체 적합 결과")

        if fits.empty:
            st.info("적합 가능한 실험이 없습니다.")
        else:
            st.dataframe(fits.round(6), use_container_width=True)
            st.caption(
                "모든 모형은 원래 흡광도 공간에서 A₀와 k를 함께 추정합니다. "
                "A₀ > 0, k ≥ 0으로 제한하며 배경 흡광도는 포함하지 않습니다. "
                "A는 무차원 흡광도를 뜻합니다."
            )

            selected_key = st.selectbox(
                "적합 곡선·잔차를 볼 실험",
                experiment_keys,
                format_func=lambda key: (
                    f"{experiment_ids[key]} · pH {key[0]:g} · "
                    f"{key[1]} · 반복 {key[2]}"
                ),
            )

            item = details[selected_key]
            series = item["data"]

            if item["message"]:
                st.warning(item["message"])

            if item["curves"]:
                elapsed = (
                    series["time"].to_numpy()
                    - series["time"].min()
                )
                dense_time = np.linspace(0, elapsed.max(), 240)

                fig, ax = make_figure()
                ax.scatter(
                    elapsed,
                    series["absorbance"],
                    color="#e8eef7",
                    s=25,
                    label="Observed",
                    zorder=4,
                )

                labels = {
                    "0차 선형": "Zero-order",
                    "1차 지수형": "First-order",
                    "2차 역수형": "Second-order",
                }
                colors = {
                    "0차 선형": "#ffbf69",
                    "1차 지수형": "#63b3ff",
                    "2차 역수형": "#86d6a0",
                }

                for name, result in item["curves"].items():
                    ax.plot(
                        dense_time,
                        decay_curve(
                            name,
                            dense_time,
                            result["a0"],
                            result["k"],
                        ),
                        label=labels[name],
                        color=colors[name],
                    )

                ax.set(
                    title="Observed data and fitted decay models",
                    xlabel="Elapsed time from first observation (s)",
                    ylabel="Absorbance, A (dimensionless)",
                )
                add_legend(ax)
                fig.tight_layout()
                fit_png = figure_bytes(fig)
                st.pyplot(fig)
                plt.close(fig)

                st.download_button(
                    "선택 실험 적합 그래프 PNG",
                    fit_png,
                    "anthocyanin_fit.png",
                    "image/png",
                )

                fig, ax = make_figure(height=3)
                for name, result in item["curves"].items():
                    ax.scatter(
                        elapsed,
                        result["residual"],
                        label=labels[name],
                        color=colors[name],
                        s=22,
                        alpha=0.8,
                    )
                ax.axhline(0, color="#dce7f5", linestyle="--")
                ax.set(
                    title="Residuals: observed minus fitted",
                    xlabel="Elapsed time from first observation (s)",
                    ylabel="Residual absorbance",
                )
                add_legend(ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                condition = (
                    (fits["pH"] == selected_key[0])
                    & (fits["group"] == selected_key[1])
                    & (fits["replicate"] == selected_key[2])
                )
                selected_fits = fits.loc[condition]
                if not selected_fits.empty:
                    best = selected_fits.sort_values("RMSE").iloc[0]
                    st.info(
                        f"이 실험의 최저 RMSE 모형: {best['모형']} "
                        f"(RMSE {best['RMSE']:.5g}). "
                        "실제 반응 차수를 확정하는 결과는 아닙니다."
                    )

        st.download_button(
            "정리된 데이터 CSV",
            csv_bytes(data),
            "anthocyanin_processed.csv",
            "text/csv",
        )
        st.download_button(
            "실험별 요약 CSV",
            csv_bytes(summaries),
            "anthocyanin_summary.csv",
            "text/csv",
        )
        if not fits.empty:
            st.download_button(
                "전체 적합 결과 CSV",
                csv_bytes(fits),
                "anthocyanin_model_fits.csv",
                "text/csv",
            )

        st.warning(
            "A/A₀ 정규화는 pH별 몰흡광계수와 화학종 분포 차이를 "
            "제거하지 않습니다. 특정 파장의 흡광도 감소가 곧 실제 분해량은 "
            "아니며, 모형 적합만으로 반응 차수나 통계적 유의성을 "
            "확정할 수 없습니다."
        )


# =========================================================
# 탭 3: 조건별 비교
# =========================================================
with tabs[2]:
    st.subheader("반복 실험의 정규화 평균 ± 표준편차")

    if data is None:
        st.info("데이터 분석 탭에서 데이터를 먼저 선택하세요.")
    else:
        # 동일 replicate·시간 중복은 먼저 평균 처리
        per_replicate = (
            data.groupby(
                ["pH", "group", "replicate", "time"],
                as_index=False,
            )["A_norm"]
            .mean()
        )

        comparison = (
            per_replicate.groupby(["pH", "group", "time"])["A_norm"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"count": "n"})
        )

        fig, ax = make_figure()
        condition_mapping = []

        for index, ((p, group), series) in enumerate(
            comparison.groupby(["pH", "group"]), start=1
        ):
            label = f"C{index:02d}"
            condition_mapping.append(
                {"그래프 ID": label, "pH": p, "group": group}
            )

            x = series["time"].to_numpy(dtype=float)
            mean = series["mean"].to_numpy(dtype=float)
            sd = series["std"].to_numpy(dtype=float)
            n = series["n"].to_numpy()

            line, = ax.plot(x, mean, marker=".", label=label)

            valid_sd = n >= 2
            ax.fill_between(
                x,
                mean - sd,
                mean + sd,
                where=valid_sd,
                color=line.get_color(),
                alpha=0.18,
            )

        ax.axhline(1, color="#dce7f5", linestyle="--", alpha=0.5)
        ax.set(
            title="Normalized absorbance: mean ± SD",
            xlabel="Time (s)",
            ylabel="Relative absorbance, A / A0",
        )
        add_legend(ax)
        fig.tight_layout()
        comparison_png = figure_bytes(fig)
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            pd.DataFrame(condition_mapping),
            use_container_width=True,
        )
        st.dataframe(comparison.round(6), use_container_width=True)

        st.caption(
            "各반복".replace("各반복", "각 반복")
            + "을 첫 측정값 기준으로 정규화한 뒤 같은 시간끼리 평균합니다. "
            "음영은 반복 간 표준편차이며 신뢰구간이 아닙니다. "
            "반복이 1개인 시간에는 표준편차를 표시하지 않습니다. "
            "시간이 다르면 보간하지 않으므로 시간별 n이 달라질 수 있습니다."
        )

        st.download_button(
            "조건 비교 CSV",
            csv_bytes(comparison),
            "anthocyanin_condition_comparison.csv",
            "text/csv",
        )
        st.download_button(
            "조건 비교 그래프 PNG",
            comparison_png,
            "anthocyanin_comparison.png",
            "image/png",
        )


# =========================================================
# 탭 4: 화학 개념
# =========================================================
with tabs[3]:
    st.subheader("화학 개념과 해석상 주의점")

    concepts = [
        (
            "1. pH와 안토시아닌 화학종",
            "안토시아닌은 양성자 이동, 수화 및 구조 전환에 의해 "
            "플라빌륨 양이온, 퀴노이드 형태, 헤미케탈, 칼콘 등 "
            "여러 화학종으로 분포합니다. 색상은 종류와 용액 조건에 따라 "
            "달라지며 특정 pH에 하나의 색이 반드시 대응하지 않습니다.",
        ),
        (
            "2. 산성 조건",
            "충분히 산성인 조건에서는 플라빌륨 양이온이 상대적으로 "
            "안정화되어 붉은 계열의 색을 나타낼 수 있습니다. "
            "pH 4 이하의 모든 조건에서 같은 화학종이 지배적이라고 "
            "단정할 수는 없습니다.",
        ),
        (
            "3. 중간 pH 영역",
            "pH가 증가하면 여러 화학종의 비율이 변합니다. "
            "보라색·청색 변화뿐 아니라 수화에 따른 탈색도 가능하므로 "
            "단순한 빨강→파랑 전환으로만 이해하면 안 됩니다.",
        ),
        (
            "4. 염기성 조건과 안정성",
            "염기성에서는 화학종 분포와 구조 안정성이 달라집니다. "
            "강한 알칼리성 환경에서는 비가역적 분해도 가능하며, "
            "앱의 녹색 표현은 정성적 도식일 뿐입니다.",
        ),
        (
            "5. H₂O₂와 산화",
            "H₂O₂는 산화제로 작용할 수 있으며 안토시아닌의 "
            "공액 구조와 흡수 특성을 변화시킬 수 있습니다. "
            "구체적인 반응 경로와 생성물은 이 앱에서 확인하지 않습니다.",
        ),
        (
            "6. Beer–Lambert 법칙",
            "A = −log₁₀(I/I₀), A = εlc입니다. 흡광도는 무차원입니다. "
            "화학종, 파장, 광로 길이와 몰흡광계수가 일정하고 "
            "적절한 측정 범위 안에 있을 때 농도와의 비례를 기대할 수 있습니다.",
        ),
        (
            "7. 화학 평형",
            "평형은 정반응과 역반응 속도가 같아 거시적 조성이 "
            "일정한 동적 상태입니다. 산·염기 및 수화 평형과 "
            "산화에 의한 비가역적 변화는 구분해야 합니다.",
        ),
        (
            "8. 르샤틀리에 원리",
            "평형계에 조건 변화가 가해지면 그 영향을 줄이는 방향으로 "
            "새로운 평형 조성이 형성되는 경향이 있습니다. "
            "안토시아닌의 pH 의존성을 정성적으로 이해하는 데 활용합니다.",
        ),
        (
            "9. 산화·환원",
            "산화는 전자를 잃거나 산화수가 증가하는 과정이며 "
            "환원은 전자를 얻거나 산화수가 감소하는 과정입니다. "
            "흡광도 변화만으로 구체적 전자 이동 경로를 확정하지 않습니다.",
        ),
        (
            "10. 반응 속도와 유사 1차",
            "이 앱은 농도 대신 흡광도 변화로 겉보기 속도를 추정합니다. "
            "H₂O₂가 충분히 과량으로 존재해 농도가 사실상 일정한 경우 등에는 "
            "유사 1차 해석이 가능할 수 있으나 실험적 확인이 필요합니다. "
            "1차 감소 모형의 반감기는 ln(2)/k입니다.",
        ),
        (
            "11. 완충 용액과 공통 이온 효과",
            "완충 용액은 약산/짝염기 또는 약염기/짝산 계를 이용해 "
            "산·염기 첨가에 따른 pH 변화를 줄입니다. 완충능은 농도와 "
            "조성에 의존하므로 부피만으로 판단할 수 없습니다. "
            "공통 이온 효과는 관련 평형 현상이지만 완충 작용과 "
            "동일한 개념은 아닙니다.",
        ),
        (
            "12. 모형 적합의 한계",
            "높은 R²나 낮은 RMSE는 특정 반응 기구를 증명하지 않습니다. "
            "배경 흡광도가 크면 A∞를 포함한 모형이 필요할 수 있습니다. "
            "잔차, 관찰 시간, 반복 실험과 독립적인 화학적 근거를 함께 "
            "검토해야 합니다.",
        ),
    ]

    for title, body in concepts:
        with st.expander(title):
            st.write(body)


# =========================================================
# 탭 5: 조건 저장 및 보고서
# =========================================================
with tabs[4]:
    st.subheader("조건 저장·불러오기")

    settings = current_settings()
    st.json(settings)

    st.download_button(
        "실험 조건 JSON 저장",
        json.dumps(
            settings,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "anthocyanin_conditions.json",
        "application/json",
    )

    settings_file = st.file_uploader(
        "조건 JSON 불러오기",
        type=["json"],
        key="settings_upload",
    )

    if settings_file is not None:
        st.button(
            "불러온 조건을 사이드바에 적용",
            on_click=apply_settings,
            args=(settings_file.getvalue(),),
        )

    if "settings_notice" in st.session_state:
        st.success(st.session_state.pop("settings_notice"))

    if "settings_error" in st.session_state:
        st.error(st.session_state.pop("settings_error"))

    st.markdown("---")
    st.subheader("요약 보고서")

    lines = [
        "Anthocyanin Lab V4",
        f"생성 시각: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "[사이드바 시뮬레이션 조건]",
        json.dumps(settings, ensure_ascii=False, indent=2),
        "",
        "주의: 사이드바 조건은 업로드 CSV의 실험 조건을 대체하지 않습니다.",
        "시뮬레이션은 검증된 실제 반응 예측 모델이 아닙니다.",
        "흡광도 기반 속도는 겉보기 탈색 속도로 해석해야 합니다.",
    ]

    if data is not None:
        source = (
            f"업로드 CSV: {uploaded.name}"
            if uploaded is not None
            else "합성 예시 데이터 — 실제 측정값 아님"
        )
        lines.extend(
            [
                "",
                "[분석 데이터 출처]",
                source,
                "",
                "[데이터 품질]",
                json.dumps(quality, ensure_ascii=False, indent=2),
                "",
                "[실험별 요약]",
                summaries.to_csv(index=False),
            ]
        )
        if not fits.empty:
            lines.extend(
                [
                    "",
                    "[모형 적합 결과]",
                    fits.to_csv(index=False),
                ]
            )

    lines.extend(
        [
            "",
            "[해석상 주의]",
            "정규화는 pH별 화학종·몰흡광계수 차이를 제거하지 않습니다.",
            "최저 RMSE 모형이 실제 반응 차수를 증명하지는 않습니다.",
            "표준편차는 신뢰구간이나 통계적 유의성을 의미하지 않습니다.",
            "실제 실험에서는 측정 파장, 광로 길이, 블랭크, "
            "혼합 후 pH와 H₂O₂ 농도를 별도로 기록하세요.",
        ]
    )

    report = "\n".join(lines)
    st.download_button(
        "TXT 요약 보고서 다운로드",
        report.encode("utf-8-sig"),
        "anthocyanin_report.txt",
        "text/plain",
    )
    st.caption(
        "PDF 파일을 직접 생성하지 않습니다. "
        "필요하면 TXT를 문서 편집기에서 열어 PDF로 저장하세요."
    )


st.markdown("---")
st.caption(
    "Anthocyanin Lab V4 · 교육용 분석 도구 · "
    "가정 시뮬레이션과 실제 측정 결과를 구분하세요."
)
