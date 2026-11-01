import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="안토시아닌 탈색 시뮬레이터", page_icon="🧪", layout="wide")

st.title("안토시아닌 탈색 시뮬레이터")
st.caption("pH · 과산화수소 · 초기 흡광도 · 겉보기 속도상수에 따른 시간-흡광도 변화를 탐색합니다.")

st.info(
    "주의: 시뮬레이션은 사용자가 지정한 수학적 모형을 시각화합니다. "
    "pH가 실제 반응속도에 미치는 영향이나 H₂O₂ 농도와 속도상수의 관계를 "
    "실험적으로 검증해 주지는 않습니다. 따라서 아래 속도상수는 '겉보기 속도상수'이며, "
    "실험에서 측정한 값과 구분해야 합니다."
)

tab_sim, tab_data, tab_theory = st.tabs(["① 시뮬레이션", "② 실제 데이터 분석", "③ 수학적 모형"])

def interpolate_k(ph, k4, k7, k10):
    """Illustrative piecewise-linear interpolation between user-defined pH anchors."""
    if ph <= 7:
        return k4 + (k7 - k4) * (ph - 4) / 3
    return k7 + (k10 - k7) * (ph - 7) / 3

def model_curve(t, A0, k, order):
    t = np.asarray(t, dtype=float)
    if order == "0차":
        return np.maximum(A0 - k * t, 0)
    if order == "1차":
        return A0 * np.exp(-k * t)
    # Absorbance-proxy second-order form; k units depend on absorbance and time.
    return A0 / (1 + k * A0 * t)

with tab_sim:
    st.subheader("변인 설정")
    left, right = st.columns([1, 1.5])

    with left:
        ph = st.slider("용액의 pH", min_value=4.0, max_value=10.0, value=7.0, step=0.1)
        A0 = st.slider("초기 흡광도 A₀", min_value=0.05, max_value=2.0, value=0.80, step=0.05)
        duration = st.slider("관찰 시간", min_value=1, max_value=180, value=60, step=1)
        n_points = st.slider("그래프 측정 지점 수", min_value=10, max_value=300, value=61, step=1)
        order = st.selectbox("적용할 반응 모형", ["1차", "0차", "2차"], index=0)

    with right:
        st.markdown("**pH별 겉보기 속도상수 설정**")
        st.caption("세 기준 pH의 값을 직접 조절합니다. 중간 pH는 직선 보간합니다. 실제 실험값이 있다면 그 값으로 설정하세요.")
        c1, c2, c3 = st.columns(3)
        with c1:
            k4 = st.number_input("pH 4의 k", min_value=0.00001, max_value=2.0, value=0.025, format="%.5f")
        with c2:
            k7 = st.number_input("pH 7의 k", min_value=0.00001, max_value=2.0, value=0.035, format="%.5f")
        with c3:
            k10 = st.number_input("pH 10의 k", min_value=0.00001, max_value=2.0, value=0.050, format="%.5f")

        h2o2_on = st.toggle("H₂O₂ 처리군 표시", value=True)
        h2o2_level = st.slider("H₂O₂ 농도 (상대 설정값, %)", 0.0, 5.0, 1.0, 0.1)
        exponent = st.slider("농도-속도상수 관계의 지수(가정값)", 0.0, 2.0, 1.0, 0.1)
        k_control = st.number_input("대조군의 겉보기 속도상수 k", min_value=0.00001, max_value=2.0, value=0.003, format="%.5f")

    k_base = interpolate_k(ph, k4, k7, k10)
    # The concentration scaling is an explicit user-selected assumption, not a fitted law.
    h2o2_factor = (h2o2_level / 1.0) ** exponent if h2o2_level > 0 else 0.0
    k_treatment = k_base * h2o2_factor
    times = np.linspace(0, duration, n_points)
    control = model_curve(times, A0, k_control, order)
    treatment = model_curve(times, A0, k_treatment, order)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=control, mode="lines+markers", name="H₂O 대조군"))
    if h2o2_on:
        fig.add_trace(go.Scatter(x=times, y=treatment, mode="lines+markers", name="H₂O₂ 처리군"))
    fig.update_layout(
        title=f"pH {ph:.1f}에서 시간에 따른 흡광도 변화",
        xaxis_title="시간 (앱에서 설정한 시간 단위)",
        yaxis_title="흡광도 (A)",
        hovermode="x unified",
        legend_title="조건",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("pH 보간 k", f"{k_base:.5f}")
    m2.metric("H₂O₂ 처리군 k", f"{k_treatment:.5f}" if h2o2_on else "표시 안 함")
    m3.metric("대조군 k", f"{k_control:.5f}")
    if order == "1차":
        half = np.log(2) / k_treatment if k_treatment > 0 else np.nan
        m4.metric("처리군 반감기", f"{half:.2f}" if h2o2_on else "—")
    else:
        m4.metric("처리군 반감기", "모형에 따라 달라짐")

    st.markdown("**현재 설정에서의 정규화 흡광도**")
    norm_fig = go.Figure()
    norm_fig.add_trace(go.Scatter(x=times, y=control / A0, mode="lines", name="대조군 A/A₀"))
    if h2o2_on:
        norm_fig.add_trace(go.Scatter(x=times, y=treatment / A0, mode="lines", name="처리군 A/A₀"))
    norm_fig.update_layout(
        xaxis_title="시간 (앱에서 설정한 시간 단위)",
        yaxis_title="정규화 흡광도 A/A₀",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(norm_fig, use_container_width=True)

    st.download_button(
        "현재 시뮬레이션 데이터 CSV 다운로드",
        data=pd.DataFrame({
            "time": times,
            "control_absorbance": control,
            "treatment_absorbance": treatment
        }).to_csv(index=False).encode("utf-8-sig"),
        file_name="anthocyanin_simulation.csv",
        mime="text/csv",
    )

    with st.expander("모형에 적용된 계산 가정"):
        st.write(f"- 선택한 모형: {order}")
        st.write(f"- pH별 k는 pH 4, 7, 10의 사용자 설정값 사이를 구간별 직선 보간했습니다.")
        st.write(f"- H₂O₂ 처리군 k = pH 보간 k × (H₂O₂ 설정 농도 / 1.0)^{exponent:.1f}")
        st.write("- 위 농도식은 사용자가 관계를 탐색하기 위한 가정이며, 실험으로 검증된 속도법칙이 아닙니다.")
        st.write("- 대조군과 처리군은 같은 초기 흡광도에서 출발하도록 설정했습니다.")
        st.write("- 실제 안토시아닌은 pH에 따라 화학종과 흡광 특성이 달라질 수 있으므로, 이 단순 모형만으로 분자 농도나 반응기작을 확정할 수 없습니다.")

with tab_data:
    st.subheader("실험 데이터 업로드")
    st.write("CSV 파일은 최소한 `time`, `pH`, `group`, `absorbance` 열을 포함해야 합니다.")
    st.code("time,pH,group,absorbance\n0,4,H2O,0.82\n1,4,H2O,0.81\n0,4,H2O2,0.80\n1,4,H2O2,0.74", language="csv")
    uploaded = st.file_uploader("실험 데이터 CSV 업로드", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            required = {"time", "pH", "group", "absorbance"}
            missing = required - set(df.columns)
            if missing:
                st.error("필수 열이 없습니다: " + ", ".join(sorted(missing)))
            else:
                for col in ["time", "pH", "absorbance"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["group"] = df["group"].astype(str)
                df = df.dropna(subset=["time", "pH", "absorbance"])
                ph_options = sorted(df["pH"].unique())
                selected_ph = st.selectbox("표시할 pH 선택", ph_options)
                view = df[df["pH"] == selected_ph].sort_values("time")
                data_fig = px.line(
                    view, x="time", y="absorbance", color="group",
                    markers=True, title=f"실험 데이터: pH {selected_ph}"
                )
                data_fig.update_layout(
                    xaxis_title="시간 (CSV의 time 단위)",
                    yaxis_title="흡광도 (A)",
                    template="plotly_white",
                )
                st.plotly_chart(data_fig, use_container_width=True)

                norm_df = view.copy()
                norm_df["A0"] = norm_df.groupby("group")["absorbance"].transform("first")
                norm_df = norm_df[norm_df["A0"] != 0].copy()
                norm_df["normalized_absorbance"] = norm_df["absorbance"] / norm_df["A0"]
                norm_fig = px.line(
                    norm_df, x="time", y="normalized_absorbance",
                    color="group", markers=True,
                    title=f"정규화 실험 데이터: pH {selected_ph}"
                )
                norm_fig.update_layout(
                    xaxis_title="시간 (CSV의 time 단위)",
                    yaxis_title="정규화 흡광도 A/A₀",
                    template="plotly_white",
                )
                st.plotly_chart(norm_fig, use_container_width=True)
                st.dataframe(df, use_container_width=True)
                st.caption("정규화는 각 그룹에서 시간순 첫 측정값을 A₀로 사용합니다. 실제 t=0 측정값이 첫 행인지 확인하세요.")
        except Exception as e:
            st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")

with tab_theory:
    st.subheader("수학적 모형")
    st.markdown("**Beer–Lambert 법칙**")
    st.latex(r"A = \varepsilon l c")
    st.write("일정한 파장·화학종·광로 길이에서 흡광도 A는 농도 c에 비례합니다. pH가 달라지면 안토시아닌 화학종과 몰흡광계수 ε가 달라질 수 있어 주의가 필요합니다.")
    st.markdown("**0차 모형**")
    st.latex(r"A(t) = A_0 - kt")
    st.markdown("**1차 모형**")
    st.latex(r"A(t) = A_0 e^{-kt}")
    st.latex(r"\ln(A(t)/A_0) = -kt")
    st.markdown("**2차 모형(흡광도를 농도 대용값으로 놓은 단순 형태)**")
    st.latex(r"\frac{1}{A(t)} = kt + \frac{1}{A_0}")
    st.warning("실제 반응 차수는 이론만으로 정하지 않습니다. 실험 데이터에 여러 모형을 적합하고 잔차와 적합도를 확인해야 합니다. pH에 따른 흡광 스펙트럼 변화가 크면 A를 농도와 동일시하는 가정이 성립하지 않을 수 있습니다.")

st.divider()
st.caption("탐구용 교육 앱 · 결과 해석 시 실측값과 가정값을 구분하세요.")
