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

tab_sim, tab_color, tab_data, tab_theory = st.tabs(["① 흡광도 시뮬레이션", "② 색 변화 시각화", "③ 실제 데이터 분석", "④ 화학·수학 개념"])

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

with tab_color:
    st.subheader("안토시아닌 용액의 색 변화")
    st.markdown(
        "이 페이지에서는 **완충용액과 안토시아닌 추출액을 섞은 직후의 색**과 "
        "**H₂O₂ 반응이 진행된 뒤의 색**을 개념적으로 비교합니다. "
        "왼쪽의 pH는 혼합이 끝난 용액의 pH를 뜻합니다."
    )

    color_left, color_right = st.columns([1, 1.5])
    with color_left:
        color_ph = st.slider("혼합 용액의 pH", 4.0, 10.0, float(ph), 0.1, key="color_ph")
        anthocyanin_ml = st.number_input("안토시아닌 추출액 부피 (mL)", 0.0, 50.0, 5.0, 0.5, key="anthocyanin_ml")
        buffer_ml = st.number_input("완충용액 부피 (mL)", 0.0, 200.0, 45.0, 0.5, key="buffer_ml")
        color_time = st.slider("H₂O₂ 반응 경과 시간", 0, int(duration), min(10, int(duration)), 1, key="color_time")
        color_h2o2 = st.toggle("H₂O₂를 첨가한 상황으로 보기", value=True, key="color_h2o2")

    total_ml = anthocyanin_ml + buffer_ml
    if total_ml <= 0:
        st.error("두 용액의 부피 합이 0보다 커야 색을 시각화할 수 있습니다.")
    else:
        extract_fraction = anthocyanin_ml / total_ml
        ratio_text = f"{anthocyanin_ml:g} : {buffer_ml:g}"
        with color_right:
            st.markdown("**현재 혼합 조건**")
            q1, q2, q3 = st.columns(3)
            q1.metric("총 부피", f"{total_ml:g} mL")
            q2.metric("추출액 비율", f"{extract_fraction*100:.1f}%")
            q3.metric("부피비 (추출액:완충액)", ratio_text)
            st.progress(min(1.0, extract_fraction / 0.10))
            st.caption("막대는 기준 혼합비(추출액 10%)에 대한 상대적인 추출액 비율을 나타내는 시각화 지표입니다.")

        # Illustrative hue anchors for a red-cabbage anthocyanin indicator.
        # This is not a calibrated RGB-to-pH or concentration model.
        anchors = [(4.0, np.array([205, 55, 91])), (7.0, np.array([111, 67, 151])), (10.0, np.array([48, 151, 145]))]
        if color_ph <= 7.0:
            lo, hi = anchors[0], anchors[1]
        else:
            lo, hi = anchors[1], anchors[2]
        w = (color_ph - lo[0]) / (hi[0] - lo[0])
        base_rgb = lo[1] * (1 - w) + hi[1] * w
        # Dilution and bleaching are represented as fading toward a white background.
        concentration_visual_factor = min(1.0, extract_fraction / 0.10)
        selected_curve = model_curve(np.array([color_time], dtype=float), A0, k_treatment, order)[0]
        remaining_color = max(0.0, min(1.0, selected_curve / A0)) if color_h2o2 else 1.0
        initial_strength = 0.90 * concentration_visual_factor
        final_strength = initial_strength * remaining_color

        def mix_with_white(rgb, strength):
            out = 255 - (255 - rgb) * strength
            return "#" + "".join(f"{int(round(v)):02X}" for v in out)

        initial_hex = mix_with_white(base_rgb, initial_strength)
        final_hex = mix_with_white(base_rgb, final_strength)
        st.markdown("### 색상 비교")
        sw1, sw2 = st.columns(2)
        with sw1:
            st.markdown(
                f'<div style="background:{initial_hex};border:1px solid #999;border-radius:12px;height:150px;display:flex;align-items:center;justify-content:center;color:#222;font-weight:600">반응 전</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**초기 색상** · `{initial_hex}`")
        with sw2:
            st.markdown(
                f'<div style="background:{final_hex};border:1px solid #999;border-radius:12px;height:150px;display:flex;align-items:center;justify-content:center;color:#222;font-weight:600">반응 후</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{color_time}시간 후 색상(모형)** · `{final_hex}`")

        st.metric("모형상 남아 있는 상대 흡광도", f"{remaining_color*100:.1f}%")
        st.markdown("**색이 변하는 이유**")
        st.write(
            "안토시아닌은 pH에 따라 분자 구조와 존재하는 화학종의 비율이 달라집니다. "
            "따라서 빛을 흡수하는 파장과 색이 달라질 수 있습니다. H₂O₂가 존재하면 "
            "안토시아닌의 산화·분해가 진행되어 색을 나타내는 물질의 양이 줄어들 수 있고, "
            "그 결과 색이 옅어지는 방향으로 표현했습니다."
        )
        st.warning(
            "색상은 원리를 설명하기 위한 개념적 근사입니다. RGB 값은 실제 분광광도계 측정값이나 "
            "안토시아닌의 정확한 농도에서 계산한 값이 아닙니다. 추출액 비율에 따른 색의 진하기도 "
            "단순화한 표현입니다. 특히 완충용액의 양은 완충능과 실제 혼합 pH에 영향을 줄 수 있으므로, "
            "pH를 별도로 측정하지 않았다면 설정한 pH가 실제 혼합액의 pH와 같다고 단정할 수 없습니다."
        )

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
    st.subheader("화학·수학 개념 이해하기")
    st.markdown("이 설명은 2022 개정 교육과정의 화학 학습 내용과 화학 반응의 세계를 이수한 학생을 기준으로 구성했습니다.")

    with st.expander("1. 안토시아닌과 pH: 왜 색이 달라질까?", expanded=True):
        st.write(
            "안토시아닌은 식물에 존재하는 색소 물질입니다. 용액의 pH가 달라지면 수소 이온(H⁺)과 관련된 "
            "산·염기 평형 및 분자 구조 변화로 인해 여러 화학종의 상대적인 비율이 달라질 수 있습니다. "
            "각 화학종은 빛을 흡수하는 특성이 다르므로 관찰되는 색도 달라집니다."
        )
        st.markdown("**핵심 연결:** pH 변화 → 화학종 분포 변화 → 빛 흡수 특성 변화 → 관찰되는 색 변화")
        st.info("색 변화는 단순히 용액에 색소가 더 많아지거나 적어지는 현상만은 아닙니다. 같은 양의 색소라도 화학종이 달라지면 색이 달라질 수 있습니다.")

    with st.expander("2. 완충용액: pH를 왜 유지할까?"):
        st.write(
            "완충용액은 약산과 그 짝염기(또는 약염기와 그 짝산)가 함께 존재하여, 소량의 산이나 염기가 들어왔을 때 "
            "pH 변화가 상대적으로 작도록 하는 용액입니다. 이 실험에서는 반응이 진행되는 동안 pH 조건을 비교적 일정하게 유지하기 위해 사용합니다."
        )
        st.markdown("**주의:** 완충용액의 부피를 바꾸면 완충능과 혼합 후 pH가 달라질 수 있습니다. 추출액의 부피비만 바꿨다고 해서 pH가 자동으로 일정하게 유지되는 것은 아닙니다.")

    with st.expander("3. 산화·환원과 H₂O₂: 왜 탈색이 일어날까?"):
        st.write(
            "과산화수소(H₂O₂)는 반응 조건에 따라 산화제로 작용할 수 있습니다. 안토시아닌의 발색 구조가 산화 등의 반응으로 변형되면 "
            "기존과 같은 빛 흡수 특성을 유지하지 못해 색이 옅어질 수 있습니다. 다만 이 앱의 단순 시뮬레이션만으로 "
            "실제 생성물이나 세부 반응 경로를 확정할 수는 없습니다."
        )
        st.markdown("**핵심 연결:** 산화 반응 → 발색 구조의 변화 또는 분해 → 특정 파장에서의 흡광도 감소 가능")

    with st.expander("4. 흡광도와 Beer–Lambert 법칙"):
        st.write("분광광도계는 시료에 빛을 통과시킨 뒤, 들어간 빛과 통과한 빛의 세기를 비교합니다.")
        st.latex(r"A = -\log_{10}(T) = \varepsilon l c")
        st.write("A는 흡광도, T는 투과도, ε는 몰흡광계수, l은 빛이 시료를 통과하는 길이, c는 흡광 물질의 농도입니다.")
        st.write("같은 화학종·파장·광로 길이에서는 농도가 커질수록 흡광도가 커집니다. 하지만 pH가 바뀌면 안토시아닌 화학종과 ε가 달라질 수 있으므로 서로 다른 pH의 흡광도를 곧바로 농도 차이로 해석하면 안 됩니다.")

    with st.expander("5. 반응속도와 시간에 따른 흡광도 변화"):
        st.write("흡광도가 시간에 따라 감소한다면, 일정한 조건에서 그 감소 양상을 반응 진행의 간접 지표로 활용할 수 있습니다.")
        st.markdown("**평균 흡광도 감소 속도**")
        st.latex(r"v_A = -\frac{A_t-A_0}{t-t_0}")
        st.markdown("**1차 반응 형태를 가정한 모형**")
        st.latex(r"A_t = A_0 e^{-kt}")
        st.latex(r"\ln\left(\frac{A_t}{A_0}\right)=-kt")
        st.write("ln(A/A₀)를 시간에 대해 그래프로 나타냈을 때 직선에 가까우면 1차 모형이 자료를 설명하는 후보가 됩니다. 기울기는 -k에 해당합니다. 다만 흡광도를 농도처럼 취급할 수 있다는 조건이 필요하며, 실제 반응 차수는 실험 데이터로 판단해야 합니다.")

    with st.expander("6. 이 앱에서 조절하는 변인과 해석 한계"):
        st.markdown(
            "- **pH:** 안토시아닌 화학종 분포와 흡광 특성에 영향을 줄 수 있는 조건\n"
            "- **안토시아닌 추출액 부피:** 혼합액에 들어가는 색소 추출액의 양\n"
            "- **완충용액 부피:** 혼합액의 희석 정도와 완충 환경에 영향을 주는 양\n"
            "- **H₂O₂ 처리 여부·농도 설정:** 산화 반응 조건을 단순화해 표현하는 입력값\n"
            "- **시간:** 반응 진행에 따른 흡광도 변화를 관찰하는 독립변인"
        )
        st.warning("앱의 pH별 속도상수, H₂O₂ 농도와 속도상수의 관계, RGB 색상은 탐색을 위한 가정값입니다. 실험으로 측정·보정하지 않은 값은 실제 화학계의 정량적 예측값으로 사용하지 마세요.")

    st.markdown("### 수학적 반응 모형")
    st.markdown("**0차 모형**")
    st.latex(r"A(t) = A_0 - kt")
    st.markdown("**1차 모형**")
    st.latex(r"A(t) = A_0 e^{-kt}")
    st.markdown("**2차 모형(흡광도를 농도 대용값으로 놓은 단순 형태)**")
    st.latex(r"\frac{1}{A(t)} = kt + \frac{1}{A_0}")
    st.warning("실제 반응 차수는 이론만으로 정하지 않습니다. 여러 모형을 실험 데이터에 적용하고 잔차와 적합도를 비교해야 합니다.")

st.divider()
st.caption("탐구용 교육 앱 · 결과 해석 시 실측값과 가정값을 구분하세요.")
