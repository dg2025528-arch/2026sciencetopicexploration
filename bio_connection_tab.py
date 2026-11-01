
import streamlit as st


def _render_plant_cell_simulator() -> None:
    """Show a simple, qualitative plant oxidative-stress scenario simulator."""
    st.markdown("---")
    st.subheader("간단한 식물세포 시뮬레이터")
    st.caption(
        "환경 상태를 선택하면 식물세포의 ROS와 안토시아닌에 관한 설명이 달라집니다. "
        "실제 세포 수치를 계산하는 모델이 아니라 문헌에 근거한 개념 시뮬레이션입니다."
    )

    scenario = st.radio(
        "식물의 환경 상태를 선택해 보세요.",
        ["평상시", "환경 스트레스", "심한·지속적인 스트레스"],
        horizontal=True,
        key="bio_plant_scenario",
    )

    if scenario == "평상시":
        ros_label = "기본적인 생성·신호 기능"
        defense_label = "생성된 ROS를 조절"
        anthocyanin_label = "색소로 저장되며 여러 생리적 역할에 관여"
        ros_text = (
            "식물세포에서도 ROS는 정상적인 대사 과정에서 만들어집니다. "
            "과산화수소(H₂O₂)는 세포 신호 전달에도 관여할 수 있습니다."
        )
        defense_text = (
            "식물은 카탈레이스, 퍼옥시다아제, 아스코르브산 등 여러 효소·물질을 이용해 "
            "ROS 수준을 조절합니다. 목표는 ROS를 모두 없애는 것이 아니라 적절한 균형을 유지하는 것입니다."
        )
        anthocyanin_text = (
            "안토시아닌은 세포질에서 합성된 뒤 주로 액포에 저장됩니다. "
            "식물의 종류와 조직에 따라 색소의 양과 역할은 달라질 수 있습니다."
        )
        takeaway = "ROS는 항상 해로운 물질이 아니며, 정상적인 세포 신호에도 필요합니다."

    elif scenario == "환경 스트레스":
        ros_label = "증가할 수 있음"
        defense_label = "항산화·스트레스 반응 활성화"
        anthocyanin_label = "일부 식물에서 합성·축적이 증가할 수 있음"
        ros_text = (
            "강한 빛, 고온, 건조 등의 환경 스트레스는 식물의 ROS 생성과 제거 균형을 흔들 수 있으며, "
            "조건에 따라 H₂O₂가 신호 분자로 작용할 수 있습니다."
        )
        defense_text = (
            "식물은 항산화 효소와 비효소성 항산화 물질의 작용을 조절해 산화·환원 균형을 유지하려고 합니다. "
            "이 반응은 스트레스의 종류와 강도, 식물 종에 따라 달라집니다."
        )
        anthocyanin_text = (
            "일부 식물에서는 환경 스트레스에 반응해 안토시아닌 생합성 관련 유전자 발현과 색소 축적이 증가할 수 있습니다. "
            "안토시아닌은 광보호와 항산화 방어에 기여할 수 있지만, 모든 식물에서 같은 반응이 나타나는 것은 아닙니다."
        )
        takeaway = "환경 스트레스는 ROS 신호와 방어 반응을 유도할 수 있고, 안토시아닌도 그 반응에 관여할 수 있습니다."

    else:
        ros_label = "방어 능력을 초과할 가능성"
        defense_label = "방어 체계만으로 균형 유지가 어려울 수 있음"
        anthocyanin_label = "보호에 기여할 수 있지만 산화·분해될 수도 있음"
        ros_text = (
            "스트레스가 매우 강하거나 오래 지속되면 ROS 생성이 제거 능력을 넘어설 수 있습니다. "
            "이때 산화 스트레스가 커지고 세포 구성 물질이 손상될 위험이 높아집니다."
        )
        defense_text = (
            "항산화 방어 체계가 작동하더라도 스트레스가 지나치면 산화·환원 균형을 회복하지 못할 수 있습니다. "
            "실제 영향은 식물의 종류와 조직, 노출 시간 등에 따라 달라집니다."
        )
        anthocyanin_text = (
            "안토시아닌은 과도한 ROS에 대한 방어에 기여할 가능성이 있지만, 그 자체가 산화되거나 분해될 수도 있습니다. "
            "일부 식물의 액포에서는 H₂O₂와 class III 과산화효소가 안토시아닌 분해에 관여할 가능성이 제시되어 있습니다."
        )
        takeaway = "안토시아닌은 무조건 산화 스트레스를 막는 물질이 아니며, 보호 역할과 산화적 분해 가능성을 함께 봐야 합니다."

    st.markdown("#### 선택한 환경에서 예상할 수 있는 변화")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**① ROS 상태**\n\n{ros_label}\n\n{ros_text}")
    with col2:
        st.info(f"**② 항산화 방어**\n\n{defense_label}\n\n{defense_text}")
    with col3:
        st.info(f"**③ 안토시아닌**\n\n{anthocyanin_label}\n\n{anthocyanin_text}")

    st.success(f"**핵심 해석:** {takeaway}")

    with st.expander("내가 한 안토시아닌 실험과 연결하기"):
        st.write(
            "네 실험에서는 적양배추 추출물에 H₂O₂를 직접 넣고 pH 조건별 흡광도 변화를 관찰했습니다. "
            "따라서 시험관에서의 산화적 탈색 현상을 식물세포에서 ROS와 안토시아닌이 관련되는 생리 현상을 생각하는 출발점으로 활용할 수 있습니다."
        )
        st.warning(
            "단, 시험관 속 추출물에는 실제 세포의 액포막, 효소, 다른 항산화 물질과 세포 내 구획이 그대로 존재하지 않습니다. "
            "그러므로 실험에서 얻은 탈색 속도를 식물세포 안의 ROS 농도나 안토시아닌 분해 속도로 바꾸어 해석할 수는 없습니다."
        )


def render_bio_connection_tab() -> None:
    """Render a concise biology explanation plus an interactive concept simulator."""
    st.subheader("생명과학으로 연결하기")
    st.caption(
        "기존 안토시아닌 실험을 식물세포의 산화 스트레스와 연결해 보는 간단한 탐구 공간입니다."
    )

    st.markdown("#### 식물세포 안에서 생각해 보기")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**① 환경 변화**\n\n"
            "강한 빛이나 고온, 건조 등의 환경 스트레스는 식물세포의 산화·환원 균형에 영향을 줄 수 있습니다."
        )
    with c2:
        st.info(
            "**② ROS**\n\n"
            "과산화수소(H₂O₂)는 식물세포에서 생성되는 ROS 중 하나이며, 세포 신호 전달에도 관여합니다."
        )
    with c3:
        st.info(
            "**③ 액포의 안토시아닌**\n\n"
            "안토시아닌은 세포질에서 합성된 뒤 주로 액포에 저장됩니다. 특정 조건에서는 H₂O₂와 과산화효소가 색소 분해에 관여할 수 있습니다."
        )

    _render_plant_cell_simulator()

    st.markdown("---")
    st.markdown("#### 기존 실험과 실제 식물세포의 차이")
    st.write(
        "이번 실험은 추출한 안토시아닌 용액에 H₂O₂를 직접 넣어 흡광도 변화를 관찰했습니다. "
        "실제 식물세포에서는 안토시아닌이 액포에 저장되고, 세포 내 구획·과산화효소·다른 항산화 물질 등이 함께 작용합니다. "
        "따라서 시험관에서 측정한 탈색 속도를 식물세포 안의 반응 속도로 그대로 해석할 수는 없습니다."
    )
    st.caption(
        "주의: 이 기능은 문헌을 바탕으로 한 정성적 개념 시뮬레이션입니다. ROS 농도, 세포 생존율 또는 생체 내 반응 속도를 계산하지 않습니다. "
        "실험에서 사용한 pH 4·7·10도 식물 액포의 실제 pH를 그대로 재현한 값이 아닙니다."
    )

    st.markdown("#### 참고 문헌")
    st.markdown(
        "- Mittler et al. (2022), [Reactive oxygen species signalling in plant stress responses](https://www.nature.com/articles/s41580-022-00499-2)\n"
        "- Zhao et al. (2021), [Anthocyanin stability and degradation in plants](https://pubmed.ncbi.nlm.nih.gov/34686106/)\n"
        "- Sharma et al. (2021), [Abiotic stress-induced anthocyanins in plants: Their role in tolerance to abiotic stresses](https://pubmed.ncbi.nlm.nih.gov/33605458/)"
    )
