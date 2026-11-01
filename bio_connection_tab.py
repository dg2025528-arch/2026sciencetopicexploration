"""Streamlit add-on: a small biology-connection tab for an anthocyanin experiment app.

Usage in the existing app.py:
    from bio_connection_tab import render_bio_connection_tab

    # Add one more tab to the existing st.tabs([...]) list, named '생명과학 연결'.
    # Then, in the new tab:
    with tab_bio:
        render_bio_connection_tab()
"""

import streamlit as st


def render_bio_connection_tab() -> None:
    """Render a concise, qualitative explanation linking the experiment to plant biology."""

    st.subheader("생명과학으로 연결하기")
    st.caption("기존 안토시아닌 실험을 식물세포의 산화 스트레스와 연결해 보는 간단한 설명입니다.")

    situation = st.radio(
        "식물의 상태를 선택해 보세요.",
        ["평상시", "강한 빛·고온 등 환경 스트레스"],
        horizontal=True,
        key="bio_connection_situation",
    )

    st.markdown("#### 식물세포 안에서 생각해 보기")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**① 환경 변화**\n\n강한 빛이나 고온 등은 식물세포의 산화·환원 균형에 영향을 줄 수 있습니다.")
    with c2:
        st.info("**② H₂O₂ 증가 가능성**\n\n과산화수소(H₂O₂)는 식물세포에서 만들어지는 ROS 중 하나이며, 신호 전달에도 관여합니다.")
    with c3:
        st.info("**③ 액포의 안토시아닌**\n\n안토시아닌은 주로 액포에 저장됩니다. 특정 조건에서는 과산화효소와 H₂O₂가 색소의 산화적 분해에 관여할 수 있습니다.")

    if situation == "평상시":
        st.success(
            "평상시에도 ROS는 만들어집니다. ROS는 무조건 해로운 물질이 아니라 세포 신호 전달에도 쓰이며, "
            "식물은 ROS 생성과 제거 사이의 균형을 조절합니다."
        )
        st.write("**핵심:** 안토시아닌을 포함한 항산화 물질은 과도한 산화 작용을 조절하는 데 기여할 가능성이 있지만, ROS의 정상적인 신호 기능도 유지되어야 합니다.")
    else:
        st.warning(
            "환경 스트레스가 커지면 ROS, 특히 H₂O₂의 양이 증가할 수 있습니다. 일부 식물에서는 액포의 "
            "class III 과산화효소(peroxidase)가 H₂O₂를 이용해 안토시아닌 등 페놀성 물질의 산화를 촉진하고, "
            "색소 분해에 관여할 수 있다는 연구가 보고되어 있습니다."
        )
        st.write("**핵심:** 네 실험에서 관찰한 H₂O₂에 의한 탈색을 식물세포의 색소 분해 과정과 연결해 생각해 볼 수 있습니다.")

    st.markdown("---")
    st.markdown("#### 기존 실험과 실제 식물세포의 차이")
    st.write(
        "이번 실험은 추출한 안토시아닌 용액에 H₂O₂를 직접 넣어 흡광도 변화를 관찰했습니다. "
        "실제 식물세포에서는 안토시아닌이 액포에 저장되고, 세포 내 구획·과산화효소·다른 항산화 물질 등이 함께 작용합니다. "
        "따라서 시험관에서 측정한 탈색 속도를 식물세포 안의 반응 속도로 그대로 해석할 수는 없습니다."
    )
    st.caption(
        "주의: 이 탭은 문헌을 바탕으로 한 정성적 개념 설명입니다. ROS 농도, 세포 생존율 또는 생체 내 반응 속도를 계산하는 시뮬레이션은 아닙니다. "
        "실험에서 사용한 pH 4·7·10도 식물 액포의 실제 pH를 그대로 재현한 값은 아닙니다."
    )

    st.markdown("#### 참고 문헌")
    st.markdown(
        "- Zhao et al. (2021), [Anthocyanin stability and degradation in plants](https://doi.org/10.1080/15592324.2021.1987767)\n"
        "- Cerqueira et al. (2023), [Anthocyanins and reactive oxygen species: a team of rivals regulating plant development?](https://doi.org/10.1007/s11103-023-01362-4)"
    )
