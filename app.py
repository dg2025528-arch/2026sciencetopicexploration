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
