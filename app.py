import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, io, math
from datetime import datetime
from bio_connection_tab import render_bio_connection_tab

st.set_page_config(page_title="Anthocyanin Lab V3",layout="wide")
st.markdown("""
/* 비커와 설명 영역 */
.liquid-display {
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}

/* 모바일 최적화 */
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

    /* 탭은 가로 스크롤 */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        flex-wrap: nowrap;
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        white-space: nowrap;
        flex-shrink: 0;
        padding: 10px 12px;
    }

    /* 비커 아래에 설명 표시 */
    .liquid-display {
        flex-direction: column;
        align-items: center;
        gap: 12px;
    }

    /* 다운로드·일반 버튼의 글자 줄바꿈 */
    [data-testid="stDownloadButton"] button,
    [data-testid="stButton"] button {
        min-height: 44px;
        white-space: normal;
    }

    /* 가로 배치된 라디오가 좁은 화면에서 줄바꿈 */
    div[role="radiogroup"] {
        flex-wrap: wrap;
    }
}
<style>
.stApp{background:linear-gradient(135deg,#071225,#0b1830 55%,#101f38);color:#e8f0fa}
[data-testid="stSidebar"]{background:#0a162b;border-right:1px solid #294260}
h1,h2,h3{color:#f1f6ff!important}
div[data-testid="stMetric"]{background:#10233e;border:1px solid #294260;padding:12px;border-radius:10px}
.stTabs [data-baseweb="tab"]{background:#10233e;border-radius:8px 8px 0 0}
.small{color:#9fb5cc;font-size:.88rem}
</style>""",unsafe_allow_html=True)
st.title("ANTHOCYANIN LAB")
st.caption("pH-dependent anthocyanin stability · H₂O₂ treatment · absorbance kinetics")
st.info("시뮬레이션은 실제 화학 반응을 예측하거나 검증한 모델이 아니라 분석 방법을 학습하기 위한 가정 모델입니다. 실제 측정 데이터와 구분하세요.")

def ph_color(pH,fade):
    anchors=[(4,(183,55,83)),(7,(61,105,170)),(10,(82,142,102))]
    a,b=(anchors[0],anchors[1]) if pH<=7 else (anchors[1],anchors[2])
    x=np.clip((pH-a[0])/(b[0]-a[0]),0,1)
    c=[int(a[1][i]*(1-x)+b[1][i]*x) for i in range(3)]
    c=[int(c[i]*(1-fade)+225*fade) for i in range(3)]
    return '#%02x%02x%02x'%tuple(c)

def prepare(df):
    d=df.copy()
    for c in ["time","pH","absorbance"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["group"]=d["group"].astype(str).str.strip()
    d=d.dropna(subset=["time","pH","absorbance"]).sort_values(["pH","group","time"])
    d["A_norm"]=d.groupby(["pH","group"])["absorbance"].transform(lambda s:s/s.iloc[0] if len(s) and s.iloc[0]!=0 else np.nan)
    return d

def model_fit(t,a):
    t=np.asarray(t,float); a=np.asarray(a,float); a0=a[0]; rows=[]
    if len(a)<3:return pd.DataFrame()
    models=[]
    p=np.polyfit(t,a,1); models.append(("0차 선형",np.polyval(p,t),max(0,-p[0]),"A=A₀−kt"))
    if np.all(a>0):
        p=np.polyfit(t,np.log(a/a0),1); models.append(("1차 지수형",a0*np.exp(np.polyval(p,t)),max(0,-p[0]),"A=A₀e⁻ᵏᵗ"))
        p=np.polyfit(t,1/a,1); models.append(("2차 역수형",1/np.maximum(np.polyval(p,t),1e-12),max(0,p[0]*a0*a0),"1/A=1/A₀+kt"))
    ss=np.sum((a-a.mean())**2)
    for n,pred,k,eq in models:
        err=a-pred; rmse=np.sqrt(np.mean(err**2)); r2=1-np.sum(err**2)/ss if ss else np.nan
        rows.append({"모형":n,"겉보기 k":k,"R²":r2,"RMSE":rmse,"식":eq})
    return pd.DataFrame(rows)

st.sidebar.header("실험 조건")
ph=st.sidebar.slider("pH",4.0,10.0,7.0,0.1)
buffer=st.sidebar.number_input("완충용액 부피 (mL)",1.0,500.0,45.0,1.0)
extract=st.sidebar.number_input("안토시아닌 추출액 부피 (mL)",0.1,100.0,5.0,0.5)
treated=st.sidebar.toggle("H₂O₂ 처리군",True)
duration=st.sidebar.slider("관찰 시간 (초)",30,3600,300,30)
st.sidebar.markdown("**온도: 25 °C (고정)**")
st.sidebar.caption("속도상수 직접 설정 기능은 없습니다. 시뮬레이션의 가정값과 실험 데이터에서 추정한 겉보기 값은 구분합니다.")
tabs = st.tabs([
    "실험 시뮬레이션",
    "데이터 분석",
    "조건별 비교",
    "화학 개념",
    "생명과학 연결",
    "보고서·조건 저장",
])

with tabs[0]:
    st.subheader("화학 실험 시뮬레이션")
    c1,c2,c3=st.columns(3); c1.metric("전체 용액 부피",f"{buffer+extract:.1f} mL"); c2.metric("관찰 시간",f"{duration} s"); c3.metric("온도","25 °C")
    st.caption("용액 부피는 시작부터 끝까지 일정하며 색상만 시간에 따라 변화합니다. 색상은 정성적 교육용 근사 표현입니다.")
    # Deliberately illustrative parameters; no real H2O2 concentration is implied.
    A0=.65*(.75+.25*np.exp(-((ph-7)/2.4)**2))
    k=(.0018 if treated else .00015)*(1+.10*abs(ph-7))
    t=np.linspace(0,duration,160); rel=np.exp(-k*t); A=A0*rel
    fig,ax=plt.subplots(figsize=(8,3.3)); ax.plot(t,A,lw=2.4); ax.set(xlabel="Time (s)",ylabel="Absorbance (a.u.)"); ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)
    now=st.slider("색 변화 관찰 시점 (초)",0,duration,0,1)
    fade=(1-math.exp(-k*now))*(1 if treated else .18)
    liquid=ph_color(ph,fade)
    st.markdown(f"""<div style="display:flex;align-items:center;gap:24px">
    <div style="width:180px;height:220px;border:3px solid #a9bfd4;border-top:0;border-radius:8px 8px 28px 28px;background:linear-gradient(to bottom,transparent 0 24%,{liquid} 24% 100%);position:relative">
    <div style="position:absolute;top:22%;left:0;right:0;border-top:1px solid #fff9"></div></div>
    <div><b>시점: {now} s</b><p class="small">상대 흡광도(가정값): {np.interp(now,t,rel):.3f}</p><p class="small">표시 색상은 실제 분광 데이터로 보정되지 않았습니다.</p></div></div>""",unsafe_allow_html=True)
    choice=st.radio("수학적 모형",["1차 지수형","0차 선형","2차 역수형"],horizontal=True)
    if choice=="1차 지수형": pred=A0*np.exp(-k*t)
    elif choice=="0차 선형": pred=np.maximum(0,A0-A0*k*t)
    else: pred=A0/(1+k*A0*t)
        # 선택 메뉴는 한글, 그래프 범례는 영어로 표시
    legend_labels = {
        "1차 지수형": "First-order exponential model",
        "0차 선형": "Zero-order linear model",
        "2차 역수형": "Second-order reciprocal model",
    }

    f, ax = plt.subplots(figsize=(8, 3.2))

    ax.plot(
        t,
        pred,
        label=legend_labels[choice],
        color="#2563eb",
        linewidth=2.4,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absorbance, A")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, A0 * 1.15)

    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    f.tight_layout()
    st.pyplot(f)
    plt.close(f)
    st.warning("초기 흡광도와 속도 계수는 교육용 가정값입니다. 실제 안토시아닌 조성, H₂O₂ 농도 및 반응 경로를 반영한 검증 모델이 아닙니다.")

with tabs[1]:
    st.subheader("실험 데이터 분석")
    st.write("CSV 필수 열: `time`, `pH`, `group`, `absorbance`")
    file=st.file_uploader("측정 CSV 업로드",type=["csv"])
    if file:
        try:
            raw=pd.read_csv(file)
            required={"time","pH","group","absorbance"}
            if not required.issubset(raw.columns): st.error("필수 열 누락: "+", ".join(required-set(raw.columns)))
            else: st.session_state["data"]=prepare(raw); st.success(f"{len(st.session_state['data'])}개 유효 행을 불러왔습니다.")
        except Exception as e: st.error(str(e))
    if "data" in st.session_state:
        d=st.session_state["data"]
        q1,q2,q3,q4=st.columns(4); q1.metric("유효 행",len(d)); q2.metric("결측 포함 원본 행",int(d.isna().any(axis=1).sum())); q3.metric("중복 행",int(d.duplicated().sum())); q4.metric("시간 범위",f"{d.time.min():g}–{d.time.max():g} s")
        st.dataframe(d,use_container_width=True,height=190)
        fig,ax=plt.subplots(figsize=(8,3.8))
        for (p,g),s in d.groupby(["pH","group"]): ax.plot(s.time,s.absorbance,marker=".",ms=3,label=f"pH {p:g} · {g}")
        ax.set(xlabel="Time (s)",ylabel="Absorbance"); ax.grid(alpha=.25); ax.legend(); st.pyplot(fig); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,3.5))
        for (p,g),s in d.groupby(["pH","group"]): ax.plot(s.time,s.A_norm,label=f"pH {p:g} · {g}")
        ax.set(xlabel="Time (s)",ylabel="A/A₀"); ax.grid(alpha=.25); ax.legend(); st.pyplot(fig); plt.close(fig)
        summaries=[]; fits=[]
        for (p,g),s in d.groupby(["pH","group"]):
            s=s.sort_values("time")
            if len(s)<2: continue
            a0=float(s.absorbance.iloc[0]); af=float(s.absorbance.iloc[-1]); dt=float(s.time.iloc[-1]-s.time.iloc[0])
            summaries.append({"pH":p,"group":g,"초기 A":a0,"최종 A":af,"상대 감소율(%)":(a0-af)/a0*100 if a0 else np.nan,"평균 감소 속도(A/s)":(a0-af)/dt if dt>0 else np.nan})
            mf=model_fit(s.time.to_numpy()-s.time.iloc[0],s.absorbance.to_numpy())
            if not mf.empty:
                best=mf.sort_values(["RMSE","R²"],ascending=[True,False]).iloc[0]
                fits.append({"pH":p,"group":g,"적합도 기준 모형":best["모형"],"겉보기 k":best["겉보기 k"],"R²":best["R²"],"RMSE":best["RMSE"]})
        st.markdown("**조건별 요약**"); st.dataframe(pd.DataFrame(summaries).round(6),use_container_width=True)
        if fits:
            st.markdown("**0차·1차·2차 모형 비교 결과**"); st.dataframe(pd.DataFrame(fits).round(6),use_container_width=True)
        st.caption("적합도가 높은 모형이 실제 반응 차수임을 증명하지는 않습니다. 잔차 패턴과 반복 실험도 검토해야 합니다.")
        st.download_button("정리된 데이터 CSV 다운로드",d.to_csv(index=False).encode("utf-8-sig"),"anthocyanin_processed.csv","text/csv")
        st.markdown("**해석상 주의점**")
        st.write("- 정규화 A/A₀는 초기값 차이를 고려하지만 pH에 따른 화학종 및 몰흡광계수 차이를 제거하지 않습니다.")
        st.write("- 특정 파장의 흡광도 감소를 실제 분해량과 동일시하지 않습니다.")
        st.write("- 반복 실험이 없다면 통계적 유의성을 주장하지 않습니다.")
    else: st.info("CSV를 업로드하면 데이터 품질 검사, 원본·정규화 그래프 및 속도 분석이 표시됩니다.")

with tabs[2]:
    st.subheader("조건별 비교")
    if "data" in st.session_state:
        d=st.session_state["data"]; fig,ax=plt.subplots(figsize=(8,4))
        for (p,g),s in d.groupby(["pH","group"]): ax.plot(s.time,s.A_norm,label=f"pH {p:g} · {g}")
        ax.axhline(1,color="gray",ls="--",lw=.8); ax.set(xlabel="Time (s)",ylabel="A/A₀"); ax.grid(alpha=.25); ax.legend(); st.pyplot(fig); plt.close(fig)
    else: st.info("데이터 분석 탭에서 CSV를 먼저 업로드하세요.")
    st.markdown("1. 동일 pH에서 H₂O₂ 처리군과 대조군의 상대 흡광도 변화를 비교합니다.\n2. 각 조건의 초기 흡광도를 기준으로 정규화합니다.\n3. pH별 차이는 화학종 분포와 몰흡광계수 차이를 고려해 해석합니다.\n4. 반복 실험이 없다면 통계적 유의성을 단정하지 않습니다.")

with tabs[3]:
    st.subheader("화학 개념 학습")
    concepts=[
    ("1. 안토시아닌의 산·염기 반응","안토시아닌은 pH에 따라 양성자 이동, 수화 반응 및 구조 전환 등의 영향을 받아 여러 화학종 사이의 평형이 변화한다. 대표적인 구조는 플라빌륨 양이온, 퀴노이드 형태, 헤미케탈, 칼콘이며 pH와 용액 환경에 따라 서로 다른 비율로 분포한다. 각 화학종은 서로 다른 전자 구조와 에너지 준위를 가지므로 빛 흡수 특성도 달라진다."),
    ("2. pH 4 이하에서의 안토시아닌 구조","산성 조건에서는 플라빌륨 양이온의 비율이 상대적으로 높아져 붉은색 계열을 나타낼 수 있다. 실제 색상은 안토시아닌 종류, 추출액 조성, 농도 및 실험 조건에 따라 달라진다."),
    ("3. pH 4~7에서의 안토시아닌 구조","pH가 증가하면 플라빌륨 양이온의 비율이 감소하고 퀴노이드 형태 등 다른 화학종의 비율이 변화하여 보라색이나 청색 계열을 나타낼 수 있다. 특정 pH에서 특정 색이 반드시 나타나는 것은 아니다."),
    ("4. pH 7 이상에서의 안토시아닌 구조","염기성 조건에서는 퀴노이드 형태와 칼콘 계열 등 여러 화학종의 분포가 달라질 수 있다. 강한 알칼리성 환경에서는 구조 안정성이 낮아지고 비가역적 분해가 일어날 가능성도 있다."),
    ("5. 과산화수소(H₂O₂)에 의한 안토시아닌의 변화","H₂O₂는 산화제로 작용할 수 있으며 안토시아닌의 구조 변화나 분해를 유발할 수 있다. 공액 전자계가 변화하면 가시광선 흡수 특성도 달라진다. 구체적인 반응 경로와 생성물은 안토시아닌 종류와 조건에 따라 달라지며 본 실험에서 직접 확인하지 않는다."),
    ("6. 흡광 특성과 Beer–Lambert 법칙","흡광도와 투과도는 A=-log₁₀(T), T=I/I₀의 관계를 가지며 Beer–Lambert 법칙은 A=εlc로 표현된다. 일정한 화학종과 조건에서 흡광도는 농도에 비례하지만 pH에 따른 화학종 및 몰흡광계수 변화 때문에 서로 다른 pH의 흡광도를 곧바로 농도 차이로 해석해서는 안 된다."),
    ("7. 화학 평형","화학 평형은 가역 반응에서 정반응과 역반응의 속도가 같아 물질의 농도가 거시적으로 일정하게 유지되는 동적 평형이다. H₂O₂에 의한 모든 변화가 가역 평형인 것은 아니다."),
    ("8. 르샤틀리에 원리","평형 상태의 화학계에 농도나 온도 등의 변화가 가해지면 계는 그 변화의 영향을 줄이는 방향으로 새로운 평형에 도달하는 경향이 있다. pH 변화에 따른 안토시아닌 화학종 분포를 이해하는 데 활용한다."),
    ("9. 산화·환원 반응","산화·환원 반응은 반응 과정에서 전자가 이동하고 산화수가 변하는 반응이다. 산화는 전자를 잃거나 산화수가 증가하는 과정이며 환원은 전자를 얻거나 산화수가 감소하는 과정이다. 본 실험에서 H₂O₂는 산화제로 작용할 수 있다."),
    ("10. 반응 속도","반응 속도는 단위 시간당 반응물의 농도 감소량 또는 생성물의 농도 증가량으로 나타낸다. 본 실험에서는 흡광도 변화율로 겉보기 탈색 속도를 비교한다. 평균 탈색 속도는 -(A₂-A₁)/(t₂-t₁), 순간 탈색 속도는 -dA/dt로 나타낼 수 있다."),
    ("11. 완충 용액과 공통 이온 효과","완충 용액은 약산과 짝염기 또는 약염기와 짝산으로 구성되어 산·염기 첨가에 따른 pH 변화를 억제한다. HA/A⁻ 완충계에서는 A⁻+H⁺→HA, HA+OH⁻→A⁻+H₂O 반응이 일어난다. 공통 이온 효과는 약전해질 평형에 이미 존재하는 이온과 같은 이온을 첨가했을 때 평형이 이동하는 현상이며 완충 작용과 관련되지만 동일한 개념은 아니다.")
    ]
    for title,body in concepts:
        with st.expander(title): st.write(body)
# 새로 추가한 생명과학 탭
with tabs[4]:
    render_bio_connection_tab()


# 기존 보고서 탭은 5번으로 이동
with tabs[5]:
    st.subheader("보고서 및 조건 저장")
    settings={"pH":ph,"buffer_volume_mL":buffer,"anthocyanin_extract_volume_mL":extract,"H2O2_treatment":treated,"observation_time_s":duration,"temperature_C":25}
    st.json(settings)
    st.download_button("실험 조건 JSON 저장",json.dumps(settings,ensure_ascii=False,indent=2).encode("utf-8"),"anthocyanin_conditions.json","application/json")
    jf=st.file_uploader("저장한 조건 JSON 불러오기",type=["json"])
    if jf:
        try: st.write("불러온 조건(확인용):",json.load(jf)); st.caption("불러온 값은 확인용이며 사이드바에 자동 적용되지는 않습니다.")
        except Exception as e: st.error(f"JSON 읽기 오류: {e}")
    st.markdown("---")
    lines=[f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}",f"조건: pH {ph:g}, 완충용액 {buffer:g} mL, 추출액 {extract:g} mL",f"H₂O₂ 처리: {'처리군' if treated else '미처리 조건'}, 관찰 시간 {duration} s, 온도 25 °C","시뮬레이션은 가정 모델이며 실제 반응을 검증한 모델이 아닙니다.","흡광도 기반 속도는 겉보기 탈색 속도로 해석해야 합니다."]
    if "data" in st.session_state: lines.append(f"업로드 데이터 유효 행 수: {len(st.session_state['data'])}")
    st.download_button("PDF 요약 보고서 다운로드","\\n".join(lines).encode("utf-8"),"anthocyanin_report.txt","text/plain")
    st.caption("현재 버전은 PDF 라이브러리 의존성 없이 요약 보고서를 TXT로 제공합니다. 브라우저 인쇄 기능으로 PDF 저장이 가능합니다.")

st.markdown("---")
st.markdown('<p class="small">Anthocyanin Lab V3 · 교육용 분석 도구 · 가정 시뮬레이션과 실제 측정 결과를 구분하세요.</p>',unsafe_allow_html=True)
