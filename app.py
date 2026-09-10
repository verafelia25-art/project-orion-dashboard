import io, math
import numpy as np
import pandas as pd
import numpy_financial as npf
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Project ORION — Management Dashboard", page_icon="📡", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.2rem}
.card{border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px;background:white;min-height:105px}
.label{font-size:.78rem;color:#64748b;text-transform:uppercase}
.value{font-size:1.65rem;font-weight:750;margin-top:5px}
.note{font-size:.78rem;color:#64748b;margin-top:4px}
.bad{color:#b91c1c}.good{color:#15803d}.warn{color:#b45309}
</style>
""", unsafe_allow_html=True)

EXPECTED = ["tenant_id","region","commercial_status","planned_activation","actual_activation","monthly_arpu",
            "bandwidth_usage","installation_cost","sla_percentage","invoice_amount","payment_days","churn_probability"]

STATUS_MAP = {
    "active":"Active","existing - verified":"Existing - Verified","loi":"LOI",
    "verbal commitment":"Verbal Commitment","pipeline":"Pipeline",
    "unidentified potential":"Unidentified Potential","terminated":"Terminated"
}
PRIORITY = {"Active":7,"Existing - Verified":6,"LOI":5,"Verbal Commitment":4,
            "Pipeline":3,"Unidentified Potential":2,"Terminated":1,"Other":0}

def rp(x):
    if pd.isna(x): return "—"
    if abs(x)>=1e9: return f"Rp {x/1e9:,.2f} B"
    if abs(x)>=1e6: return f"Rp {x/1e6:,.2f} M"
    return f"Rp {x:,.0f}"

def pct(x):
    return "—" if pd.isna(x) else f"{100*x:.1f}%"

def card(label,value,note="",tone=""):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value {tone}">{value}</div><div class="note">{note}</div></div>',unsafe_allow_html=True)

def norm_status(v):
    if pd.isna(v): return "Other"
    return STATUS_MAP.get(str(v).strip().lower(),"Other")

@st.cache_data(show_spinner=False)
def prepare(file_bytes):
    xls=pd.ExcelFile(io.BytesIO(file_bytes))
    if "Tenant Dataset" not in xls.sheet_names:
        raise ValueError("Sheet 'Tenant Dataset' tidak ditemukan.")
    raw=pd.read_excel(io.BytesIO(file_bytes),sheet_name="Tenant Dataset")
    miss=[c for c in EXPECTED if c not in raw.columns]
    if miss: raise ValueError("Missing required columns: "+", ".join(miss))

    df=raw.copy()
    df["tenant_id"]=df["tenant_id"].astype("string").str.strip()
    df["commercial_status"]=df["commercial_status"].astype("string")
    for c in ["planned_activation","actual_activation"]:
        df[c]=pd.to_datetime(df[c],errors="coerce")
    nums=["monthly_arpu","bandwidth_usage","installation_cost","sla_percentage","invoice_amount","payment_days","churn_probability"]
    for c in nums: df[c]=pd.to_numeric(df[c],errors="coerce")

    profile={
        "Raw rows":len(df),
        "Unique tenant IDs":df["tenant_id"].nunique(),
        "Duplicate rows":len(df)-df["tenant_id"].nunique(),
        "Missing actual activation":int(df["actual_activation"].isna().sum()),
        "Missing invoice amount":int(df["invoice_amount"].isna().sum()),
        "Missing payment days":int(df["payment_days"].isna().sum()),
        "ARPU <= 0":int((df["monthly_arpu"]<=0).sum()),
        "Bandwidth > 1,000 Mbps":int((df["bandwidth_usage"]>1000).sum()),
        "SLA outside 0–100":int(((df["sla_percentage"]<0)|(df["sla_percentage"]>100)).sum()),
        "Payment days <0 or >365":int(((df["payment_days"]<0)|(df["payment_days"]>365)).sum()),
        "Churn outside 0–1":int(((df["churn_probability"]<0)|(df["churn_probability"]>1)).sum())
    }

    df["status_clean"]=df["commercial_status"].apply(norm_status)
    df["_priority"]=df["status_clean"].map(PRIORITY).fillna(0)
    comp=["actual_activation","monthly_arpu","sla_percentage","invoice_amount","payment_days","churn_probability"]
    df["_complete"]=df[comp].notna().sum(axis=1)
    df["_row"]=np.arange(len(df))
    clean=(df.sort_values(["tenant_id","_priority","_complete","_row"],ascending=[True,False,False,True])
             .drop_duplicates("tenant_id",keep="first").copy())
    clean["commercial_status"]=clean["status_clean"]

    clean["flag_arpu_invalid"]=clean["monthly_arpu"].notna()&(clean["monthly_arpu"]<=0)
    clean["flag_sla_invalid"]=clean["sla_percentage"].notna()&~clean["sla_percentage"].between(0,100)
    clean["flag_payment_invalid"]=clean["payment_days"].notna()&~clean["payment_days"].between(0,365)
    clean["flag_churn_invalid"]=clean["churn_probability"].notna()&~clean["churn_probability"].between(0,1)
    clean["flag_bandwidth_outlier"]=clean["bandwidth_usage"].notna()&(clean["bandwidth_usage"]>1000)

    clean.loc[clean["monthly_arpu"]<=0,"monthly_arpu"]=np.nan
    clean.loc[~clean["sla_percentage"].between(0,100),"sla_percentage"]=np.nan
    clean.loc[~clean["payment_days"].between(0,365),"payment_days"]=np.nan
    clean.loc[~clean["churn_probability"].between(0,1),"churn_probability"]=np.nan

    bmk={
        "arpu":float(clean["monthly_arpu"].median()),
        "churn":float(clean["churn_probability"].median()),
        "sla":float(clean["sla_percentage"].median()),
        "payment":float(clean["payment_days"].median()),
        "install":float(clean["installation_cost"].median())
    }
    return raw,clean,profile,bmk

def sla_penalty(mean=99.18,sd=.55):
    def cdf(x): return .5*(1+math.erf((x-mean)/(sd*math.sqrt(2))))
    return cdf(98)*.10+(cdf(99)-cdf(98))*.05+(cdf(99.5)-cdf(99))*.02

def model(arpu,churn,realization,arpu_mult=1,erosion=.02,delay=68,bad=.03,capex=17.929e9,
          opex_mult=1,sla=None,terminate=False):
    plan=np.array([400,650,850,1000,1100.])
    success=(1-.06)*(1-.03)
    cum=plan*realization*success
    new=np.r_[cum[0],np.diff(cum)]
    active=np.zeros(5); conn=np.zeros(5); tenant_month=np.zeros(5)
    for y in range(5):
        if terminate and y>=2: continue
        prev=active[y-1]*(1-churn) if y else 0
        active[y]=prev+new[y]
        a=arpu*arpu_mult*((1-erosion)**y)
        conn[y]=prev*12*a+new[y]*max(0,12*(1-delay/365))*a
        tenant_month[y]=prev*12+new[y]*12*(1-44/365)
    service=np.array([(425e6+720e6)*max(0,1-delay/365),1.145e9,1.145e9,1.145e9,1.145e9])
    if terminate: service[2:]=0
    install=new*2.5e6
    if terminate: install[2:]=0
    gross=conn+service+install+np.array([18.5e9,0,0,0,0])
    pen=(conn+service)*(sla_penalty() if sla is None else sla)
    net=gross-pen
    network=180e6*12+420000*tenant_month
    if terminate: network[2:]=0
    other=np.repeat(12.3e9/5*opex_mult,5)
    if terminate: other[2:]=0
    badexp=net*bad
    ebitda=net-badexp-network-other
    dso=(.45*15+.30*45.5+.15*75.5+.07*120)/(1-.03)
    ar=net*(1-bad)*dso/365
    dar=np.r_[ar[0],np.diff(ar)]
    avg=np.r_[ar[0]/2,(ar[:-1]+ar[1:])/2]
    funding=avg*.11
    backbone=np.zeros(5)
    for y in range(5):
        prev=active[y-1] if y else 0
        if active[y]>800 and prev<=800: backbone[y]+=1.8e9
        if active[y]>1000 and prev<=1000: backbone[y]+=1.2e9
    release=np.zeros(5); release[-1]=ar[-1]
    fcf=ebitda-dar-funding-backbone+release
    cf=np.r_[-capex,fcf]
    npv=cf[0]+sum(cf[t]/1.12**t for t in range(1,6))
    irr=npf.irr(cf)
    return dict(active=active,gross=gross,net=net,network=network,other=other,badexp=badexp,
                ebitda=ebitda,fcf=fcf,npv=float(npv),irr=float(irr) if np.isfinite(irr) else np.nan,
                revenue=float(gross.sum()),ebitda5=float(ebitda.sum()),
                margin=float(ebitda.sum()/net.sum()) if net.sum() else np.nan,
                maxcap=float(sum(fcf[t]/1.12**(t+1) for t in range(5))))

@st.cache_data(show_spinner=False)
def mc(arpu,churn,n=10000,seed=20260909):
    rng=np.random.default_rng(seed)
    real=rng.triangular(.258181818,.511818182,1,n)
    delay=rng.triangular(55,65,80,n)+(rng.random(n)<.11)*rng.triangular(30,60,120,n)
    chrn=rng.triangular(.05,np.clip(churn,.051,.249),.25,n)
    erosion=rng.triangular(0,.02,.06,n)
    bad=rng.triangular(.01,.03,.08,n)
    sla=rng.normal(99.18,.55,(n,5))
    sp=np.where(sla>=99.5,0,np.where(sla>=99,.02,np.where(sla>=98,.05,.10))).mean(axis=1)
    budgets=np.array([6.2,3.4,1.8,1.5,2.1,1,.8])*1e9
    avg=np.array([.08,.04,.04,.04,.15,.05,0]); worst=np.array([.20,.12,.12,.12,.35,.15,0])
    ov=np.zeros((n,7))
    for j in range(7):
        if worst[j]>0: ov[:,j]=rng.triangular(0,avg[j],worst[j],n)
    cap=(budgets*(1+ov)).sum(axis=1)
    npvs=np.zeros(n); irrs=np.full(n,np.nan)
    for i in range(n):
        r=model(arpu,chrn[i],real[i],erosion=erosion[i],delay=delay[i],bad=bad[i],capex=cap[i],sla=sp[i])
        npvs[i]=r["npv"]; irrs[i]=r["irr"]
    raw=pd.DataFrame({"tenant_realization":real,"activation_delay":delay,"capex":cap,
                      "sla_penalty":sp,"bad_debt":bad,"arpu_erosion":erosion,"churn":chrn,"npv":npvs})
    corr=raw.corr(numeric_only=True)["npv"].drop("npv").sort_values()
    neg=npvs[npvs<0]
    return raw,corr,{
        "p_loss":float(np.mean(npvs<0)),
        "p_irr":float(np.mean(np.where(np.isnan(irrs),True,irrs<.12))),
        "p10":float(np.percentile(npvs,10)),"p50":float(np.percentile(npvs,50)),
        "p90":float(np.percentile(npvs,90)),"var95":float(max(0,-np.percentile(npvs,5))),
        "expected_loss":float(-neg.mean()) if len(neg) else 0
    }

st.title("PROJECT ORION — Management Dashboard")
st.caption("Commercial, financial, operational, risk and data-quality decision support")

f=st.file_uploader("Upload original Project ORION tenant workbook",type=["xlsx"])
if f is None:
    st.info("Upload Project_ORION_Tenant_Dataset_5000_Rows (1).xlsx untuk menjalankan dashboard.")
    st.stop()

try:
    raw,clean,dq,bmk=prepare(f.getvalue())
except Exception as e:
    st.error(str(e)); st.stop()

st.sidebar.header("Management assumptions")
p1=st.sidebar.slider("Existing verified recognition",0.,1.,1.,.05)
p2=st.sidebar.slider("LOI recognition",0.,1.,.90,.05)
p3=st.sidebar.slider("Verbal recognition",0.,1.,.65,.05)
p4=st.sidebar.slider("Pipeline recognition",0.,1.,.40,.05)
p5=st.sidebar.slider("Unidentified recognition",0.,1.,.15,.05)
counts=np.array([180,120,170,280,350.]); probs=np.array([p1,p2,p3,p4,p5])
real=float((counts*probs).sum()/counts.sum())
erosion=st.sidebar.slider("ARPU erosion / year",0.,.10,.02,.005)
bad=st.sidebar.slider("Bad debt",0.,.10,.03,.005)
delay=st.sidebar.slider("FAB→BAST delay (days)",45,150,68,1)
om=st.sidebar.slider("Other OPEX multiplier",0.,1.5,1.,.05)

base=model(bmk["arpu"],bmk["churn"],real,erosion=erosion,delay=delay,bad=bad,opex_mult=om)
up=model(bmk["arpu"],max(.05,bmk["churn"]-.02),1,arpu_mult=1.05,erosion=0,delay=65,bad=.02,capex=16.8e9,opex_mult=.9,sla=.02)
down=model(bmk["arpu"],min(.30,bmk["churn"]+.05),.360454545,arpu_mult=.95,erosion=.04,delay=90,bad=.04,opex_mult=1.1,sla=.05)
sev=model(bmk["arpu"],min(.35,bmk["churn"]+.10),.258181818,arpu_mult=.90,erosion=.06,delay=120,bad=.06,capex=19.729e9,opex_mult=1.2,sla=.07,terminate=True)

be_real=real+(0-base["npv"])*(1-real)/(up["npv"]-base["npv"]) if up["npv"]!=base["npv"] else np.nan
be_eff=be_real*(1-.06)*(1-.03) if np.isfinite(be_real) else np.nan
be_tenant=1100*be_eff if np.isfinite(be_eff) else np.nan
recommendation="CONDITIONAL GO" if base["npv"]>0 and base["irr"]>=.12 else "NO GO"

tabs=st.tabs(["Executive","Commercial","Financial","Operations","Data Quality","Risk / Monte Carlo"])

with tabs[0]:
    c=st.columns(4)
    with c[0]: card("Recommendation",recommendation,"Current model evidence","warn" if recommendation=="CONDITIONAL GO" else "bad")
    with c[1]: card("Base NPV @12%",rp(base["npv"]),"Deterministic Base","good" if base["npv"]>=0 else "bad")
    with c[2]: card("Base IRR",pct(base["irr"]),"Hurdle 12%","good" if base["irr"]>=.12 else "bad")
    with c[3]: card("5Y EBITDA",rp(base["ebitda5"]),f"Margin {pct(base['margin'])}")
    c=st.columns(4)
    with c[0]: card("Base realization",pct(real),"Risk-weighted pipeline")
    with c[1]: card("Break-even realization",pct(be_real),"Approx. NPV=0","warn")
    with c[2]: card("Expected CAPEX",rp(17.929e9),"Budget + avg overrun")
    with c[3]: card("Max tolerable CAPEX",rp(base["maxcap"]),"PV of future Base FCF")
    sdf=pd.DataFrame({"Scenario":["Upside","Base","Downside","Severe Stress"],
                      "NPV":[up["npv"],base["npv"],down["npv"],sev["npv"]]})
    fig=px.bar(sdf,x="Scenario",y="NPV",title="NPV by Scenario"); fig.add_hline(y=0,line_dash="dash")
    st.plotly_chart(fig,use_container_width=True)
    st.warning("Management update: occupancy is 69%, below the requested 70% termination threshold.")
    st.info("Sales OPEX requires reconciliation: Rp12.3B stated 5Y OPEX versus Rp19.2B implied by Rp320M/month flat bandwidth cost.")

with tabs[1]:
    pipe=pd.DataFrame({"Status":["Existing Verified","LOI","Verbal","Pipeline","Unidentified"],
                       "Count":counts.astype(int),"Recognition Probability":probs})
    pipe["Recognized Tenants"]=pipe["Count"]*pipe["Recognition Probability"]
    st.plotly_chart(px.bar(pipe,x="Status",y=["Count","Recognized Tenants"],barmode="group",title="Pipeline vs Risk-weighted Recognition"),use_container_width=True)
    st.dataframe(pipe.style.format({"Recognition Probability":"{:.0%}","Recognized Tenants":"{:,.1f}"}),use_container_width=True,hide_index=True)
    s=clean["commercial_status"].value_counts().reset_index(); s.columns=["Status","Tenant Count"]
    st.plotly_chart(px.bar(s,x="Status",y="Tenant Count",title="Cleaned Dataset Commercial Status"),use_container_width=True)

with tabs[2]:
    years=["Y1","Y2","Y3","Y4","Y5"]
    fin=pd.DataFrame({"Metric":["Gross Revenue","Net Revenue","Network OPEX","Other OPEX","Bad Debt","EBITDA","Free Cash Flow"]})
    for i,y in enumerate(years):
        fin[y]=[base["gross"][i],base["net"][i],base["network"][i],base["other"][i],base["badexp"][i],base["ebitda"][i],base["fcf"][i]]
    show=fin.copy()
    for y in years: show[y]=show[y].map(lambda x:f"{x/1e9:,.2f}")
    st.caption("Rp billion"); st.dataframe(show,use_container_width=True,hide_index=True)
    st.plotly_chart(px.line(pd.DataFrame({"Year":years,"Revenue":base["gross"]/1e9,"EBITDA":base["ebitda"]/1e9,"FCF":base["fcf"]/1e9}).melt("Year",var_name="Metric",value_name="Rp B"),x="Year",y="Rp B",color="Metric",markers=True,title="Revenue, EBITDA and FCF"),use_container_width=True)
    sc=pd.DataFrame({"Scenario":["Upside","Base","Downside","Severe Stress"],
                     "Y5 Active":[up["active"][-1],base["active"][-1],down["active"][-1],sev["active"][-1]],
                     "5Y Revenue":[up["revenue"],base["revenue"],down["revenue"],sev["revenue"]],
                     "5Y EBITDA":[up["ebitda5"],base["ebitda5"],down["ebitda5"],sev["ebitda5"]],
                     "NPV":[up["npv"],base["npv"],down["npv"],sev["npv"]]})
    st.dataframe(sc.style.format({"Y5 Active":"{:,.0f}","5Y Revenue":lambda x:rp(x),"5Y EBITDA":lambda x:rp(x),"NPV":lambda x:rp(x)}),use_container_width=True,hide_index=True)
    c=st.columns(3)
    with c[0]: card("Minimum realization",pct(be_real),"Approx. break-even")
    with c[1]: card("Effective activated realization",pct(be_eff),"After cancel/failure")
    with c[2]: card("Break-even tenant target",f"{be_tenant:,.0f}","Cumulative activated Y5")

with tabs[3]:
    c=st.columns(4)
    with c[0]: card("FAB→BAST","65 days","Normal historical path")
    with c[1]: card("BAST→Billing","18 days","Collection timing")
    with c[2]: card("BAST >90d","11%","Historical exposure","warn")
    with c[3]: card("Activation success",pct((1-.06)*(1-.03)),"After cancel/failure")
    delaydf=pd.DataFrame({"Stage":["FAB→WO","WO→Installation","Install→Go Live","Go Live→BAST","BAST→Billing"],"Days":[14,18,12,21,18]})
    st.plotly_chart(px.bar(delaydf,x="Stage",y="Days",title="Activation & Billing Lead Time"),use_container_width=True)
    st.write(f"Expected SLA penalty: **{pct(sla_penalty())}**")
    sh=clean["sla_percentage"].dropna()
    fig=px.histogram(sh,nbins=45,title="Cleaned SLA Distribution"); fig.add_vline(x=99.5,line_dash="dash")
    st.plotly_chart(fig,use_container_width=True)

with tabs[4]:
    c=st.columns(4)
    with c[0]: card("Raw rows",f"{dq['Raw rows']:,}")
    with c[1]: card("Unique tenants",f"{dq['Unique tenant IDs']:,}")
    with c[2]: card("Duplicate rows",f"{dq['Duplicate rows']:,}","Removed in cleaned layer","warn")
    with c[3]: card("Median ARPU",rp(bmk["arpu"]))
    st.dataframe(pd.DataFrame(dq.items(),columns=["Check","Count"]),use_container_width=True,hide_index=True)
    issues=pd.DataFrame({"Issue":["Invalid ARPU","Invalid SLA","Invalid payment days","Invalid churn","Bandwidth outlier"],
                         "Count":[int(clean["flag_arpu_invalid"].sum()),int(clean["flag_sla_invalid"].sum()),
                                  int(clean["flag_payment_invalid"].sum()),int(clean["flag_churn_invalid"].sum()),
                                  int(clean["flag_bandwidth_outlier"].sum())]})
    st.plotly_chart(px.bar(issues,x="Issue",y="Count",title="Post-dedup Data Quality Issues"),use_container_width=True)
    st.download_button("Download cleaned tenant CSV",clean.to_csv(index=False).encode(),file_name="Project_ORION_Cleaned_Tenant_Data.csv",mime="text/csv")

with tabs[5]:
    with st.spinner("Running 10,000 simulations..."):
        rawmc,corr,m=mc(bmk["arpu"],bmk["churn"])
    c=st.columns(4)
    with c[0]: card("P(NPV < 0)",pct(m["p_loss"]),"Value destruction probability","bad")
    with c[1]: card("P(IRR < 12%)",pct(m["p_irr"]),"Below hurdle","bad")
    with c[2]: card("P50 NPV",rp(m["p50"]),"Median simulation","bad" if m["p50"]<0 else "good")
    with c[3]: card("VaR 95%",rp(m["var95"]),"5th percentile loss magnitude","warn")
    fig=px.histogram(rawmc,x=rawmc["npv"]/1e9,nbins=60,title="Monte Carlo NPV Distribution"); fig.add_vline(x=0,line_dash="dash"); fig.update_xaxes(title="NPV (Rp B)")
    st.plotly_chart(fig,use_container_width=True)
    cdf=corr.reset_index(); cdf.columns=["Risk Driver","Correlation with NPV"]
    st.plotly_chart(px.bar(cdf,x="Correlation with NPV",y="Risk Driver",orientation="h",title="Downside / Upside Drivers"),use_container_width=True)
    st.dataframe(pd.DataFrame([
        ["Tenant realization","Triangular(25.8%, 51.2%, 100%)","Bounded by severe/base/full pipeline"],
        ["Activation delay","89% Triangular(55,65,80) + 11% extra Triangular(30,60,120)","65-day baseline + delayed BAST exposure"],
        ["CAPEX overrun","Component-level Triangular(0, average, worst)","Case average/worst ranges"],
        ["SLA","Normal(99.18%, 0.55%)","Case historical parameters"],
        ["Bad debt","Triangular(1%, 3%, 8%)","3% case base + analyst bounds"],
        ["ARPU erosion","Triangular(0%, 2%, 6%)","Case requires uncertainty; rate not supplied"],
        ["Churn",f"Triangular(5%, {pct(bmk['churn'])}, 25%)","Mode from cleaned dataset"]
    ],columns=["Variable","Distribution","Rationale"]),use_container_width=True,hide_index=True)

st.divider()
st.caption("Project ORION upload-driven management dashboard. Tenant benchmarks come from the uploaded workbook; project case assumptions are embedded and transparently adjustable.")
