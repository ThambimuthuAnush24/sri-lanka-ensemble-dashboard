"""Sri Lanka dengue ensemble model comparison dashboard.
Data: National Dengue Control Unit reports mirrored by lanka-data-tracker.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SOURCE = "https://raw.githubusercontent.com/dhananjaya-hbc/lanka-data-tracker/main/data/dengue.csv"
HISTORICAL_SOURCE = "https://raw.githubusercontent.com/devmividanage-hub/Dengue-Risk-MLOps-1/master/data/raw/dengue/sri_lanka_dengue_weekly.csv"
SOURCE_PAGE = "https://github.com/dhananjaya-hbc/lanka-data-tracker"
NDCU = "https://www.dengue.health.gov.lk/"
REQUIRED = {"year", "week", "week_start", "district", "cases"}
SEED = 42

st.set_page_config(page_title="Lanka Ensemble Lab", page_icon="🇱🇰", layout="wide", initial_sidebar_state="expanded")

# Use live Streamlit CSS variables so appearance changes do not need a rerun.
st.markdown("""
<style>
.block-container {padding-top: 4rem; padding-bottom: 3rem; max-width: 1400px;}
h1,h2,h3 {letter-spacing:-.02em;}
[data-testid="stMetric"] {
    background: var(--secondary-background-color) !important;
    padding: 15px 17px;
    border-radius: 14px;
    border: 1px solid var(--border-color, var(--text-color));
    min-height: 112px;
    overflow-wrap: anywhere;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {color: var(--text-color) !important; opacity:1 !important;}
[data-testid="stMetricLabel"] p {color: var(--text-color) !important; opacity:1 !important;}
section[data-testid="stSidebar"] hr {border-color:var(--border-color, var(--text-color));}
/* Keep the existing hero colors readable in either appearance. */
.hero {
    background: linear-gradient(105deg,#172d4b 0%,#10213b 58%,#102f37 100%) !important;
    border: 1px solid #294665;
    border-radius:18px;
    padding: 24px 29px;
    margin-bottom:20px;
}
.hero h1 {margin:0; color:#fff !important; font-size:clamp(1.5rem,3vw,2.2rem); line-height:1.25;}
.hero p {margin:10px 0 0;color:#d4e4f6 !important;opacity:1 !important;}
.hero p b {color:#fff !important;}
.hero .eyebrow {font-size:12px;letter-spacing:.16em;color:#81dfd5 !important;font-weight:750;margin-bottom:8px;}
.smallnote {color:var(--text-color);font-size:13px;}
/* Allow long metric values and navigation labels to remain accessible. */
[data-testid="stMetricLabel"] {height:auto; min-height:1.5rem;}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
    white-space:normal;
    overflow:visible;
    text-overflow:clip;
    overflow-wrap:anywhere;
}
[data-testid="stTabs"] [role="tablist"] {overflow-x:auto;}
[data-testid="stTabs"] [role="tab"] {white-space:nowrap; flex-shrink:0;}
@media(max-width:700px){.hero{padding:18px}.hero h1{font-size:1.6rem}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert two independently published CSV schemas into district-week format."""
    out = frame.copy()
    out.columns = [str(c).strip().lower().replace(".", "_") for c in out.columns]
    out = out.rename(columns={"start_date": "week_start", "end_date": "week_end"})
    missing = REQUIRED - set(out.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out["district"] = out["district"].astype(str).str.strip().replace({
        "NuwaraEliya": "Nuwara Eliya", "Hambanthota": "Hambantota", "Kalmune": "Kalmunai"
    })
    out["cases"] = pd.to_numeric(out["cases"], errors="coerce")
    return out[["year", "week", "week_start", "district", "cases"]].dropna(subset=["week_start", "district", "cases"])


def load_combined_from_frames(historical: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    old, new = canonicalize(historical), canonicalize(recent)
    # Reporting area names are harmonized; no invented observations or resampling.
    # Retain latest-source value if reporting year/week/district overlaps.
    return (pd.concat([old, new], ignore_index=True)
            .drop_duplicates(["year", "week", "district"], keep="last")
            .sort_values(["week_start", "district"]).reset_index(drop=True))


@st.cache_data(ttl=3600, show_spinner=False)
def download_data():
    historical = pd.read_csv(HISTORICAL_SOURCE)
    recent = pd.read_csv(SOURCE)
    return load_combined_from_frames(historical, recent)


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")
    df = raw.copy()
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce")
    df["district"] = df["district"].astype(str).str.strip()
    df = df.dropna(subset=["week_start", "cases", "district"])
    df = df[df["cases"] >= 0]
    # One record per district/week. Use published weekly cases, NOT cumulative_cases.
    df = df.sort_values(["district", "week_start"]).drop_duplicates(["district", "week_start"], keep="last")
    if df["week_start"].nunique() < 12:
        raise ValueError("At least 12 distinct weeks are needed for the lag features and temporal evaluation.")
    grp = df.groupby("district")["cases"]
    for lag in [1, 2, 3]:
        df[f"cases_lag_{lag}"] = grp.shift(lag)
    df["previous_3wk_mean"] = df.groupby("district")["cases"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52.18)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52.18)
    features = df.dropna(subset=["cases_lag_1", "cases_lag_2", "cases_lag_3", "previous_3wk_mean"]).copy()
    features = features.sort_values(["week_start", "district"]).reset_index(drop=True)
    return df, features


def feature_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    numeric = ["cases_lag_1", "cases_lag_2", "cases_lag_3", "previous_3wk_mean", "month", "week_sin", "week_cos"]
    x = pd.get_dummies(frame[numeric + ["district"]], columns=["district"], dtype=float)
    return x.reindex(columns=columns, fill_value=0.0) if columns is not None else x


def models_for(n_estimators: int, max_depth: int):
    depth = None if max_depth == 0 else max_depth
    return {
        "Random Forest · Bagging": RandomForestRegressor(n_estimators=n_estimators, max_depth=depth, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "Extra Trees · Bagging": ExtraTreesRegressor(n_estimators=n_estimators, max_depth=depth, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "AdaBoost · Boosting": AdaBoostRegressor(n_estimators=n_estimators, learning_rate=0.06, random_state=SEED),
        "HistGB · Boosting": HistGradientBoostingRegressor(max_iter=n_estimators, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=1.0, random_state=SEED),
    }


def stack_fit_predict(x_train, y_train, weeks_train, x_test, base_fitted, n_estimators, max_depth):
    """Expanding-week OOF stacking, avoiding random CV that mixes future weeks."""
    base_keys = ["Random Forest · Bagging", "Extra Trees · Bagging", "HistGB · Boosting"]
    week_arr = np.asarray(weeks_train)
    unique_weeks = np.sort(np.unique(week_arr))
    # Build OOF meta-features on expanding temporal blocks.
    boundaries = np.linspace(max(5, len(unique_weeks)//3), len(unique_weeks), 5, dtype=int)
    oof_x, oof_y = [], []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper <= lower:
            continue
        is_tr = np.isin(week_arr, unique_weeks[:lower])
        is_val = np.isin(week_arr, unique_weeks[lower:upper])
        if is_tr.sum() < 15 or is_val.sum() < 1:
            continue
        preds = []
        for key in base_keys:
            base = models_for(n_estimators, max_depth)[key]
            base.fit(x_train.loc[is_tr], y_train.loc[is_tr])
            preds.append(base.predict(x_train.loc[is_val]))
        oof_x.append(np.column_stack(preds))
        oof_y.append(y_train.loc[is_val].to_numpy())
    if not oof_x:
        raise ValueError("Insufficient history to build chronological stacking folds.")
    meta = Ridge(alpha=100.0, positive=True)
    meta.fit(np.vstack(oof_x), np.concatenate(oof_y))
    test_meta = np.column_stack([base_fitted[k].predict(x_test) for k in base_keys])
    return meta, np.maximum(0, meta.predict(test_meta)), base_keys


@st.cache_resource(show_spinner=False)
def train_models(_raw_fingerprint: str, dataframe_csv: bytes, test_share: float, n_estimators: int, max_depth: int):
    raw = pd.read_csv(io.BytesIO(dataframe_csv))
    history, feat = prepare_data(raw)
    weeks = np.sort(feat["week_start"].unique())
    cut = max(6, min(len(weeks)-2, round(len(weeks)*(1-test_share))))
    split_date = weeks[cut]
    train = feat[feat["week_start"] < split_date]
    test = feat[feat["week_start"] >= split_date]
    cols = feature_matrix(train).columns.tolist()
    x_train = feature_matrix(train, cols)
    x_test = feature_matrix(test, cols)
    y_train, y_test = train["cases"], test["cases"]
    fitted, predictions = {}, {}
    for name, estimator in models_for(n_estimators, max_depth).items():
        estimator.fit(x_train, y_train)
        fitted[name] = estimator
        predictions[name] = np.maximum(0, estimator.predict(x_test))
    meta, stacking_pred, base_keys = stack_fit_predict(x_train, y_train, train["week_start"].to_numpy(), x_test, fitted, n_estimators, max_depth)
    predictions["Stacking · Hybrid"] = stacking_pred
    leaderboard = []
    for name, pred in predictions.items():
        leaderboard.append({"Model": name, "MAE": mean_absolute_error(y_test,pred), "RMSE": np.sqrt(mean_squared_error(y_test,pred)), "R²": r2_score(y_test,pred), "Group": name.split("·")[-1].strip()})
    board = pd.DataFrame(leaderboard).sort_values("MAE").reset_index(drop=True)
    return {"history":history,"features":feat,"train":train,"test":test,"cols":cols,"fitted":fitted,"meta":meta,"base_keys":base_keys,"predictions":predictions,"board":board,"split_date":pd.Timestamp(split_date)}


def upcoming_features(history: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    last_global_week = history["week_start"].max()
    upcoming_week = last_global_week + pd.Timedelta(days=7)
    for district, group in history.groupby("district"):
        sub = group.sort_values("week_start")
        if len(sub)<3 or sub["week_start"].max()!=last_global_week:
            continue
        recent = sub.tail(3)["cases"].to_numpy()
        wk = upcoming_week.isocalendar().week
        rows.append({"district":district, "week_start":upcoming_week,"cases_lag_1":recent[-1], "cases_lag_2":recent[-2], "cases_lag_3":recent[-3],"previous_3wk_mean":recent.mean(), "month":upcoming_week.month,"week_sin":np.sin(2*np.pi*wk/52.18),"week_cos":np.cos(2*np.pi*wk/52.18)})
    return pd.DataFrame(rows)


def plot_theme(fig):
    """Let Streamlit apply the live chart theme, including text and tooltips."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(l=26, r=20, t=42, b=46),
        legend=dict(orientation="h", y=-0.24, font=dict(size=12)),
        uniformtext=dict(minsize=10, mode="show"),
    )
    # Inside labels sit on the existing pastel bars; outside labels follow
    # the chart theme. Do not force both to the same SVG text color.
    fig.update_traces(
        insidetextfont=dict(color="#14243c"), cliponaxis=False,
        selector=dict(type="bar"),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


with st.sidebar:
    st.markdown("### 🇱🇰 Lanka Ensemble Lab")
    st.caption("Sri Lankan public-health data · ML project")
    st.divider()
    st.markdown("**Data source**")
    source_choice = st.radio("Choose input", ["Combined historical + recent (~27k rows)", "Upload local CSV"], label_visibility="collapsed")
    uploaded = st.file_uploader("Upload dengue dataset",type=["csv"]) if source_choice == "Upload local CSV" else None
    st.divider()
    st.markdown("**Training settings**")
    test_share = st.slider("Hold-out test weeks", 0.15, 0.40, 0.25, 0.05, help="The latest weeks are reserved for testing.")
    n_estimators = st.slider("Trees / boosting rounds", 30, 200, 80, 10)
    max_depth = st.select_slider("Bagging max tree depth", options=[0,4,6,8,12,16],value=8, format_func=lambda v: "Unlimited" if v==0 else str(v))
    st.divider()
    st.markdown("**Methods**")
    st.caption("🌳 Bagging — independent trees\n\n⚡ Boosting — sequential learners\n\n🧩 Stacking — out-of-fold meta-learner")
    st.markdown(f"[Original Sri Lankan data mirror]({SOURCE_PAGE})")
    st.markdown(f"[National Dengue Control Unit]({NDCU})")

st.markdown("""<div class="hero"><div class="eyebrow">MACHINE LEARNING • SRI LANKA • ENSEMBLE BENCHMARK</div>
<h1>Ensemble Intelligence Dashboard</h1>
<p>Compare Bagging, Boosting and Stacking to predict <b>next-week dengue case counts</b> across Sri Lankan districts.</p></div>""", unsafe_allow_html=True)

try:
    if source_choice == "Upload local CSV":
        if uploaded is None:
            st.info("Upload a locally saved `dengue.csv` to continue. Download it using the data link below.")
            st.code(SOURCE, language=None)
            st.stop()
        raw = pd.read_csv(uploaded)
        input_name = "Uploaded CSV"
    else:
        with st.spinner("Loading Sri Lankan dengue reports…"):
            raw = download_data()
        input_name = "Historical Sri Lanka district records + 2026 reports"
    raw_csv = raw.to_csv(index=False).encode()
    # Cache key hashes exact input bytes via content digest, so uploads invalidate properly.
    import hashlib
    with st.spinner("Training and evaluating ensemble models…"):
        out = train_models(hashlib.sha256(raw_csv).hexdigest(), raw_csv,test_share,n_estimators,max_depth)
except Exception as exc:
    st.error(f"Cannot load or train on the dataset: {exc}")
    st.info("The online CSV is hosted on GitHub. If the network blocks it, download the file in a browser and select **Upload local CSV** in the sidebar. No sample or fabricated model scores are shown.")
    st.markdown(f"[Open the Sri Lankan dengue CSV]({SOURCE})")
    st.stop()

hist, feat, board, test = out["history"],out["features"],out["board"],out["test"]
best_name = board.iloc[0]["Model"]
all_districts = sorted(hist["district"].unique())

m1,m2,m3,m4 = st.columns(4)
m1.metric("Real district-week rows", f"{len(hist):,}")
m2.metric("Reporting areas", f"{hist['district'].nunique():,}")
m3.metric("Best model", best_name.split(" · ")[0])
m4.metric("Lowest test MAE", f"{board.iloc[0]['MAE']:.1f} cases")
st.caption(f"Source: {input_name}  ·  {hist['week_start'].nunique():,} reporting weeks  ·  Period: {hist['week_start'].min():%d %b %Y} – {hist['week_start'].max():%d %b %Y}  ·  Test starts: {out['split_date']:%d %b %Y}  ·  {len(out['train']):,} train / {len(test):,} test district-weeks")

tabs=st.tabs(["📊 Overview","🏆 Model comparison","📈 District explorer","🔮 Next-week forecast","🧠 Explain models","🗃️ Data & methodology"])
with tabs[0]:
    left,right=st.columns([1.5,1], gap="large")
    with left:
        st.subheader("Reported dengue cases by week")
        weekly=hist.groupby("week_start",as_index=False)["cases"].sum()
        fig=px.area(weekly,x="week_start",y="cases",labels={"week_start":"Reporting week","cases":"Weekly cases"},color_discrete_sequence=["#5dd7ca"])
        fig.update_traces(line_width=2,fillcolor="rgba(93,215,202,.17)")
        st.plotly_chart(plot_theme(fig), use_container_width=True, theme="streamlit")
    with right:
        st.subheader("Top reporting areas")
        ranking=hist.groupby("district",as_index=False)["cases"].sum().sort_values("cases",ascending=False).head(10)
        fig=px.bar(ranking.sort_values("cases"),x="cases",y="district",orientation="h",color_discrete_sequence=["#75a8ff"],labels={"cases":"Reported cases","district":""})
        st.plotly_chart(plot_theme(fig), use_container_width=True, theme="streamlit")
    st.subheader("How the ensemble pipeline works")
    a,b,c=st.columns(3)
    a.info("**01 · Bagging**\n\nRandom Forest and Extra Trees train multiple randomized decision trees and average predictions.")
    b.info("**02 · Boosting**\n\nAdaBoost and Histogram Gradient Boosting learn sequentially to improve prediction.")
    c.info("**03 · Stacking**\n\nA Ridge meta-model combines chronological out-of-fold predictions from three base learners.")

with tabs[1]:
    st.subheader("Model leaderboard — unseen weeks")
    st.dataframe(board.style.format({"MAE":"{:.2f}","RMSE":"{:.2f}","R²":"{:.3f}"}),hide_index=True,use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a:
        metric=st.selectbox("Comparison metric",["MAE","RMSE","R²"],index=0)
        ascending = metric != "R²"
        fig=px.bar(board.sort_values(metric,ascending=not ascending),x=metric,y="Model",orientation="h",color="Group",color_discrete_map={"Bagging":"#69b0fc","Boosting":"#ffba72","Hybrid":"#6bdbbf"},text_auto=".2f")
        st.plotly_chart(plot_theme(fig), use_container_width=True, theme="streamlit")
    with col_b:
        chosen=st.selectbox("Inspect a model",list(board["Model"]),index=0)
        dfpred=test[["week_start","district","cases"]].copy()
        dfpred["Predicted cases"]=out["predictions"][chosen]
        district_pick=st.selectbox("Test predictions",["All districts"] + all_districts)
        if district_pick!="All districts":
            dfpred=dfpred[dfpred["district"]==district_pick]
        daily=dfpred.groupby("week_start",as_index=False)[["cases","Predicted cases"]].sum().rename(columns={"cases":"Actual cases"})
        fig=px.line(daily,x="week_start",y=["Actual cases","Predicted cases"],markers=True,color_discrete_sequence=["#5bd9c7","#fdb570"],labels={"week_start":"Week","value":"Cases","variable":"Series"})
        st.plotly_chart(plot_theme(fig), use_container_width=True, theme="streamlit")
    st.caption("MAE and RMSE are in reported dengue cases per district-week (lower is better). R² is higher-is-better; negative R² is possible. Model selection uses test MAE for this educational comparison, not an independent production validation set.")

with tabs[2]:
    st.subheader("Explore reporting-area history")
    district=st.selectbox("District / reporting area",all_districts,index=all_districts.index("Colombo") if "Colombo" in all_districts else 0,key="explore_district")
    series=hist[hist["district"]==district].sort_values("week_start")
    c1,c2,c3=st.columns(3)
    c1.metric("Latest week",f"{int(series.iloc[-1]['cases']):,} cases")
    c2.metric("Total observed",f"{int(series['cases'].sum()):,} cases")
    c3.metric("Average / week",f"{series['cases'].mean():.1f}")
    fig=px.line(series,x="week_start",y="cases",markers=True,color_discrete_sequence=["#6bd8ca"],labels={"week_start":"Week","cases":"Reported cases"})
    st.plotly_chart(plot_theme(fig), use_container_width=True, theme="streamlit")
    st.caption("Kalmunai is tracked separately from Ampara; there are 26 reporting areas, not 26 administrative districts.")

with tabs[3]:
    st.subheader("One-week-ahead model forecast")
    st.caption("All models below are refitted to every available labeled week for prediction. This forecast is illustrative and not a health-service alert.")
    future=upcoming_features(hist)
    if future.empty:
        st.warning("No consistent latest-week district history found for forecasting.")
    else:
        choice=st.selectbox("Forecast model",list(board["Model"]),index=0)
        train_all=feat.copy()
        x_all=feature_matrix(train_all)
        x_future=feature_matrix(future,x_all.columns.tolist())
        if choice=="Stacking · Hybrid":
            # Fit a new time-respecting meta-model on all labeled observations (without touching future).
            fitted_all={}
            for name in out["base_keys"]:
                m=models_for(n_estimators,max_depth)[name]
                m.fit(x_all,train_all["cases"])
                fitted_all[name]=m
            # Only use the meta model trained on historical out-of-fold predictions.
            pred=np.maximum(0,out["meta"].predict(np.column_stack([fitted_all[k].predict(x_future) for k in out["base_keys"]])))
        else:
            model=models_for(n_estimators,max_depth)[choice]
            model.fit(x_all,train_all["cases"])
            pred=np.maximum(0,model.predict(x_future))
        forecast=future[["district","week_start","cases_lag_1"]].copy()
        forecast["Predicted cases"]=np.rint(pred).astype(int)
        forecast=forecast.rename(columns={"cases_lag_1":"Last reported week"})
        c1,c2,c3=st.columns(3)
        c1.metric("Forecast week",f"{future['week_start'].iloc[0]:%d %b %Y}")
        c2.metric("Predicted total",f"{forecast['Predicted cases'].sum():,} cases")
        c3.metric("Included areas",len(forecast))
        selected_districts=st.multiselect("Show districts",all_districts,default=all_districts[:8])
        selected=forecast[forecast["district"].isin(selected_districts)].sort_values("Predicted cases",ascending=False)
        fig=px.bar(selected,x="district",y=["Last reported week","Predicted cases"],barmode="group",color_discrete_sequence=["#77a9ff","#60d6ba"],labels={"value":"Cases","variable":"Measure","district":""})
        st.plotly_chart(plot_theme(fig), use_container_width=True, theme="streamlit")
        st.dataframe(forecast.sort_values("Predicted cases",ascending=False),hide_index=True,use_container_width=True)
        st.download_button("Download forecast CSV",forecast.to_csv(index=False),file_name="sri_lanka_dengue_forecast.csv",mime="text/csv")

with tabs[4]:
    st.subheader("Feature importance — Random Forest")
    rf=out["fitted"]["Random Forest · Bagging"]
    feature_labels=pd.Series(rf.feature_importances_,index=out["cols"]).sort_values(ascending=False).head(16)
    fi=pd.DataFrame({"Feature":feature_labels.index,"Importance":feature_labels.values})
    fig=px.bar(fi.sort_values("Importance"),x="Importance",y="Feature",orientation="h",color_discrete_sequence=["#77cabb"])
    st.plotly_chart(plot_theme(fig), use_container_width=True, theme="streamlit")
    st.caption("Impurity-based importances are indicative, not causal explanations. Correlated lag features can share importance.")
    st.markdown("""#### Understand the ensemble methods
**Bagging:** Trains many trees on randomized data/features; averaging reduces variance. Random Forest and Extra Trees both provide tree ensembles (Extra Trees uses more randomized splits).  
**Boosting:** Learns weak-to-strong corrections sequentially, often improving fit but requiring careful tuning.  
**Stacking:** Uses out-of-fold predictions from several base models as inputs to a second-stage learner; here we create these predictions using expanding-time folds to respect chronology.  
**Key limitation:** Only previous reported cases, district and calendar features are used. Weather, human mobility, reporting delays, and public-health interventions are omitted.""")

with tabs[5]:
    st.subheader("Source and reproducibility")
    st.markdown(f"**Original Sri Lankan authority:** [National Dengue Control Unit, Ministry of Health]({NDCU})  \n**2026 CSV mirror:** [Lanka Data Tracker]({SOURCE_PAGE})  \n**Historic 2006–2025 CSV:** [GitHub historical dataset]({HISTORICAL_SOURCE})  \n**Recent 2026 file:** [dengue.csv]({SOURCE})")
    st.markdown("""**Target:** number of weekly dengue cases for a district/reporting area.  
**Predictors:** lagged cases (previous 1, 2, 3 weeks), shifted 3-week mean, district, month and cyclical week-of-year.  
**Leakage prevention:** the current week's cases and cumulative figures are never used as input features; holdout weeks occur strictly after training weeks.  
**Stacking:** expanding chronological out-of-fold base-model predictions train Ridge; all fit to earlier weeks only.  
**Evaluation:** same time-based holdout for all models. Rows are district-weeks, so districts are not independent observations.  
**Limits:** observational reporting delays, revisions, incomplete weeks, no exogenous covariates. Forecasts are non-operational educational predictions.  
**Provenance caution:** historical repository states the original provider URLs and licensing metadata are not independently verified. Do not claim this historical file is a verified official government publication. The two sources use different reporting-week conventions. Model results should be treated as educational.""")
    st.write("#### First rows of dataset")
    st.dataframe(hist[["week_start","district","cases"]].sort_values("week_start").head(100),hide_index=True,use_container_width=True)
    st.download_button("Download input data",raw_csv,file_name="sri_lanka_dengue_source.csv",mime="text/csv")
    st.write("#### Test-set predictions")
    rows=test[["week_start","district","cases"]].copy().rename(columns={"cases":"actual_cases"})
    for key,val in out["predictions"].items():
        rows[key]=np.round(val,2)
    st.download_button("Download predictions",rows.to_csv(index=False),file_name="ensemble_holdout_predictions.csv",mime="text/csv")
