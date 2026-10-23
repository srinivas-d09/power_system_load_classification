"""Interactive dashboard for Power System Load Type Classification."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Power Load Intelligence", page_icon="⚡", layout="wide")
st.title("⚡ Power System Load Type Classification")
st.caption("Exploratory analysis, model evaluation, and single-record prediction")

data_path=Path("data/load_data.csv")
metrics_path=Path("artifacts/metrics.json")
model_path=Path("artifacts/load_type_model.joblib")

with st.sidebar:
    st.header("Project controls")
    uploaded=st.file_uploader("Optional: upload a CSV for exploration", type=["csv"])
    st.markdown("**Expected target:** `Load_Type`")
    st.markdown("Training uses the latest calendar month as a chronological holdout.")

df=None
if uploaded is not None:
    df=pd.read_csv(uploaded)
elif data_path.exists():
    df=pd.read_csv(data_path)

if df is not None:
    st.subheader("Dataset overview")
    c1,c2,c3=st.columns(3)
    c1.metric("Rows",f"{len(df):,}")
    c2.metric("Columns",len(df.columns))
    target=next((c for c in df.columns if c.lower().replace("_","").replace(" ","")=="loadtype"),None)
    c3.metric("Target available","Yes" if target else "No")
    with st.expander("Preview data"):
        st.dataframe(df.head(30), use_container_width=True)
    if target:
        st.subheader("Load type distribution")
        counts=df[target].astype(str).value_counts()
        fig,ax=plt.subplots(figsize=(8,4)); counts.plot(kind="bar",ax=ax)
        ax.set_xlabel("Load type"); ax.set_ylabel("Records"); ax.set_title("Class balance")
        st.pyplot(fig); plt.close(fig)
    numeric=df.select_dtypes("number")
    if numeric.shape[1]>=2:
        st.subheader("Correlation heatmap")
        fig,ax=plt.subplots(figsize=(10,6))
        im=ax.imshow(numeric.corr(),aspect="auto",cmap="coolwarm",vmin=-1,vmax=1)
        ax.set_xticks(range(len(numeric.columns))); ax.set_xticklabels(numeric.columns,rotation=70,ha="right")
        ax.set_yticks(range(len(numeric.columns))); ax.set_yticklabels(numeric.columns)
        fig.colorbar(im,ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
else:
    st.info("No dataset found yet. Add your assignment CSV at `data/load_data.csv` or upload it in the sidebar. The project intentionally does not invent results without the real dataset.")

if metrics_path.exists():
    metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Latest-month holdout evaluation")
    a,b,c,d=st.columns(4)
    a.metric("Accuracy",f"{metrics['accuracy']:.1%}")
    b.metric("Balanced accuracy",f"{metrics.get('balanced_accuracy',0):.1%}")
    c.metric("Macro F1",f"{metrics.get('macro_f1',0):.1%}")
    d.metric("Test month",metrics["test_month"])
    st.caption(f"Train rows: {metrics['train_rows']:,} · Test rows: {metrics['test_rows']:,}")
    st.json(metrics["classification_report"])
    for p in sorted(Path("figures").glob("*.png")):
        st.image(str(p), caption=p.stem.replace("_"," ").title(), use_container_width=True)
else:
    st.info("Evaluation charts will appear here after you run `python train_model.py`.")

if model_path.exists():
    st.subheader("Predict one record")
    bundle=joblib.load(model_path)
    model=bundle["model"]; features=bundle["feature_columns"]
    st.write("Enter numeric feature values. Time-derived features are also required by the trained model.")
    values={}
    medians=bundle.get("medians",{})
    cols=st.columns(3)
    for i,feature in enumerate(features):
        values[feature]=cols[i%3].number_input(feature, value=float(medians.get(feature,0) or 0), format="%.4f")
    if st.button("Predict load type", type="primary"):
        prediction=model.predict(pd.DataFrame([values],columns=features))[0]
        st.success(f"Predicted load type: **{prediction}**")
        if hasattr(model,"predict_proba"):
            probs=model.predict_proba(pd.DataFrame([values],columns=features))[0]
            st.dataframe(pd.DataFrame({"Load type":model.classes_,"Probability":probs}).sort_values("Probability",ascending=False),use_container_width=True)
