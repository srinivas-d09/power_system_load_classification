"""
Power System Load Type Classification
Run: python train_model.py --data data/load_data.csv
Uses the latest calendar month as a chronological test set.
"""
from __future__ import annotations
import argparse, json, os, warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

TARGET = "Load_Type"
CLASS_ORDER = ["Light_Load", "Medium_Load", "Maximum_Load"]
PALETTE = {"Light_Load":"#3977A8", "Medium_Load":"#E49B45", "Maximum_Load":"#C84D4D"}
TIME_COL_CANDIDATES = ["Date_Time", "Datetime", "Date", "Timestamp", "date", "datetime"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common variants in the assignment dataset to canonical column names."""
    df = df.copy()
    rename = {}
    for col in df.columns:
        c = str(col).strip()
        low = c.lower().replace(" ", "").replace("_", "").replace(".", "")
        if low in {"loadtype", "load_type"}: rename[col] = TARGET
        elif low in {"datetime", "date", "timestamp"}: rename[col] = "Date_Time"
        elif low in {"usagekwh", "industryenergyconsumption"}: rename[col] = "Usage_kWh"
        elif low in {"laggingcurrentreactivepowerkvarh", "laggingcurrentkvarh", "laggingcurrentreactivepower"}: rename[col] = "Lagging_Current_Reactive.Power_kVarh"
        elif low in {"leadingcurrentreactivepowerkvarh", "leadingcurrentkvarh", "leadingcurrentreactivepower"}: rename[col] = "Leading_Current_Reactive_Power_kVarh"
        elif low in {"co2", "co2tco2", "co2ppm"}: rename[col] = "CO2(tCO2)"
        elif low in {"nsm", "numberofsecondsfrommidnight"}: rename[col] = "NSM"
        elif low in {"laggingcurrentpowerfactor"}: rename[col] = "Lagging_Current_Power_Factor"
        elif low in {"leadingcurrentpowerfactor"}: rename[col] = "Leading_Current_Power_Factor"
    df = df.rename(columns=rename)
    return df

def load_and_prepare(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPlace your provided CSV at data/load_data.csv, or pass --data PATH."
        )
    df = normalize_columns(pd.read_csv(path))
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Available columns: {list(df.columns)}")
    time_col = next((c for c in TIME_COL_CANDIDATES if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No date/time column found. Expected one of {TIME_COL_CANDIDATES}.")
    if time_col != "Date_Time":
        df = df.rename(columns={time_col:"Date_Time"})
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], dayfirst=True, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Date_Time", TARGET]).copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET] != ""].sort_values("Date_Time").drop_duplicates().reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"Only {len(df)} valid rows found; at least 10 are required.")
    if df["Date_Time"].dt.to_period("M").nunique() < 2:
        raise ValueError("At least two calendar months are required for a last-month holdout.")
    print(f"Loaded {len(df):,} usable rows (removed {before-len(df):,} rows with invalid date/target).")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["Date_Time"]
    out["hour"] = dt.dt.hour + dt.dt.minute / 60 + dt.dt.second / 3600
    out["day"] = dt.dt.day
    out["weekday"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["dayofyear"] = dt.dt.dayofyear
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24)
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24)
    out["weekday_sin"] = np.sin(2*np.pi*out["weekday"]/7)
    out["weekday_cos"] = np.cos(2*np.pi*out["weekday"]/7)
    return out

def make_figures(df, X_train, X_test, y_test, pred, model, features, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    order = [c for c in CLASS_ORDER if c in df[TARGET].unique()]
    if not order: order = sorted(df[TARGET].unique())
    # 1 class balance
    fig, ax = plt.subplots(figsize=(8,5))
    counts = df[TARGET].value_counts().reindex(order, fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=[PALETTE.get(c,"#777777") for c in counts.index])
    ax.bar_label(bars, fmt="%d", padding=3); ax.set_title("1. Load type distribution"); ax.set_ylabel("Rows")
    fig.tight_layout(); fig.savefig(out_dir/"01_class_distribution.png", dpi=160); plt.close(fig)
    # 2 usage over time
    if "Usage_kWh" in df:
        fig, ax = plt.subplots(figsize=(12,5))
        for label in order:
            sub = df[df[TARGET] == label]
            ax.scatter(sub["Date_Time"], sub["Usage_kWh"], s=8, alpha=.55, label=label,
                       color=PALETTE.get(label, None))
        ax.set_title("2. Energy usage over time"); ax.set_xlabel("Date"); ax.set_ylabel("Usage (kWh)")
        ax.legend(); fig.tight_layout(); fig.savefig(out_dir/"02_usage_over_time.png", dpi=160); plt.close(fig)
    # 3 correlation
    num = df.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(11,8))
        sns.heatmap(num.corr(), cmap="coolwarm", center=0, annot=num.shape[1] <= 10, fmt=".2f", ax=ax)
        ax.set_title("3. Numeric feature correlation"); fig.tight_layout()
        fig.savefig(out_dir/"03_correlation_heatmap.png", dpi=160); plt.close(fig)
    # 4 boxplot usage by class
    if "Usage_kWh" in df:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(data=df, x=TARGET, y="Usage_kWh", order=order, ax=ax, showfliers=False)
        ax.set_title("4. Usage distribution by load type"); fig.tight_layout()
        fig.savefig(out_dir/"04_usage_by_class.png", dpi=160); plt.close(fig)
    # 5 hourly profile
    if "hour" in features and "Usage_kWh" in df:
        temp = df.copy(); temp["hour_of_day"] = temp["Date_Time"].dt.hour
        hourly = temp.groupby(["hour_of_day", TARGET])["Usage_kWh"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10,5))
        sns.lineplot(data=hourly, x="hour_of_day", y="Usage_kWh", hue=TARGET, hue_order=order, marker="o", ax=ax)
        ax.set_title("5. Mean usage by hour"); fig.tight_layout()
        fig.savefig(out_dir/"05_hourly_usage.png", dpi=160); plt.close(fig)
    # 6 feature importance
    imp = pd.Series(model.feature_importances_, index=features).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(9,7)); imp.plot.barh(ax=ax)
    ax.set_title("6. Top feature importances"); ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(out_dir/"06_feature_importance.png", dpi=160); plt.close(fig)
    # 7 confusion counts
    labels = order
    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("7. Confusion matrix (counts)"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(out_dir/"07_confusion_matrix.png", dpi=160); plt.close(fig)
    # 8 normalized confusion matrix
    cmn = confusion_matrix(y_test, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(cmn, annot=True, fmt=".1%", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax, vmin=0, vmax=1)
    ax.set_title("8. Normalized confusion matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(out_dir/"08_confusion_normalized.png", dpi=160); plt.close(fig)
    # 9 per-class metrics
    rep = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    metric_names = ["precision","recall","f1-score"]
    vals = [[rep.get(label,{}).get(m,0) for label in labels] for m in metric_names]
    fig, ax = plt.subplots(figsize=(10,5)); x=np.arange(len(labels)); width=.24
    for i,m in enumerate(metric_names): ax.bar(x+(i-1)*width, vals[i], width, label=m.title())
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,1.08); ax.set_ylabel("Score")
    ax.set_title("9. Per-class evaluation metrics"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir/"09_classification_metrics.png", dpi=160); plt.close(fig)
    # 10 predicted vs actual timeline
    result = pd.DataFrame({"Date_Time":df.loc[y_test.index,"Date_Time"].values,
                           "Actual":y_test.values, "Predicted":pred})
    fig, ax = plt.subplots(figsize=(13,4))
    ax.scatter(result["Date_Time"], result["Actual"], label="Actual", s=14, alpha=.7)
    ax.scatter(result["Date_Time"], result["Predicted"], label="Predicted", s=14, alpha=.7, marker="x")
    ax.set_title("10. Actual vs predicted labels across test month"); ax.set_xlabel("Date")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir/"10_test_predictions_timeline.png", dpi=160); plt.close(fig)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data", default="data/load_data.csv", help="Path to input CSV")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--model", choices=["extra_trees","random_forest"], default="extra_trees")
    args=parser.parse_args()
    Path(args.artifacts).mkdir(parents=True, exist_ok=True)
    df=load_and_prepare(args.data)
    engineered=engineer_features(df)
    last_month=engineered["Date_Time"].dt.to_period("M").max()
    test_mask=engineered["Date_Time"].dt.to_period("M").eq(last_month)
    feature_cols=[c for c in engineered.columns if c not in ["Date_Time",TARGET]]
    X=engineered[feature_cols].replace([np.inf,-np.inf],np.nan)
    # Numeric median imputation is learned on training data only.
    X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
    y_train, y_test = engineered.loc[~test_mask,TARGET], engineered.loc[test_mask,TARGET]
    medians=X_train.median(numeric_only=True)
    X_train=X_train.fillna(medians).fillna(0)
    X_test=X_test.fillna(medians).fillna(0)
    if X_train.empty or X_test.empty: raise ValueError("Chronological split produced an empty train or test set.")
    if y_train.nunique()<2: warnings.warn("Training data contains fewer than two classes.")
    model = (ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
             if args.model=="extra_trees" else
             RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,} | Holdout month: {last_month}")
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    labels=[c for c in CLASS_ORDER if c in set(y_train)|set(y_test)]
    labels += sorted((set(y_train)|set(y_test))-set(labels))
    report=classification_report(y_test,pred,labels=labels,output_dict=True,zero_division=0)
    metrics={"test_month":str(last_month),"train_rows":int(len(X_train)),"test_rows":int(len(X_test)),
             "accuracy":float(accuracy_score(y_test,pred)),
             "balanced_accuracy":float(balanced_accuracy_score(y_test,pred)),
             "macro_precision":float(precision_score(y_test,pred,average="macro",zero_division=0)),
             "macro_recall":float(recall_score(y_test,pred,average="macro",zero_division=0)),
             "macro_f1":float(f1_score(y_test,pred,average="macro",zero_division=0)),
             "classification_report":report,"labels":labels,
             "confusion_matrix":confusion_matrix(y_test,pred,labels=labels).tolist(),
             "features":feature_cols,"model":args.model}
    print("\nEvaluation on latest-month holdout")
    print(classification_report(y_test,pred,labels=labels,zero_division=0))
    print(f"Accuracy: {metrics['accuracy']:.4f} | Balanced accuracy: {metrics['balanced_accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
    result=make_figures(engineered,X_train,X_test,y_test,pred,model,feature_cols,args.figures)
    bundle={"model":model,"feature_columns":feature_cols,"medians":medians.to_dict(),"test_month":str(last_month)}
    joblib.dump(bundle,Path(args.artifacts)/"load_type_model.joblib")
    with open(Path(args.artifacts)/"metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
    result.to_csv(Path(args.artifacts)/"last_month_predictions.csv",index=False)
    print(f"\nSaved model/metrics/predictions in {args.artifacts}/ and visualizations in {args.figures}/")
if __name__=="__main__": main()
