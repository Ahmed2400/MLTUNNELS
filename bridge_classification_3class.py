"""
Bridge/Dam Condition Assessment — 3-Class Classification
Classes: Good/Excellent (1+2) | Fair (3) | Poor/Failing (4)
Pipeline: SMOTE + Optuna (LGB / CatBoost / ExtraTrees) + Soft Voting
"""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, time
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_recall_fscore_support)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
TUNNELS_PATH = "/root/.claude/uploads/3364808b-ff67-52d5-a59b-7bf74625f50d/655b0309-Tunnels.xlsx"
NATION_PATH  = "/root/.claude/uploads/3364808b-ff67-52d5-a59b-7bf74625f50d/a18014c9-nation.xlsx"

# 3-class label map (merge Excellent+Good → "Good/Excellent")
LABEL_MAP3 = {1: "Good/Excellent", 2: "Fair", 3: "Poor/Failing"}

# ── Load & join ───────────────────────────────────────────────────────────────
df_t   = pd.read_excel(TUNNELS_PATH)
df_raw = pd.read_excel(NATION_PATH, header=1)
print(f"Raw tunnels: {df_t.shape}")
print(f"4-class dist: {df_t['Condition Assessment'].value_counts().sort_index().to_dict()}")

EXTRA = ['Dam Name','Hazard Potential Classification','Last Inspection Date',
         'Inspection Frequency','EAP Prepared','Primary Purpose','Primary Owner Type',
         'State','State Regulated Dam','Federally Regulated Dam']
raw_dedup = (df_raw[EXTRA]
             .dropna(subset=['Dam Name'])
             .drop_duplicates(subset='Dam Name'))
df = df_t.merge(raw_dedup, on='Dam Name', how='left')
print(f"After join: {df.shape}")

# ── Remap 4 classes → 3 classes ───────────────────────────────────────────────
# Original: 1=Excellent, 2=Good, 3=Fair, 4=Poor/Failing
# New:      1=Good/Excellent, 2=Fair, 3=Poor/Failing
remap = {1: 1, 2: 1, 3: 2, 4: 3}
df["Condition3"] = df["Condition Assessment"].map(remap)
print(f"3-class dist: {df['Condition3'].value_counts().sort_index().to_dict()}")

# ── Encode categorical features from nation.xlsx ───────────────────────────────
eap_map  = {"Yes": 3, "Not Required": 2, "No": 1}
haz_map  = {"High": 4, "Significant": 3, "Undetermined": 2, "Low": 1}
freq_map = {"Annual": 5, "Biennial": 4, "Periodic": 3, "Not Applicable": 2, "Not Required": 1}

df["EAP_enc"]      = df["EAP Prepared"].map(eap_map).fillna(0).astype(int)
df["Hazard_enc"]   = df["Hazard Potential Classification"].map(haz_map).fillna(0).astype(int)
df["InspFreq_enc"] = df["Inspection Frequency"].map(freq_map).fillna(0).astype(int)
df["StateReg_enc"] = (df["State Regulated Dam"].astype(str).str.upper() == "YES").astype(int)
df["FedReg_enc"]   = (df["Federally Regulated Dam"].astype(str).str.upper() == "YES").astype(int)

# Target-mean encodings (computed on 3-class target)
for col, enc_name in [("State","State_enc"), ("Primary Purpose","Purpose_enc"),
                       ("Primary Owner Type","Owner_enc")]:
    mean_map = df.groupby(col)["Condition3"].mean()
    df[enc_name] = df[col].map(mean_map).fillna(df["Condition3"].mean())

df["LastInsp_date"]  = pd.to_datetime(df["Last Inspection Date"], errors="coerce")
df["DaysSinceInsp"]  = (pd.Timestamp("2024-01-01") - df["LastInsp_date"]).dt.days.fillna(-1)

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer(d):
    d = d.copy()
    d["Age"]              = 2024 - d["Year Completed"]
    d["Age_sq"]           = d["Age"] ** 2
    d["Age_bin"]          = pd.cut(d["Age"], [0,20,40,60,80,200], labels=False)
    hcols = ["Dam Height (Ft)","Hydraulic Height (Ft)","Structural Height (Ft)","NID Height (Ft)"]
    d["Height_range"]     = d["Dam Height (Ft)"] - d["Hydraulic Height (Ft)"]
    d["Height_ratio"]     = d["Hydraulic Height (Ft)"] / (d["Dam Height (Ft)"] + 1)
    d["Height_mean"]      = d[hcols].mean(axis=1)
    d["Height_spread"]    = d[hcols].max(axis=1) - d[hcols].min(axis=1)
    d["Storage_ratio"]    = d["NID Storage (Acre-Ft)"] / (d["Max Storage (Acre-Ft)"] + 1)
    d["Normal_max_ratio"] = d["Normal Storage (Acre-Ft)"] / (d["Max Storage (Acre-Ft)"] + 1)
    d["Storage_diff"]     = d["Max Storage (Acre-Ft)"] - d["Normal Storage (Acre-Ft)"]
    d["Storage_per_area"] = d["NID Storage (Acre-Ft)"] / (d["Surface Area (Acres)"] + 1)
    d["Log_NID"]          = np.log1p(d["NID Storage (Acre-Ft)"])
    d["Log_vol"]          = np.log1p(d["Volume (Cubic Yards)"])
    d["Vol_per_ht"]       = d["Volume (Cubic Yards)"] / (d["Dam Height (Ft)"] + 1)
    d["Log_Q"]            = np.log1p(d["Max Discharge (Cubic Ft/Second)"])
    d["Q_per_area"]       = d["Max Discharge (Cubic Ft/Second)"] / (d["Surface Area (Acres)"] + 1)
    d["Q_per_drain"]      = d["Max Discharge (Cubic Ft/Second)"] / (d["Drainage Area (Sq Miles)"] + 1)
    d["Q_per_ht"]         = d["Max Discharge (Cubic Ft/Second)"] / (d["Dam Height (Ft)"] + 1)
    d["Log_Q_area"]       = np.log1p(d["Q_per_area"])
    d["Spill_per_ht"]     = d["Spillway Width (Ft)"] / (d["Dam Height (Ft)"] + 1)
    d["Log_spill"]        = np.log1p(d["Spillway Width (Ft)"])
    d["Compactness"]      = d["Dam Height (Ft)"] / (d["Dam Length (Ft)"] + 1)
    d["Log_len"]          = np.log1p(d["Dam Length (Ft)"])
    d["Log_area"]         = np.log1p(d["Surface Area (Acres)"])
    d["Log_drain"]        = np.log1p(d["Drainage Area (Sq Miles)"])
    d["Age_x_ht"]         = d["Age"] * d["Dam Height (Ft)"]
    d["Age_x_vol"]        = d["Age"] * d["Log_vol"]
    d["Age_x_storage"]    = d["Age"] * d["Log_NID"]
    d["Age_x_Q"]          = d["Age"] * d["Log_Q"]
    d["Age_x_core"]       = d["Age"] * d["Core Types"]
    d["Age_x_found"]      = d["Age"] * d["Foundation"]
    d["Type_x_age"]       = d["Primary Dam Type"] * d["Age"]
    d["Core_x_found"]     = d["Core Types"] * d["Foundation"]
    d["Core_x_ht"]        = d["Core Types"] * d["Dam Height (Ft)"]
    d["Found_x_ht"]       = d["Foundation"] * d["Dam Height (Ft)"]
    d["Spill_x_Q"]        = d["Spillway Type"] * d["Log_Q"]
    n = d["Dam Name"].astype(str).str.lower()
    d["nm_len"]           = d["Dam Name"].astype(str).str.len()
    d["nm_words"]         = d["Dam Name"].astype(str).str.split().str.len()
    d["nm_detention"]     = n.str.contains("detention").astype(int)
    d["nm_scs"]           = n.str.contains("scs").astype(int)
    d["nm_has_num"]       = d["Dam Name"].astype(str).str.contains(r'\d').astype(int)
    d["Log_dist"]         = np.log1p(d["Distance to Nearest City (Miles)"])
    d["Is_urban"]         = (d["Distance to Nearest City (Miles)"] <= 2).astype(int)
    for f, s in [("Age","age"),("Hydraulic Height (Ft)","hh"),
                 ("NID Storage (Acre-Ft)","nid"),("Log_Q","lq")]:
        d[f"sq_{s}"] = d[f] ** 2
    for c, s in [("Dam Height (Ft)","ht"),("Volume (Cubic Yards)","vol"),
                 ("Max Discharge (Cubic Ft/Second)","Q"),("NID Storage (Acre-Ft)","stor"),
                 ("Spillway Width (Ft)","spill"),("Surface Area (Acres)","area")]:
        d[f"rk_{s}"] = d[c].rank(pct=True)
    # Nation.xlsx interactions
    d["EAP_x_age"]         = d["EAP_enc"]    * d["Age"]
    d["Haz_x_age"]         = d["Hazard_enc"] * d["Age"]
    d["EAP_x_haz"]         = d["EAP_enc"]    * d["Hazard_enc"]
    d["EAP_x_ht"]          = d["EAP_enc"]    * d["Dam Height (Ft)"]
    d["Haz_x_storage"]     = d["Hazard_enc"] * d["Log_NID"]
    d["State_x_haz"]       = d["State_enc"]  * d["Hazard_enc"]
    d["Insp_x_age"]        = d["InspFreq_enc"] * d["Age"]
    d["DaysSinceInsp_log"] = np.log1p(d["DaysSinceInsp"].clip(lower=0))
    return d

df = engineer(df)

DROP = ["Dam Name","Condition Assessment","Condition3","Year Completed",
        "EAP Prepared","Hazard Potential Classification","State",
        "Inspection Frequency","Primary Purpose","Primary Owner Type",
        "State Regulated Dam","Federally Regulated Dam",
        "Last Inspection Date","LastInsp_date"]
feat_cols = [c for c in df.columns if c not in DROP]
X = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
y = df["Condition3"]
print(f"\nFeatures: {len(feat_cols)}")
print(f"Class distribution:\n{y.value_counts().sort_index()}\n")

# ── Train/test split ──────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)

# ── MI feature selection ──────────────────────────────────────────────────────
print("Mutual information selection ...")
mi = mutual_info_classif(X_tr, y_tr, random_state=42)
mi_s = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
sel = mi_s[mi_s > 0].index.tolist()
print(f"  {len(sel)}/{len(feat_cols)} features kept")
X_tr_s = X_tr[sel]; X_te_s = X_te[sel]

# ── SMOTE balancing ───────────────────────────────────────────────────────────
print("SMOTE ...")
X_bal, y_bal = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr_s, y_tr)
print(f"  Balanced: {X_bal.shape}  {pd.Series(y_bal).value_counts().sort_index().to_dict()}")
Xb_tr, Xb_val, yb_tr, yb_val = train_test_split(X_bal, y_bal, test_size=0.2,
                                                   random_state=42, stratify=y_bal)

# ── Optuna: LightGBM ─────────────────────────────────────────────────────────
print("\n=== Optuna: LightGBM (25 trials) ==="); t0 = time.time()
def obj_lgb(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 50, 400),
        num_leaves        = trial.suggest_int("num_leaves", 15, 100),
        learning_rate     = trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        max_depth         = trial.suggest_int("max_depth", 3, 10),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 3, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 3, log=True),
    )
    m = lgb.LGBMClassifier(**p, class_weight="balanced",
                           random_state=42, verbose=-1, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sl = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sl.optimize(obj_lgb, n_trials=25, show_progress_bar=False)
lgb_p = {**dict(sl.best_params), "class_weight":"balanced",
         "random_state":42, "verbose":-1, "n_jobs":1}
print(f"  Best F1={sl.best_value:.4f}  ({time.time()-t0:.0f}s)")
lgb_m = lgb.LGBMClassifier(**lgb_p); lgb_m.fit(X_bal, y_bal)

# ── Optuna: CatBoost ─────────────────────────────────────────────────────────
print("\n=== Optuna: CatBoost (15 trials) ==="); t0 = time.time()
def obj_cb(trial):
    p = dict(
        iterations          = trial.suggest_int("iterations", 50, 350),
        depth               = trial.suggest_int("depth", 3, 8),
        learning_rate       = trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        random_strength     = trial.suggest_float("random_strength", 0.1, 5, log=True),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 2.0),
    )
    m = CatBoostClassifier(**p, auto_class_weights="Balanced", random_seed=42, verbose=0)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sc = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sc.optimize(obj_cb, n_trials=15, show_progress_bar=False)
cb_p = {**dict(sc.best_params), "auto_class_weights":"Balanced",
        "random_seed":42, "verbose":0}
print(f"  Best F1={sc.best_value:.4f}  ({time.time()-t0:.0f}s)")
cb_m = CatBoostClassifier(**cb_p); cb_m.fit(X_bal, y_bal, verbose=0)

# ── Optuna: ExtraTrees ───────────────────────────────────────────────────────
print("\n=== Optuna: ExtraTrees (15 trials) ==="); t0 = time.time()
def obj_et(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500),
        max_depth        = trial.suggest_int("max_depth", 5, 30),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features     = trial.suggest_float("max_features", 0.3, 1.0),
    )
    m = ExtraTreesClassifier(**p, class_weight="balanced", random_state=42, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

se = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
se.optimize(obj_et, n_trials=15, show_progress_bar=False)
et_p = {**se.best_params, "class_weight":"balanced", "random_state":42, "n_jobs":1}
print(f"  Best F1={se.best_value:.4f}  ({time.time()-t0:.0f}s)")
et_m = ExtraTreesClassifier(**et_p); et_m.fit(X_bal, y_bal)

# ── Optuna: LightGBM v2 ──────────────────────────────────────────────────────
print("\n=== Optuna: LightGBM-v2 (20 trials) ==="); t0 = time.time()
def obj_lgb2(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 600),
        num_leaves        = trial.suggest_int("num_leaves", 20, 150),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth         = trial.suggest_int("max_depth", 4, 12),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 40),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
    )
    m = lgb.LGBMClassifier(**p, class_weight="balanced",
                           random_state=0, verbose=-1, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sl2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
sl2.optimize(obj_lgb2, n_trials=20, show_progress_bar=False)
lgb_p2 = {**dict(sl2.best_params), "class_weight":"balanced",
           "random_state":0, "verbose":-1, "n_jobs":1}
print(f"  Best F1={sl2.best_value:.4f}  ({time.time()-t0:.0f}s)")
lgb_m2 = lgb.LGBMClassifier(**lgb_p2); lgb_m2.fit(X_bal, y_bal)

# ── Evaluate all + soft/weighted vote ────────────────────────────────────────
print("\n=== Evaluating on test set ===")
lgb_pred  = lgb_m.predict(X_te_s)
cb_pred   = cb_m.predict(X_te_s)
et_pred   = et_m.predict(X_te_s)
lgb2_pred = lgb_m2.predict(X_te_s)

p_lgb  = lgb_m.predict_proba(X_te_s)
p_cb   = cb_m.predict_proba(X_te_s)
p_et   = et_m.predict_proba(X_te_s)
p_lgb2 = lgb_m2.predict_proba(X_te_s)

# Soft vote — proba arrays are already 3-class; argmax + shift to 1-based
classes = lgb_m.classes_   # [1, 2, 3]
soft_proba   = (p_lgb + p_cb + p_et + p_lgb2) / 4
wtd_proba    = (2*p_lgb + p_cb + p_et + 2*p_lgb2) / 6
soft_pred    = classes[np.argmax(soft_proba, axis=1)]
wtd_pred     = classes[np.argmax(wtd_proba,  axis=1)]

def ev(p): return accuracy_score(y_te, p), f1_score(y_te, p, average="weighted")

all_res = {
    "LightGBM":   (lgb_pred,  *ev(lgb_pred)),
    "CatBoost":   (cb_pred,   *ev(cb_pred)),
    "ExtraTrees": (et_pred,   *ev(et_pred)),
    "LightGBM-2": (lgb2_pred, *ev(lgb2_pred)),
    "Soft Vote":  (soft_pred, *ev(soft_pred)),
    "Wtd Vote":   (wtd_pred,  *ev(wtd_pred)),
}

print("\n" + "="*55 + "\nFINAL RESULTS (3-class)\n" + "="*55)
best_name, best_acc, best_pred = None, 0, None
for mn, (pred, acc, f1) in all_res.items():
    tag = " ◄" if acc == max(r[1] for r in all_res.values()) else ""
    print(f"  {mn:<14}  Acc={acc:.4f}  F1={f1:.4f}{tag}")
    if acc > best_acc:
        best_acc, best_name, best_pred = acc, mn, pred

print(f"\nBest: {best_name}  Acc={best_acc:.4f} ({best_acc*100:.2f}%)")
print("\nClassification Report:")
labels = [LABEL_MAP3[i] for i in sorted(LABEL_MAP3)]
print(classification_report(y_te, best_pred, target_names=labels))

print("\nTop 15 features by MI:")
for i, (f, v) in enumerate(mi_s.head(15).items()):
    print(f"  {i+1:2d}. {f:<40} {v:.4f}")

# ── Visualizations ────────────────────────────────────────────────────────────
fi_s = pd.Series(lgb_m.feature_importances_, index=sel).sort_values(ascending=False)
PALETTE = {"Good/Excellent": "#4CAF50", "Fair": "#FF9800", "Poor/Failing": "#F44336"}
CLASS_COLORS = [PALETTE[LABEL_MAP3[i]] for i in sorted(LABEL_MAP3)]

fig = plt.figure(figsize=(22, 26))
fig.suptitle("Bridge/Dam Condition — 3-Class Classification Pipeline",
             fontsize=17, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.42)

# (a) Class distribution (3-class)
ax0 = fig.add_subplot(gs[0, 0])
counts = y.value_counts().sort_index()
clrs = [PALETTE[LABEL_MAP3[i]] for i in counts.index]
bars = ax0.bar([LABEL_MAP3[i] for i in counts.index], counts.values,
               color=clrs, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
             str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax0.set_title("3-Class Distribution\n(merged Excellent+Good)", fontsize=11, fontweight="bold")
ax0.set_ylabel("Count"); ax0.tick_params(axis="x", rotation=10)

# (b) Model comparison
ax1 = fig.add_subplot(gs[0, 1:])
names = list(all_res.keys())
accs  = [all_res[n][1] for n in names]
f1s   = [all_res[n][2] for n in names]
xp = np.arange(len(names)); w = 0.36
b1 = ax1.bar(xp-w/2, accs, w, label="Accuracy",    color="#5C6BC0", alpha=0.87)
b2 = ax1.bar(xp+w/2, f1s,  w, label="F1 Weighted", color="#26A69A", alpha=0.87)
ax1.set_xticks(xp); ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylim(0, 1.08)
ax1.axhline(0.95, color="red", ls="--", lw=1.8, label="95% target")
ax1.set_title("Model Comparison — 3-Class (Optuna-tuned)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Score"); ax1.legend(fontsize=9)
for b in [*b1, *b2]:
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
             f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

# (c) Optuna convergence
ax2 = fig.add_subplot(gs[1, 2])
for study, lbl, col in [(sl,"LightGBM","#5C6BC0"), (sc,"CatBoost","#66BB6A"),
                         (se,"ExtraTrees","#FFA726"), (sl2,"LightGBM-2","#AB47BC")]:
    v = [t.value for t in study.trials if t.value is not None]
    if v: ax2.plot(np.maximum.accumulate(v), lw=2, label=lbl, color=col)
ax2.set_title("Optuna Convergence", fontsize=12, fontweight="bold")
ax2.set_xlabel("Trial"); ax2.set_ylabel("Best Val F1"); ax2.legend(fontsize=8)

# (d) Confusion matrix
ax3 = fig.add_subplot(gs[1, :2])
cm_arr = confusion_matrix(y_te, best_pred, labels=sorted(LABEL_MAP3))
cm_pct = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax3, linewidths=0.5, cbar_kws={"label": "%"})
for i in range(3):
    for j in range(3):
        ax3.text(j+0.5, i+0.72, f"n={cm_arr[i,j]}", ha="center", fontsize=7, color="gray")
ax3.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")

# (e) Top-20 LGB feature importance
ax4 = fig.add_subplot(gs[2, :2])
top20 = fi_s.head(20)
ax4.barh(range(20), top20.values[::-1], color="#7E57C2", alpha=0.85)
ax4.set_yticks(range(20)); ax4.set_yticklabels(top20.index[::-1], fontsize=8)
ax4.set_title("Top 20 Features by LGB Importance", fontsize=12, fontweight="bold")
ax4.set_xlabel("Importance")

# (f) Top MI features
ax5 = fig.add_subplot(gs[2, 2])
top_mi = mi_s.head(20)
ax5.barh(range(20), top_mi.values[::-1], color="#42A5F5", alpha=0.85)
ax5.set_yticks(range(20)); ax5.set_yticklabels(top_mi.index[::-1], fontsize=7)
ax5.set_title("Top 20 by Mutual Info", fontsize=11, fontweight="bold")
ax5.set_xlabel("MI Score")

# (g) Per-class P/R/F1
ax6 = fig.add_subplot(gs[3, :2])
prec, rec, f1pc, supp = precision_recall_fscore_support(y_te, best_pred,
                                                          labels=sorted(LABEL_MAP3))
xp2 = np.arange(3); w2 = 0.25
lnames = [f"{LABEL_MAP3[i]}\n(n={s})" for i, s in zip(sorted(LABEL_MAP3), supp)]
for off, vals, lbl, col in [(-w2, prec, "Precision", "#5C6BC0"),
                              (0,  rec,  "Recall",    "#26A69A"),
                              (w2, f1pc, "F1",        "#FFA726")]:
    bars2 = ax6.bar(xp2+off, vals, w2, label=lbl, color=col, alpha=0.85)
    for b, v in zip(bars2, vals):
        ax6.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}",
                 ha="center", fontsize=8)
ax6.set_xticks(xp2); ax6.set_xticklabels(lnames, fontsize=10)
ax6.set_ylim(0, 1.15)
ax6.set_title(f"Per-Class Metrics — {best_name}", fontsize=12, fontweight="bold")
ax6.legend(fontsize=9); ax6.set_ylabel("Score")

# (h) Confidence histogram
ax7 = fig.add_subplot(gs[3, 2])
avg_p = soft_proba; maxp = avg_p.max(axis=1)
ok = (soft_pred == y_te.values)
ax7.hist(maxp[ok],  bins=25, alpha=0.65, color="#4CAF50",
         label=f"Correct ({ok.sum()})")
ax7.hist(maxp[~ok], bins=25, alpha=0.65, color="#F44336",
         label=f"Wrong ({(~ok).sum()})")
ax7.set_title("Soft-Vote Confidence", fontsize=11, fontweight="bold")
ax7.set_xlabel("Max probability"); ax7.set_ylabel("Count"); ax7.legend(fontsize=8)

OUT = "/home/user/MLTUNNELS/bridge_3class_report.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nReport saved → {OUT}")

print("\n" + "="*55)
print(f"Classes       : 3  (Good/Excellent | Fair | Poor/Failing)")
print(f"Features      : {len(feat_cols)} engineered → {len(sel)} MI-selected")
print(f"Optuna trials : LGB=25  CB=15  ET=15  LGB2=20")
print(f"Best model    : {best_name}  Acc={best_acc:.4f}  ({best_acc*100:.2f}%)")
print("=" * 55)
