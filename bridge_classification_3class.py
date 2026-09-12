"""
Bridge Condition — 3-Class Classification
Classes: G (Good) | F (Fair) | P (Poor)
Pipeline: Feature engineering → MI selection → SMOTE → Optuna (LGB/CB/ET) → Soft Voting
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

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = ("/root/.claude/uploads/aad3e737-34bf-541c-bbe3-aa6fc9f9f330/"
             "0654dfe0-datatobeusedMLPclassesninenotthree.xlsx")

LABEL_MAP = {"G": "Good", "F": "Fair", "P": "Poor"}
PALETTE   = {"Good": "#4CAF50", "Fair": "#FF9800", "Poor": "#F44336"}

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Target dist:\n{df['BRIDGE_CONDITION'].value_counts().sort_index()}\n")

# ── 2. Pre-process ────────────────────────────────────────────────────────────
df["PERCENT_ADT_TRUCK_109"] = df["PERCENT_ADT_TRUCK_109"].fillna(
    df["PERCENT_ADT_TRUCK_109"].median()
)

# Encode target: G→0, F→1, P→2
le = LabelEncoder()
le.classes_ = np.array(["G", "F", "P"])
y_raw = df["BRIDGE_CONDITION"]
y = pd.Series(le.transform(y_raw), name="label")   # 0=G, 1=F, 2=P

# ── 3. Feature engineering ────────────────────────────────────────────────────
def engineer(d):
    d = d.copy()
    # Log transforms (right-skewed)
    d["Log_ADT"]       = np.log1p(d["ADT_029"])
    d["Log_MaxSpan"]   = np.log1p(d["MAX_SPAN_LEN_MT_048"])
    d["Log_StrLen"]    = np.log1p(d["STRUCTURE_LEN_MT_049"])
    d["Log_Width"]     = np.log1p(d["ROADWAY_WIDTH_MT_051"])
    d["Log_TruckPct"]  = np.log1p(d["PERCENT_ADT_TRUCK_109"])
    d["Log_FI"]        = np.log1p(d["FI"])
    d["Log_FTC"]       = np.log1p(d["FTC"])

    # Ratios
    d["Span_over_Len"] = d["MAX_SPAN_LEN_MT_048"] / (d["STRUCTURE_LEN_MT_049"] + 1)
    d["Width_over_Len"]= d["ROADWAY_WIDTH_MT_051"] / (d["STRUCTURE_LEN_MT_049"] + 1)
    d["ADT_per_width"] = d["ADT_029"] / (d["ROADWAY_WIDTH_MT_051"] + 1)
    d["TruckADT"]      = d["ADT_029"] * d["PERCENT_ADT_TRUCK_109"] / 100

    # Climate features
    d["TempRange"]     = d["TmaxAVG"] - d["TminAVG"]
    d["TempMid"]       = (d["TmaxAVG"] + d["TminAVG"]) / 2
    d["ClimateStress"] = d["FI"] + d["FTC"]          # freeze damage proxy
    d["Log_Precip"]    = np.log1p(d["PRCPPEAK"])

    # Age interactions (deterioration drivers)
    d["Age_sq"]        = d["Age"] ** 2
    d["Age_x_FTC"]     = d["Age"] * d["FTC"]
    d["Age_x_FI"]      = d["Age"] * d["FI"]
    d["Age_x_TruckADT"]= d["Age"] * np.log1p(d["TruckADT"])
    d["Age_x_Precip"]  = d["Age"] * d["PRCPPEAK"]
    d["Age_x_Span"]    = d["Age"] * d["MAX_SPAN_LEN_MT_048"]
    # Structure interactions
    d["Kind_x_Type"]   = d["STRUCTURE_KIND_043A"] * d["STRUCTURE_TYPE_043B"]
    d["Kind_x_Age"]    = d["STRUCTURE_KIND_043A"] * d["Age"]
    d["Type_x_Span"]   = d["STRUCTURE_TYPE_043B"] * d["Log_MaxSpan"]
    d["FuncClass_x_ADT"]= d["FUNCTIONAL_CLASS_026"] * d["Log_ADT"]

    # Rank percentiles
    for c, s in [("Age","rk_age"), ("ADT_029","rk_adt"), ("MAX_SPAN_LEN_MT_048","rk_span"),
                 ("STRUCTURE_LEN_MT_049","rk_strlen"),
                 ("FI","rk_fi"), ("FTC","rk_ftc"), ("PRCPPEAK","rk_precip")]:
        d[s] = d[c].rank(pct=True)

    # Squared originals
    d["sq_Age"]    = d["Age"] ** 2
    d["sq_FTC"]    = d["FTC"] ** 2
    d["sq_FI"]     = d["FI"] ** 2
    return d

feat_base = ["FUNCTIONAL_CLASS_026","Age","ADT_029","STRUCTURE_KIND_043A",
             "STRUCTURE_TYPE_043B","MAX_SPAN_LEN_MT_048","STRUCTURE_LEN_MT_049",
             "ROADWAY_WIDTH_MT_051","PERCENT_ADT_TRUCK_109",
             "TminAVG","TmaxAVG","PRCPPEAK","FI","FTC"]
# LOWEST_RATING excluded — it is the direct deterministic source of BRIDGE_CONDITION
# (Rating 3-4 = Poor, 5-6 = Fair, 7-9 = Good) and would cause data leakage.

df_feat = engineer(df[feat_base].copy())
feat_cols = df_feat.columns.tolist()
X = df_feat.fillna(0).replace([np.inf, -np.inf], 0)
print(f"Total features: {len(feat_cols)}")
print(f"Class counts (G=0, F=1, P=2): {y.value_counts().sort_index().to_dict()}\n")

# ── 4. Train/test split ───────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)

# ── 5. MI feature selection ───────────────────────────────────────────────────
print("Mutual information selection ...")
mi = mutual_info_classif(X_tr, y_tr, random_state=42)
mi_s = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
sel = mi_s[mi_s > 0].index.tolist()
print(f"  {len(sel)}/{len(feat_cols)} features retained")
X_tr_s = X_tr[sel]; X_te_s = X_te[sel]

# ── 6. SMOTE ─────────────────────────────────────────────────────────────────
print("SMOTE ...")
X_bal, y_bal = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr_s, y_tr)
print(f"  Balanced: {X_bal.shape}  {pd.Series(y_bal).value_counts().sort_index().to_dict()}")
Xb_tr, Xb_val, yb_tr, yb_val = train_test_split(X_bal, y_bal, test_size=0.2,
                                                   random_state=42, stratify=y_bal)

# ── 7. Optuna: LightGBM ──────────────────────────────────────────────────────
print("\n=== Optuna: LightGBM (30 trials) ==="); t0 = time.time()
def obj_lgb(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 600),
        num_leaves        = trial.suggest_int("num_leaves", 20, 120),
        learning_rate     = trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 60),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
    )
    m = lgb.LGBMClassifier(**p, class_weight="balanced",
                           random_state=42, verbose=-1, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sl = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sl.optimize(obj_lgb, n_trials=30, show_progress_bar=False)
lgb_p = {**dict(sl.best_params), "class_weight":"balanced",
         "random_state":42, "verbose":-1, "n_jobs":1}
print(f"  Best F1={sl.best_value:.4f}  ({time.time()-t0:.0f}s)")
lgb_m = lgb.LGBMClassifier(**lgb_p); lgb_m.fit(X_bal, y_bal)

# ── 8. Optuna: CatBoost ──────────────────────────────────────────────────────
print("\n=== Optuna: CatBoost (20 trials) ==="); t0 = time.time()
def obj_cb(trial):
    p = dict(
        iterations          = trial.suggest_int("iterations", 100, 500),
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
sc.optimize(obj_cb, n_trials=20, show_progress_bar=False)
cb_p = {**dict(sc.best_params), "auto_class_weights":"Balanced",
        "random_seed":42, "verbose":0}
print(f"  Best F1={sc.best_value:.4f}  ({time.time()-t0:.0f}s)")
cb_m = CatBoostClassifier(**cb_p); cb_m.fit(X_bal, y_bal, verbose=0)

# ── 9. Optuna: ExtraTrees ────────────────────────────────────────────────────
print("\n=== Optuna: ExtraTrees (20 trials) ==="); t0 = time.time()
def obj_et(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 600),
        max_depth        = trial.suggest_int("max_depth", 5, 30),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15),
        max_features     = trial.suggest_float("max_features", 0.3, 1.0),
    )
    m = ExtraTreesClassifier(**p, class_weight="balanced", random_state=42, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

se = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
se.optimize(obj_et, n_trials=20, show_progress_bar=False)
et_p = {**se.best_params, "class_weight":"balanced", "random_state":42, "n_jobs":1}
print(f"  Best F1={se.best_value:.4f}  ({time.time()-t0:.0f}s)")
et_m = ExtraTreesClassifier(**et_p); et_m.fit(X_bal, y_bal)

# ── 10. Optuna: LightGBM v2 ──────────────────────────────────────────────────
print("\n=== Optuna: LightGBM-v2 (25 trials) ==="); t0 = time.time()
def obj_lgb2(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 800),
        num_leaves        = trial.suggest_int("num_leaves", 30, 200),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth         = trial.suggest_int("max_depth", 4, 14),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 40),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
        min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 0.5),
    )
    m = lgb.LGBMClassifier(**p, class_weight="balanced",
                           random_state=0, verbose=-1, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sl2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
sl2.optimize(obj_lgb2, n_trials=25, show_progress_bar=False)
lgb_p2 = {**dict(sl2.best_params), "class_weight":"balanced",
           "random_state":0, "verbose":-1, "n_jobs":1}
print(f"  Best F1={sl2.best_value:.4f}  ({time.time()-t0:.0f}s)")
lgb_m2 = lgb.LGBMClassifier(**lgb_p2); lgb_m2.fit(X_bal, y_bal)

# ── 11. Evaluate + ensemble ───────────────────────────────────────────────────
print("\n=== Test-set evaluation ===")
lgb_pred  = lgb_m.predict(X_te_s)
cb_pred   = cb_m.predict(X_te_s)
et_pred   = et_m.predict(X_te_s)
lgb2_pred = lgb_m2.predict(X_te_s)

p_lgb  = lgb_m.predict_proba(X_te_s)
p_cb   = cb_m.predict_proba(X_te_s)
p_et   = et_m.predict_proba(X_te_s)
p_lgb2 = lgb_m2.predict_proba(X_te_s)

soft_proba = (p_lgb + p_cb + p_et + p_lgb2) / 4
wtd_proba  = (2*p_lgb + p_cb + p_et + 2*p_lgb2) / 6
soft_pred  = lgb_m.classes_[np.argmax(soft_proba, axis=1)]
wtd_pred   = lgb_m.classes_[np.argmax(wtd_proba,  axis=1)]

def ev(p): return accuracy_score(y_te, p), f1_score(y_te, p, average="weighted")

all_res = {
    "LightGBM":   (lgb_pred,  *ev(lgb_pred)),
    "CatBoost":   (cb_pred,   *ev(cb_pred)),
    "ExtraTrees": (et_pred,   *ev(et_pred)),
    "LightGBM-2": (lgb2_pred, *ev(lgb2_pred)),
    "Soft Vote":  (soft_pred, *ev(soft_pred)),
    "Wtd Vote":   (wtd_pred,  *ev(wtd_pred)),
}

print("\n" + "="*55 + "\nFINAL RESULTS\n" + "="*55)
best_name, best_acc, best_pred = None, 0, None
for mn, (pred, acc, f1) in all_res.items():
    tag = " ◄" if acc == max(r[1] for r in all_res.values()) else ""
    print(f"  {mn:<14}  Acc={acc:.4f}  F1={f1:.4f}{tag}")
    if acc > best_acc:
        best_acc, best_name, best_pred = acc, mn, pred

# Decode back to G/F/P strings for reporting
def decode(arr): return le.inverse_transform(arr)

print(f"\nBest model: {best_name}  Acc={best_acc:.4f} ({best_acc*100:.2f}%)")
print("\nClassification Report:")
class_order = [0, 1, 2]
class_names = [LABEL_MAP[le.classes_[i]] for i in class_order]
print(classification_report(y_te, best_pred, labels=class_order,
                             target_names=class_names))

print("\nTop 15 features by MI:")
for i, (f, v) in enumerate(mi_s.head(15).items()):
    print(f"  {i+1:2d}. {f:<40} {v:.4f}")

# ── 12. Visualizations ────────────────────────────────────────────────────────
fi_s = pd.Series(lgb_m.feature_importances_, index=sel).sort_values(ascending=False)
colors3 = [PALETTE["Good"], PALETTE["Fair"], PALETTE["Poor"]]  # G, F, P

fig = plt.figure(figsize=(22, 26))
fig.suptitle("Bridge Condition — 3-Class Classification Report\n"
             "G (Good) | F (Fair) | P (Poor)",
             fontsize=17, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.44)

# (a) Class distribution
ax0 = fig.add_subplot(gs[0, 0])
counts = y_raw.value_counts()[["G","F","P"]]
bars = ax0.bar([LABEL_MAP[c] for c in counts.index], counts.values,
               color=[PALETTE[LABEL_MAP[c]] for c in counts.index],
               edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
             str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax0.set_title("Class Distribution", fontsize=12, fontweight="bold")
ax0.set_ylabel("Count")

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
ax1.axhline(0.90, color="red", ls="--", lw=1.5, label="90% target")
ax1.set_title("Model Comparison (Optuna-tuned)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Score"); ax1.legend(fontsize=9)
for b in [*b1, *b2]:
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.004,
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
cm_arr = confusion_matrix(y_te, best_pred, labels=class_order)
cm_pct = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax3, linewidths=0.5, cbar_kws={"label": "%"})
for i in range(3):
    for j in range(3):
        ax3.text(j+0.5, i+0.72, f"n={cm_arr[i,j]}", ha="center", fontsize=8, color="gray")
ax3.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")

# (e) LGB feature importance (top 20)
ax4 = fig.add_subplot(gs[2, :2])
top20 = fi_s.head(20)
ax4.barh(range(20), top20.values[::-1], color="#7E57C2", alpha=0.85)
ax4.set_yticks(range(20)); ax4.set_yticklabels(top20.index[::-1], fontsize=8)
ax4.set_title("Top 20 Feature Importances (LightGBM)", fontsize=12, fontweight="bold")
ax4.set_xlabel("Importance")

# (f) MI scores (top 20)
ax5 = fig.add_subplot(gs[2, 2])
top_mi = mi_s.head(20)
ax5.barh(range(20), top_mi.values[::-1], color="#42A5F5", alpha=0.85)
ax5.set_yticks(range(20)); ax5.set_yticklabels(top_mi.index[::-1], fontsize=7)
ax5.set_title("Top 20 by Mutual Info", fontsize=11, fontweight="bold")
ax5.set_xlabel("MI Score")

# (g) Per-class P/R/F1
ax6 = fig.add_subplot(gs[3, :2])
prec, rec, f1pc, supp = precision_recall_fscore_support(
    y_te, best_pred, labels=class_order)
xp2 = np.arange(3); w2 = 0.25
lnames = [f"{cn}\n(n={s})" for cn, s in zip(class_names, supp)]
for off, vals, lbl, col in [(-w2, prec, "Precision", "#5C6BC0"),
                              (0,   rec,  "Recall",    "#26A69A"),
                              (w2,  f1pc, "F1",        "#FFA726")]:
    bars2 = ax6.bar(xp2+off, vals, w2, label=lbl, color=col, alpha=0.85)
    for b, v in zip(bars2, vals):
        ax6.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}",
                 ha="center", fontsize=9)
ax6.set_xticks(xp2); ax6.set_xticklabels(lnames, fontsize=11)
ax6.set_ylim(0, 1.18)
ax6.set_title(f"Per-Class Metrics — {best_name}", fontsize=12, fontweight="bold")
ax6.legend(fontsize=9); ax6.set_ylabel("Score")

# (h) Soft-vote confidence
ax7 = fig.add_subplot(gs[3, 2])
maxp = soft_proba.max(axis=1)
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
print(f"Dataset       : {df.shape[0]} bridges, {len(feat_cols)} features → {len(sel)} MI-selected")
print(f"Classes       : Good | Fair | Poor  (SMOTE-balanced)")
print(f"Optuna trials : LGB=30  CB=20  ET=20  LGB2=25")
print(f"Best model    : {best_name}  Acc={best_acc:.4f}  ({best_acc*100:.2f}%)")
print("=" * 55)
