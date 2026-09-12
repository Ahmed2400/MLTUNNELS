"""
Bridge Condition — Strong 3-Class Model
  • 10-Fold Stratified CV + Optuna (LGB/CB/XGB/ET) + Stacking
  • Full honest test accuracy reported
  • Artificial accuracy curve: iteratively remove misclassified samples
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

from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     cross_val_score, cross_val_predict)
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = ("/root/.claude/uploads/aad3e737-34bf-541c-bbe3-aa6fc9f9f330/"
             "0654dfe0-datatobeusedMLPclassesninenotthree.xlsx")

LABEL_MAP = {0: "Good", 1: "Fair", 2: "Poor"}
PALETTE   = {"Good": "#4CAF50", "Fair": "#FF9800", "Poor": "#F44336"}

# ─── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH)
df["PERCENT_ADT_TRUCK_109"] = df["PERCENT_ADT_TRUCK_109"].fillna(
    df["PERCENT_ADT_TRUCK_109"].median())

le = LabelEncoder()
le.classes_ = np.array(["G", "F", "P"])
y = pd.Series(le.transform(df["BRIDGE_CONDITION"]), name="label")
# LOWEST_RATING excluded — it is the deterministic source of BRIDGE_CONDITION
feat_base = ["FUNCTIONAL_CLASS_026","Age","ADT_029","STRUCTURE_KIND_043A",
             "STRUCTURE_TYPE_043B","MAX_SPAN_LEN_MT_048","STRUCTURE_LEN_MT_049",
             "ROADWAY_WIDTH_MT_051","PERCENT_ADT_TRUCK_109",
             "TminAVG","TmaxAVG","PRCPPEAK","FI","FTC"]

print(f"Samples: {len(df)}  |  Class dist: {y.value_counts().sort_index().to_dict()}")

# ─── 2. Feature engineering ───────────────────────────────────────────────────
def engineer(d):
    d = d.copy()
    d["Log_ADT"]        = np.log1p(d["ADT_029"])
    d["Log_MaxSpan"]    = np.log1p(d["MAX_SPAN_LEN_MT_048"])
    d["Log_StrLen"]     = np.log1p(d["STRUCTURE_LEN_MT_049"])
    d["Log_Width"]      = np.log1p(d["ROADWAY_WIDTH_MT_051"])
    d["Log_TruckPct"]   = np.log1p(d["PERCENT_ADT_TRUCK_109"])
    d["Log_FI"]         = np.log1p(d["FI"])
    d["Log_FTC"]        = np.log1p(d["FTC"])
    d["Log_Precip"]     = np.log1p(d["PRCPPEAK"])
    d["Span_over_Len"]  = d["MAX_SPAN_LEN_MT_048"] / (d["STRUCTURE_LEN_MT_049"] + 1)
    d["Width_over_Len"] = d["ROADWAY_WIDTH_MT_051"] / (d["STRUCTURE_LEN_MT_049"] + 1)
    d["ADT_per_width"]  = d["ADT_029"] / (d["ROADWAY_WIDTH_MT_051"] + 1)
    d["TruckADT"]       = d["ADT_029"] * d["PERCENT_ADT_TRUCK_109"] / 100
    d["Log_TruckADT"]   = np.log1p(d["TruckADT"])
    d["TempRange"]      = d["TmaxAVG"] - d["TminAVG"]
    d["TempMid"]        = (d["TmaxAVG"] + d["TminAVG"]) / 2
    d["ClimateStress"]  = d["FI"] + d["FTC"]
    d["FI_x_FTC"]       = d["FI"] * d["FTC"]
    d["Age_sq"]         = d["Age"] ** 2
    d["Age_cb"]         = d["Age"] ** 3
    d["Age_x_FTC"]      = d["Age"] * d["FTC"]
    d["Age_x_FI"]       = d["Age"] * d["FI"]
    d["Age_x_Climate"]  = d["Age"] * d["ClimateStress"]
    d["Age_x_TruckADT"] = d["Age"] * d["Log_TruckADT"]
    d["Age_x_Precip"]   = d["Age"] * d["PRCPPEAK"]
    d["Age_x_Span"]     = d["Age"] * d["MAX_SPAN_LEN_MT_048"]
    d["Age_x_Kind"]     = d["Age"] * d["STRUCTURE_KIND_043A"]
    d["Age_x_Type"]     = d["Age"] * d["STRUCTURE_TYPE_043B"]
    d["Age_x_Width"]    = d["Age"] * d["ROADWAY_WIDTH_MT_051"]
    d["Age_x_FuncClass"]= d["Age"] * d["FUNCTIONAL_CLASS_026"]
    d["Kind_x_Type"]    = d["STRUCTURE_KIND_043A"] * d["STRUCTURE_TYPE_043B"]
    d["Kind_x_Span"]    = d["STRUCTURE_KIND_043A"] * d["Log_MaxSpan"]
    d["Type_x_Span"]    = d["STRUCTURE_TYPE_043B"] * d["Log_MaxSpan"]
    d["FuncClass_x_ADT"]= d["FUNCTIONAL_CLASS_026"] * d["Log_ADT"]
    d["Precip_x_FTC"]   = d["PRCPPEAK"] * d["FTC"]
    d["Precip_x_FI"]    = d["PRCPPEAK"] * d["FI"]
    d["Truck_x_Span"]   = d["Log_TruckADT"] * d["Log_MaxSpan"]
    d["Width_x_ADT"]    = d["ROADWAY_WIDTH_MT_051"] * d["Log_ADT"]
    for c, s in [("Age","rk_age"),("ADT_029","rk_adt"),("MAX_SPAN_LEN_MT_048","rk_span"),
                 ("STRUCTURE_LEN_MT_049","rk_strlen"),("ROADWAY_WIDTH_MT_051","rk_width"),
                 ("FI","rk_fi"),("FTC","rk_ftc"),("PRCPPEAK","rk_precip"),
                 ("TruckADT","rk_truck"),("TempRange","rk_temprange")]:
        d[s] = d[c].rank(pct=True)
    d["sq_FTC"]   = d["FTC"] ** 2
    d["sq_FI"]    = d["FI"] ** 2
    d["sq_ADT"]   = d["Log_ADT"] ** 2
    d["sq_Width"] = d["ROADWAY_WIDTH_MT_051"] ** 2
    return d

df_feat  = engineer(df[feat_base].copy())
feat_cols = df_feat.columns.tolist()
X = df_feat.fillna(0).replace([np.inf, -np.inf], 0)
print(f"Features: {len(feat_cols)}")

# ─── 3. Train / test split ────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)

# ─── 4. MI selection ──────────────────────────────────────────────────────────
print("\nMI selection ...")
mi = mutual_info_classif(X_tr, y_tr, random_state=42)
mi_s = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
sel  = mi_s[mi_s > 0].index.tolist()
print(f"  {len(sel)}/{len(feat_cols)} features kept")
X_tr_s = X_tr[sel]; X_te_s = X_te[sel]

# ─── 5. SMOTE ─────────────────────────────────────────────────────────────────
print("SMOTE ...")
X_bal, y_bal = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr_s, y_tr)
print(f"  Balanced: {X_bal.shape}  {pd.Series(y_bal).value_counts().sort_index().to_dict()}")
Xb_tr, Xb_val, yb_tr, yb_val = train_test_split(X_bal, y_bal, test_size=0.15,
                                                   random_state=42, stratify=y_bal)

# 10-fold CV helper on full balanced set
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ─── 6. Optuna: LightGBM ─────────────────────────────────────────────────────
print("\n=== Optuna: LightGBM (50 trials) ==="); t0 = time.time()
def obj_lgb(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 800),
        num_leaves        = trial.suggest_int("num_leaves", 20, 200),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth         = trial.suggest_int("max_depth", 3, 14),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 60),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 0.5),
    )
    m = lgb.LGBMClassifier(**p, class_weight="balanced",
                           random_state=42, verbose=-1, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="macro")   # macro = care about Poor

sl = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sl.optimize(obj_lgb, n_trials=50, show_progress_bar=False)
lgb_p = {**dict(sl.best_params), "class_weight":"balanced",
         "random_state":42, "verbose":-1, "n_jobs":1}
print(f"  Best macro-F1={sl.best_value:.4f}  ({time.time()-t0:.0f}s)")
lgb_m = lgb.LGBMClassifier(**lgb_p); lgb_m.fit(X_bal, y_bal)

# 10-fold CV score
cv_lgb = cross_val_score(lgb.LGBMClassifier(**lgb_p), X_bal, y_bal,
                          cv=cv10, scoring="accuracy", n_jobs=-1)
print(f"  10-Fold CV Acc = {cv_lgb.mean():.4f} ± {cv_lgb.std():.4f}")

# ─── 7. Optuna: CatBoost ─────────────────────────────────────────────────────
print("\n=== Optuna: CatBoost (35 trials) ==="); t0 = time.time()
def obj_cb(trial):
    p = dict(
        iterations          = trial.suggest_int("iterations", 100, 600),
        depth               = trial.suggest_int("depth", 3, 10),
        learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1e-2, 15, log=True),
        random_strength     = trial.suggest_float("random_strength", 0.1, 8, log=True),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 2.5),
        border_count        = trial.suggest_int("border_count", 32, 255),
    )
    m = CatBoostClassifier(**p, auto_class_weights="Balanced", random_seed=42, verbose=0)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="macro")

sc = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sc.optimize(obj_cb, n_trials=35, show_progress_bar=False)
cb_p = {**dict(sc.best_params), "auto_class_weights":"Balanced",
        "random_seed":42, "verbose":0}
print(f"  Best macro-F1={sc.best_value:.4f}  ({time.time()-t0:.0f}s)")
cb_m = CatBoostClassifier(**cb_p); cb_m.fit(X_bal, y_bal, verbose=0)
cv_cb = cross_val_score(CatBoostClassifier(**cb_p), X_bal, y_bal,
                         cv=cv10, scoring="accuracy", n_jobs=-1)
print(f"  10-Fold CV Acc = {cv_cb.mean():.4f} ± {cv_cb.std():.4f}")

# ─── 8. Optuna: XGBoost ──────────────────────────────────────────────────────
print("\n=== Optuna: XGBoost (35 trials) ==="); t0 = time.time()
def obj_xgb(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 700),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
        gamma             = trial.suggest_float("gamma", 0, 5),
    )
    m = xgb.XGBClassifier(**p, use_label_encoder=False, eval_metric="mlogloss",
                           random_state=42, verbosity=0, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="macro")

sx = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sx.optimize(obj_xgb, n_trials=35, show_progress_bar=False)
xgb_p = {**dict(sx.best_params), "use_label_encoder":False, "eval_metric":"mlogloss",
          "random_state":42, "verbosity":0, "n_jobs":1}
print(f"  Best macro-F1={sx.best_value:.4f}  ({time.time()-t0:.0f}s)")
xgb_m = xgb.XGBClassifier(**xgb_p); xgb_m.fit(X_bal, y_bal)
cv_xgb = cross_val_score(xgb.XGBClassifier(**xgb_p), X_bal, y_bal,
                          cv=cv10, scoring="accuracy", n_jobs=-1)
print(f"  10-Fold CV Acc = {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")

# ─── 9. Optuna: ExtraTrees ───────────────────────────────────────────────────
print("\n=== Optuna: ExtraTrees (30 trials) ==="); t0 = time.time()
def obj_et(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 800),
        max_depth        = trial.suggest_int("max_depth", 5, 35),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15),
        max_features     = trial.suggest_float("max_features", 0.2, 1.0),
        min_samples_split= trial.suggest_int("min_samples_split", 2, 20),
    )
    m = ExtraTreesClassifier(**p, class_weight="balanced", random_state=42, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="macro")

se = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
se.optimize(obj_et, n_trials=30, show_progress_bar=False)
et_p = {**se.best_params, "class_weight":"balanced", "random_state":42, "n_jobs":1}
print(f"  Best macro-F1={se.best_value:.4f}  ({time.time()-t0:.0f}s)")
et_m = ExtraTreesClassifier(**et_p); et_m.fit(X_bal, y_bal)
cv_et = cross_val_score(ExtraTreesClassifier(**et_p), X_bal, y_bal,
                         cv=cv10, scoring="accuracy", n_jobs=-1)
print(f"  10-Fold CV Acc = {cv_et.mean():.4f} ± {cv_et.std():.4f}")

# ─── 10. Stacking ensemble ────────────────────────────────────────────────────
print("\n=== Stacking ensemble ==="); t0 = time.time()
estimators = [
    ("lgb",  lgb.LGBMClassifier(**lgb_p)),
    ("cb",   CatBoostClassifier(**cb_p)),
    ("xgb",  xgb.XGBClassifier(**xgb_p)),
    ("et",   ExtraTreesClassifier(**et_p)),
]
stack_m = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    cv=5, passthrough=False, n_jobs=-1
)
stack_m.fit(X_bal, y_bal)
cv_stack = cross_val_score(stack_m, X_bal, y_bal,
                            cv=cv10, scoring="accuracy", n_jobs=-1)
print(f"  10-Fold CV Acc = {cv_stack.mean():.4f} ± {cv_stack.std():.4f}  ({time.time()-t0:.0f}s)")

# ─── 11. Full test evaluation ─────────────────────────────────────────────────
print("\n=== Full test evaluation ===")
p_lgb   = lgb_m.predict_proba(X_te_s)
p_cb    = cb_m.predict_proba(X_te_s)
p_xgb   = xgb_m.predict_proba(X_te_s)
p_et    = et_m.predict_proba(X_te_s)
p_stack = stack_m.predict_proba(X_te_s)

soft_proba = (p_lgb + p_cb + p_xgb + p_et) / 4
wtd_proba  = (2*p_lgb + 2*p_cb + p_xgb + p_et + 2*p_stack) / 8

classes = lgb_m.classes_

def make_pred(proba): return classes[np.argmax(proba, axis=1)]
def ev(p): return accuracy_score(y_te, p), f1_score(y_te, p, average="weighted"), f1_score(y_te, p, average="macro")

preds = {
    "LightGBM":   (make_pred(p_lgb),   *ev(make_pred(p_lgb))),
    "CatBoost":   (make_pred(p_cb),    *ev(make_pred(p_cb))),
    "XGBoost":    (make_pred(p_xgb),   *ev(make_pred(p_xgb))),
    "ExtraTrees": (make_pred(p_et),    *ev(make_pred(p_et))),
    "Stacking":   (make_pred(p_stack), *ev(make_pred(p_stack))),
    "Soft Vote":  (make_pred(soft_proba), *ev(make_pred(soft_proba))),
    "Wtd Vote":   (make_pred(wtd_proba),  *ev(make_pred(wtd_proba))),
}
cv_scores = {
    "LightGBM": cv_lgb, "CatBoost": cv_cb, "XGBoost": cv_xgb,
    "ExtraTrees": cv_et, "Stacking": cv_stack,
    "Soft Vote": cv_et, "Wtd Vote": cv_et,  # placeholder
}

print("\n" + "="*65)
print(f"{'Model':<14} {'CV Acc':>8} {'Test Acc':>9} {'Wtd F1':>8} {'Macro F1':>9}")
print("="*65)
best_name, best_acc, best_pred, best_proba = None, 0, None, None
for mn, (pred, acc, wf1, mf1) in preds.items():
    cv_m = cv_scores.get(mn, np.array([0])).mean()
    cv_label = f"{cv_m:.4f}" if mn not in ("Soft Vote","Wtd Vote") else "  --  "
    tag = " ◄" if acc == max(r[1] for r in preds.values()) else ""
    print(f"  {mn:<14} {cv_label:>8}  {acc:>8.4f}  {wf1:>8.4f}  {mf1:>8.4f}{tag}")
    if acc > best_acc:
        best_acc, best_name, best_pred = acc, mn, pred
        best_proba = wtd_proba if mn == "Wtd Vote" else (
                     p_stack if mn == "Stacking" else
                     p_lgb if mn == "LightGBM" else
                     p_cb  if mn == "CatBoost"  else
                     p_xgb if mn == "XGBoost"   else p_et)

print("="*65)
print(f"\nBest: {best_name}  Test Acc = {best_acc:.4f} ({best_acc*100:.2f}%)")
print("\nClassification Report (full test set):")
class_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
print(classification_report(y_te, best_pred, labels=[0,1,2], target_names=class_names))

# ─── 12. Artificial accuracy: remove failures progressively ───────────────────
print("\n" + "="*65)
print("ARTIFICIAL ACCURACY — removing misclassified samples")
print("="*65)

# Use the best model's confidence (max proba) to rank samples
# Least confident first → most likely to be wrong
max_conf = best_proba.max(axis=1)
is_wrong  = (best_pred != y_te.values)
n_total   = len(y_te)
n_wrong   = is_wrong.sum()

print(f"Total test samples : {n_total}")
print(f"Initially wrong    : {n_wrong}  ({n_wrong/n_total*100:.1f}%)")
print(f"Initially correct  : {n_total-n_wrong}  ({(n_total-n_wrong)/n_total*100:.1f}%)")

# Build the curve: sort all wrong samples by confidence (remove lowest confidence first)
wrong_idx = np.where(is_wrong)[0]
sort_order = wrong_idx[np.argsort(max_conf[wrong_idx])]  # least confident wrong samples first

acc_curve  = [best_acc]
size_curve = [n_total]
removed_curve = [0]

mask = np.ones(n_total, dtype=bool)
for step, idx in enumerate(sort_order):
    mask[idx] = False
    remaining_pred = best_pred[mask]
    remaining_true = y_te.values[mask]
    new_acc = accuracy_score(remaining_true, remaining_pred)
    acc_curve.append(new_acc)
    size_curve.append(mask.sum())
    removed_curve.append(step + 1)

# Print milestones
print(f"\n{'Removed':>8}  {'Remaining':>10}  {'Accuracy':>9}  {'% Removed':>10}")
print("-"*45)
milestones = [0, 5, 10, 20, 30, 50, n_wrong]
for n_rem in milestones:
    if n_rem <= n_wrong:
        idx_c = n_rem
        print(f"  {n_rem:>6}     {size_curve[idx_c]:>8}      {acc_curve[idx_c]:>8.4f}     {n_rem/n_total*100:>8.1f}%")

# ─── 13. Visualizations ───────────────────────────────────────────────────────
fi_s = pd.Series(lgb_m.feature_importances_, index=sel).sort_values(ascending=False)

fig = plt.figure(figsize=(24, 28))
fig.suptitle("Bridge Condition — Strong 3-Class Model + Artificial Accuracy Analysis",
             fontsize=17, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.44)

# (a) CV Accuracy comparison
ax0 = fig.add_subplot(gs[0, :2])
cv_models  = ["LightGBM","CatBoost","XGBoost","ExtraTrees","Stacking"]
cv_means   = [cv_scores[m].mean() for m in cv_models]
cv_stds    = [cv_scores[m].std()  for m in cv_models]
test_accs  = [preds[m][1] for m in cv_models]
xp = np.arange(len(cv_models)); w = 0.36
b1 = ax0.bar(xp-w/2, cv_means, w, yerr=cv_stds, capsize=4,
             label="10-Fold CV Acc", color="#5C6BC0", alpha=0.87)
b2 = ax0.bar(xp+w/2, test_accs, w, label="Test Acc",
             color="#26A69A", alpha=0.87)
ax0.set_xticks(xp); ax0.set_xticklabels(cv_models, fontsize=10)
ax0.set_ylim(0.5, 1.0)
ax0.set_title("CV vs Test Accuracy per Model (10-Fold CV)", fontsize=12, fontweight="bold")
ax0.set_ylabel("Accuracy"); ax0.legend(fontsize=9)
for b, v in zip([*b1, *b2], [*cv_means, *test_accs]):
    ax0.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
             f"{v:.3f}", ha="center", va="bottom", fontsize=8)

# (b) Class distribution
ax1 = fig.add_subplot(gs[0, 2])
counts = pd.Series(le.inverse_transform(y.values)).value_counts()[["G","F","P"]]
colors_bar = [PALETTE["Good"], PALETTE["Fair"], PALETTE["Poor"]]
bars = ax1.bar([LABEL_MAP[i] for i in [0,1,2]], [counts["G"],counts["F"],counts["P"]],
               color=colors_bar, edgecolor="white")
for b, v in zip(bars, [counts["G"],counts["F"],counts["P"]]):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+5,
             str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_title("Class Distribution", fontsize=12, fontweight="bold")
ax1.set_ylabel("Count")

# (c) Confusion matrix — best model
ax2 = fig.add_subplot(gs[1, :2])
cm_arr = confusion_matrix(y_te, best_pred, labels=[0,1,2])
cm_pct = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax2, linewidths=0.5, cbar_kws={"label": "%"})
for i in range(3):
    for j in range(3):
        ax2.text(j+0.5, i+0.72, f"n={cm_arr[i,j]}", ha="center", fontsize=8, color="gray")
ax2.set_title(f"Confusion Matrix — {best_name} (Full Test Set)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")

# (d) Optuna convergence
ax3 = fig.add_subplot(gs[1, 2])
for study, lbl, col in [(sl,"LightGBM","#5C6BC0"), (sc,"CatBoost","#66BB6A"),
                         (sx,"XGBoost","#EF5350"),  (se,"ExtraTrees","#FFA726")]:
    v = [t.value for t in study.trials if t.value is not None]
    if v: ax3.plot(np.maximum.accumulate(v), lw=2, label=lbl, color=col)
ax3.set_title("Optuna Convergence\n(macro-F1 objective)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Trial"); ax3.set_ylabel("Best Val macro-F1"); ax3.legend(fontsize=8)

# (e) ★ ARTIFICIAL ACCURACY CURVE ★
ax4 = fig.add_subplot(gs[2, :])
pct_removed = np.array(removed_curve) / n_total * 100
ax4.plot(pct_removed, acc_curve, lw=2.5, color="#5C6BC0", zorder=3)
ax4.fill_between(pct_removed, best_acc, acc_curve, alpha=0.15, color="#5C6BC0")
ax4.axhline(best_acc, color="#EF5350", ls="--", lw=1.5, label=f"Baseline ({best_acc:.3f})")
ax4.axhline(1.0, color="#4CAF50", ls="--", lw=1.5, label="100%")
for milestone_pct in [5, 10, 20]:
    idx_m = int(milestone_pct / 100 * n_total)
    if idx_m <= n_wrong:
        ax4.annotate(f"{acc_curve[idx_m]:.3f}\n({milestone_pct}% removed)",
                     xy=(pct_removed[idx_m], acc_curve[idx_m]),
                     xytext=(pct_removed[idx_m]+0.5, acc_curve[idx_m]-0.04),
                     fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))
ax4.set_xlabel("% of Test Samples Removed (lowest-confidence failures first)", fontsize=11)
ax4.set_ylabel("Accuracy on Remaining Samples", fontsize=11)
ax4.set_title("★  Artificial Accuracy Curve  ★\n"
              "Removing misclassified samples (least confident first) — "
              "accuracy climbs to 100% when all failures removed",
              fontsize=12, fontweight="bold")
ax4.set_ylim(best_acc - 0.05, 1.03)
ax4.set_xlim(0, pct_removed[-1] + 0.5)
ax4.legend(fontsize=10); ax4.grid(True, alpha=0.3)

# (f) Per-class metrics
ax5 = fig.add_subplot(gs[3, :2])
prec, rec, f1pc, supp = precision_recall_fscore_support(y_te, best_pred, labels=[0,1,2])
xp2 = np.arange(3); w2 = 0.25
lnames = [f"{class_names[i]}\n(n={s})" for i, s in enumerate(supp)]
for off, vals, lbl, col in [(-w2, prec, "Precision", "#5C6BC0"),
                              (0,   rec,  "Recall",    "#26A69A"),
                              (w2,  f1pc, "F1",        "#FFA726")]:
    bars2 = ax5.bar(xp2+off, vals, w2, label=lbl, color=col, alpha=0.85)
    for b, v in zip(bars2, vals):
        ax5.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=9)
ax5.set_xticks(xp2); ax5.set_xticklabels(lnames, fontsize=11)
ax5.set_ylim(0, 1.18)
ax5.set_title(f"Per-Class Metrics — {best_name}", fontsize=12, fontweight="bold")
ax5.legend(fontsize=9); ax5.set_ylabel("Score")

# (g) Top feature importance
ax6 = fig.add_subplot(gs[3, 2])
top20 = fi_s.head(20)
ax6.barh(range(20), top20.values[::-1], color="#7E57C2", alpha=0.85)
ax6.set_yticks(range(20)); ax6.set_yticklabels(top20.index[::-1], fontsize=7)
ax6.set_title("Top 20 Features (LGB)", fontsize=11, fontweight="bold")
ax6.set_xlabel("Importance")

OUT = "/home/user/MLTUNNELS/bridge_strong_model_report.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nReport saved → {OUT}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"Dataset        : {len(df)} bridges  |  {len(feat_cols)} features → {len(sel)} MI-selected")
print(f"CV strategy    : 10-Fold Stratified  |  SMOTE balanced training")
print(f"Algorithms     : LGB(50) + CB(35) + XGB(35) + ET(30) + Stacking")
print(f"Best model     : {best_name}  Test Acc = {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"Wrong samples  : {n_wrong}/{n_total} in test set")
print(f"Artificial Acc : Remove {n_wrong} failures → 100.00% on remaining {n_total-n_wrong}")
print("="*65)
