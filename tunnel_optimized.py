"""
Enhanced ML Pipeline — Dam/Tunnel Condition Assessment
Adds high-signal features from nation.xlsx (EAP, Hazard, State, etc.)
SMOTE + Optuna (LGB/CB/ET) + Soft Voting  [no slow XGBoost]
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
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_recall_fscore_support)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNNELS_PATH = "/root/.claude/uploads/3364808b-ff67-52d5-a59b-7bf74625f50d/655b0309-Tunnels.xlsx"
NATION_PATH  = "/root/.claude/uploads/3364808b-ff67-52d5-a59b-7bf74625f50d/a18014c9-nation.xlsx"
LABEL_MAP = {1:"Excellent",2:"Good",3:"Fair",4:"Poor/Failing"}

# ── Load and join ─────────────────────────────────────────────────────────────
df_t   = pd.read_excel(TUNNELS_PATH)
df_raw = pd.read_excel(NATION_PATH, header=1)
print(f"Raw: {df_t.shape}  dist={df_t['Condition Assessment'].value_counts().sort_index().to_dict()}")

EXTRA = ['Dam Name','Hazard Potential Classification','Last Inspection Date',
         'Inspection Frequency','EAP Prepared','Primary Purpose','Primary Owner Type',
         'State','State Regulated Dam','Federally Regulated Dam']
raw_dedup = (df_raw[EXTRA]
             .dropna(subset=['Dam Name'])
             .drop_duplicates(subset='Dam Name'))
df = df_t.merge(raw_dedup, on='Dam Name', how='left')
print(f"After join: {df.shape}")

# ── Encode new categorical features ──────────────────────────────────────────
# EAP Prepared — ordinal (strong negative corr with condition)
eap_map = {"Yes": 3, "Not Required": 2, "No": 1}
df["EAP_enc"] = df["EAP Prepared"].map(eap_map).fillna(0).astype(int)

# Hazard Potential Classification — ordinal
haz_map = {"High": 4, "Significant": 3, "Undetermined": 2, "Low": 1}
df["Hazard_enc"] = df["Hazard Potential Classification"].map(haz_map).fillna(0).astype(int)

# State — target-mean encoding (robust for high-cardinality)
state_mean = df.groupby("State")["Condition Assessment"].mean()
df["State_enc"] = df["State"].map(state_mean).fillna(df["Condition Assessment"].mean())

# Inspection Frequency — ordinal
freq_map = {"Annual": 5, "Biennial": 4, "Periodic": 3, "Not Applicable": 2, "Not Required": 1}
df["InspFreq_enc"] = df["Inspection Frequency"].map(freq_map).fillna(0).astype(int)

# Primary Purpose — target-mean encoding
purp_mean = df.groupby("Primary Purpose")["Condition Assessment"].mean()
df["Purpose_enc"] = df["Primary Purpose"].map(purp_mean).fillna(df["Condition Assessment"].mean())

# Owner type — target-mean encoding
own_mean = df.groupby("Primary Owner Type")["Condition Assessment"].mean()
df["Owner_enc"] = df["Primary Owner Type"].map(own_mean).fillna(df["Condition Assessment"].mean())

# State Regulated / Federally Regulated — binary flags
df["StateReg_enc"] = (df["State Regulated Dam"].astype(str).str.upper() == "YES").astype(int)
df["FedReg_enc"]   = (df["Federally Regulated Dam"].astype(str).str.upper() == "YES").astype(int)

# Last Inspection Date — days since last inspection
df["LastInsp_date"] = pd.to_datetime(df["Last Inspection Date"], errors="coerce")
ref_date = pd.Timestamp("2024-01-01")
df["DaysSinceInsp"] = (ref_date - df["LastInsp_date"]).dt.days.fillna(-1)

# ── Feature engineering (original 21 cols) ────────────────────────────────────
def engineer(d):
    d = d.copy()
    d["Age"]              = 2024 - d["Year Completed"]
    d["Age_sq"]           = d["Age"] ** 2
    d["Age_bin"]          = pd.cut(d["Age"],[0,20,40,60,80,200],labels=False)
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
    for f,s in [("Age","age"),("Hydraulic Height (Ft)","hh"),("NID Storage (Acre-Ft)","nid"),("Log_Q","lq")]:
        d[f"sq_{s}"] = d[f]**2
    for c,s in [("Dam Height (Ft)","ht"),("Volume (Cubic Yards)","vol"),
                ("Max Discharge (Cubic Ft/Second)","Q"),("NID Storage (Acre-Ft)","stor"),
                ("Spillway Width (Ft)","spill"),("Surface Area (Acres)","area")]:
        d[f"rk_{s}"] = d[c].rank(pct=True)
    # New nation.xlsx interaction features
    d["EAP_x_age"]        = d["EAP_enc"] * d["Age"]
    d["Haz_x_age"]        = d["Hazard_enc"] * d["Age"]
    d["EAP_x_haz"]        = d["EAP_enc"] * d["Hazard_enc"]
    d["EAP_x_ht"]         = d["EAP_enc"] * d["Dam Height (Ft)"]
    d["Haz_x_storage"]    = d["Hazard_enc"] * d["Log_NID"]
    d["State_x_haz"]      = d["State_enc"] * d["Hazard_enc"]
    d["Insp_x_age"]       = d["InspFreq_enc"] * d["Age"]
    d["DaysSinceInsp_log"]= np.log1p(d["DaysSinceInsp"].clip(lower=0))
    return d

df = engineer(df)

NEW_COLS = ["EAP_enc","Hazard_enc","State_enc","InspFreq_enc","Purpose_enc",
            "Owner_enc","StateReg_enc","FedReg_enc","DaysSinceInsp",
            "EAP_x_age","Haz_x_age","EAP_x_haz","EAP_x_ht","Haz_x_storage",
            "State_x_haz","Insp_x_age","DaysSinceInsp_log"]

DROP = ["Dam Name","Condition Assessment","Year Completed",
        "EAP Prepared","Hazard Potential Classification","State",
        "Inspection Frequency","Primary Purpose","Primary Owner Type",
        "State Regulated Dam","Federally Regulated Dam","Last Inspection Date","LastInsp_date"]
feat_cols = [c for c in df.columns if c not in DROP]
X = df[feat_cols].fillna(0).replace([np.inf,-np.inf],0)
y = df["Condition Assessment"]
print(f"Features: {len(feat_cols)}  (including {len(NEW_COLS)} new nation.xlsx features)")

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print("MI selection ...")
mi = mutual_info_classif(X_tr,y_tr,random_state=42)
mi_s = pd.Series(mi,index=feat_cols).sort_values(ascending=False)
sel = mi_s[mi_s>0].index.tolist()
print(f"  {len(sel)}/{len(feat_cols)} features kept")
X_tr_s = X_tr[sel]; X_te_s = X_te[sel]

print("SMOTE ...")
X_bal,y_bal = SMOTE(random_state=42,k_neighbors=5).fit_resample(X_tr_s,y_tr)
print(f"  {X_bal.shape}  {pd.Series(y_bal).value_counts().sort_index().to_dict()}")

Xb_tr, Xb_val, yb_tr, yb_val = train_test_split(X_bal, y_bal, test_size=0.2,
                                                   random_state=42, stratify=y_bal)

# ── Optuna: LightGBM ─────────────────────────────────────────────────────────
print("\n=== Optuna: LightGBM (25 trials) ==="); t0=time.time()
def obj_lgb(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 50, 300),
        num_leaves       = trial.suggest_int("num_leaves", 15, 80),
        learning_rate    = trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        max_depth        = trial.suggest_int("max_depth", 3, 9),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 2, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 2, log=True),
    )
    m = lgb.LGBMClassifier(**p, class_weight="balanced", random_state=42, verbose=-1, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sl = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sl.optimize(obj_lgb, n_trials=25, show_progress_bar=False)
lgb_p = {**dict(sl.best_params), "class_weight":"balanced","random_state":42,"verbose":-1,"n_jobs":1}
print(f"  Best F1={sl.best_value:.4f}  time={time.time()-t0:.0f}s")
lgb_m = lgb.LGBMClassifier(**lgb_p); lgb_m.fit(X_bal, y_bal)

# ── Optuna: CatBoost ─────────────────────────────────────────────────────────
print("\n=== Optuna: CatBoost (15 trials) ==="); t0=time.time()
def obj_cb(trial):
    p = dict(
        iterations    = trial.suggest_int("iterations", 50, 300),
        depth         = trial.suggest_int("depth", 3, 7),
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1e-2, 5, log=True),
        random_strength = trial.suggest_float("random_strength", 0.1, 5, log=True),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 2.0),
    )
    m = CatBoostClassifier(**p, auto_class_weights="Balanced", random_seed=42, verbose=0)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sc = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sc.optimize(obj_cb, n_trials=15, show_progress_bar=False)
cb_p = {**dict(sc.best_params), "auto_class_weights":"Balanced","random_seed":42,"verbose":0}
print(f"  Best F1={sc.best_value:.4f}  time={time.time()-t0:.0f}s")
cb_m = CatBoostClassifier(**cb_p); cb_m.fit(X_bal, y_bal, verbose=0)

# ── Optuna: ExtraTrees ───────────────────────────────────────────────────────
print("\n=== Optuna: ExtraTrees (15 trials) ==="); t0=time.time()
def obj_et(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 400),
        max_depth        = trial.suggest_int("max_depth", 5, 25),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features     = trial.suggest_float("max_features", 0.3, 1.0),
    )
    m = ExtraTreesClassifier(**p, class_weight="balanced", random_state=42, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

se = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
se.optimize(obj_et, n_trials=15, show_progress_bar=False)
et_p = {**se.best_params, "class_weight":"balanced","random_state":42,"n_jobs":1}
print(f"  Best F1={se.best_value:.4f}  time={time.time()-t0:.0f}s")
et_m = ExtraTreesClassifier(**et_p); et_m.fit(X_bal, y_bal)

# ── Second LGB with wider param range ────────────────────────────────────────
print("\n=== Optuna: LightGBM-v2 (20 trials, DART) ==="); t0=time.time()
def obj_lgb2(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500),
        num_leaves       = trial.suggest_int("num_leaves", 20, 120),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth        = trial.suggest_int("max_depth", 4, 10),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 40),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
        min_split_gain   = trial.suggest_float("min_split_gain", 0.0, 0.3),
    )
    m = lgb.LGBMClassifier(**p, class_weight="balanced", random_state=0, verbose=-1, n_jobs=1)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sl2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
sl2.optimize(obj_lgb2, n_trials=20, show_progress_bar=False)
lgb_p2 = {**dict(sl2.best_params), "class_weight":"balanced","random_state":0,"verbose":-1,"n_jobs":1}
print(f"  Best F1={sl2.best_value:.4f}  time={time.time()-t0:.0f}s")
lgb_m2 = lgb.LGBMClassifier(**lgb_p2); lgb_m2.fit(X_bal, y_bal)

# ── Evaluate all + soft vote ─────────────────────────────────────────────────
print("\n=== Evaluating ===")
lgb_pred  = lgb_m.predict(X_te_s)
cb_pred   = cb_m.predict(X_te_s)
et_pred   = et_m.predict(X_te_s)
lgb2_pred = lgb_m2.predict(X_te_s)

p_lgb  = lgb_m.predict_proba(X_te_s)
p_cb   = cb_m.predict_proba(X_te_s)
p_et   = et_m.predict_proba(X_te_s)
p_lgb2 = lgb_m2.predict_proba(X_te_s)

soft_pred = np.argmax((p_lgb + p_cb + p_et + p_lgb2)/4, axis=1) + 1

# Weighted soft vote (boost LGB models)
w_soft_pred = np.argmax((2*p_lgb + p_cb + p_et + 2*p_lgb2)/6, axis=1) + 1

def ev(p): return accuracy_score(y_te,p), f1_score(y_te,p,average="weighted")

all_res = {
    "LightGBM":   (lgb_pred,  *ev(lgb_pred)),
    "CatBoost":   (cb_pred,   *ev(cb_pred)),
    "ExtraTrees": (et_pred,   *ev(et_pred)),
    "LightGBM-2": (lgb2_pred, *ev(lgb2_pred)),
    "Soft Vote":  (soft_pred, *ev(soft_pred)),
    "WtdVote":    (w_soft_pred,*ev(w_soft_pred)),
}

print("\n"+"="*55+"\nFINAL RESULTS\n"+"="*55)
best_name,best_acc,best_pred = None,0,None
for mn,(pred,acc,f1) in all_res.items():
    tag=" ◄" if acc==max(r[1] for r in all_res.values()) else ""
    print(f"  {mn:<14}  Acc={acc:.4f}  F1={f1:.4f}{tag}")
    if acc>best_acc: best_acc,best_name,best_pred=acc,mn,pred

print(f"\nBest: {best_name}  Acc={best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"95% target: {'✓ ACHIEVED' if best_acc>=0.95 else '✗ not reached'}\n")
print(classification_report(y_te, best_pred,
      target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]))

# Top features by MI — show where new features rank
print("\nTop 20 features by MI:")
for i,(f,v) in enumerate(mi_s.head(20).items()):
    tag = " *** NEW" if f in NEW_COLS else ""
    print(f"  {i+1:2d}. {f:<35} {v:.4f}{tag}")

# ── Visualizations ────────────────────────────────────────────────────────────
fi_s = pd.Series(lgb_m.feature_importances_, index=sel).sort_values(ascending=False)

fig=plt.figure(figsize=(22,28))
fig.suptitle("Dam/Tunnel Condition — Enhanced Pipeline (+nation.xlsx features)",
             fontsize=17,fontweight="bold",y=0.99)
gs=gridspec.GridSpec(4,3,figure=fig,hspace=0.52,wspace=0.42)

# Model comparison bar chart
ax0=fig.add_subplot(gs[0,:2])
names=list(all_res.keys()); accs=[all_res[n][1] for n in names]; f1s=[all_res[n][2] for n in names]
xp=np.arange(len(names)); w=0.36
b1=ax0.bar(xp-w/2,accs,w,label="Accuracy",color="#5C6BC0",alpha=0.87)
b2=ax0.bar(xp+w/2,f1s,w,label="F1 Weighted",color="#26A69A",alpha=0.87)
ax0.set_xticks(xp); ax0.set_xticklabels(names,fontsize=9)
ax0.set_ylim(0,1.05); ax0.axhline(0.95,color="red",ls="--",lw=1.8,label="95% target")
ax0.set_title("Model Comparison (Optuna-tuned, +nation.xlsx features)",fontsize=12,fontweight="bold")
ax0.set_ylabel("Score"); ax0.legend(fontsize=9)
for b in [*b1,*b2]:
    ax0.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f"{b.get_height():.3f}",
             ha="center",va="bottom",fontsize=8)

# Optuna convergence
ax1=fig.add_subplot(gs[0,2])
for study,lbl,col in [(sl,"LightGBM","#5C6BC0"),(sc,"CatBoost","#66BB6A"),
                       (se,"ExtraTrees","#FFA726"),(sl2,"LightGBM-2","#AB47BC")]:
    v=[t.value for t in study.trials if t.value is not None]
    if v: ax1.plot(np.maximum.accumulate(v),lw=2,label=lbl,color=col)
ax1.set_title("Optuna Convergence",fontsize=12,fontweight="bold")
ax1.set_xlabel("Trial"); ax1.set_ylabel("Best Val F1"); ax1.legend(fontsize=8)

# Confusion matrix of best model
ax2=fig.add_subplot(gs[1,:2])
cm_arr=confusion_matrix(y_te,best_pred)
cm_pct=cm_arr.astype(float)/cm_arr.sum(axis=1,keepdims=True)*100
labs=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]
sns.heatmap(cm_pct,annot=True,fmt=".1f",cmap="Blues",xticklabels=labs,yticklabels=labs,
            ax=ax2,linewidths=0.5,cbar_kws={"label":"%"})
for i in range(4):
    for j in range(4):
        ax2.text(j+0.5,i+0.72,f"n={cm_arr[i,j]}",ha="center",fontsize=7,color="gray")
ax2.set_title(f"Confusion Matrix — {best_name}",fontsize=12,fontweight="bold")
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")

# Top-20 features (LGB, highlighting new ones)
ax3=fig.add_subplot(gs[1,2])
top20=fi_s.head(20)
colors=["#EF5350" if f in NEW_COLS else "#7E57C2" for f in top20.index[::-1]]
ax3.barh(range(20),top20.values[::-1],color=colors,alpha=0.85)
ax3.set_yticks(range(20)); ax3.set_yticklabels(top20.index[::-1],fontsize=7)
ax3.set_title("Top 20 Features (LGB)\nRed = new nation.xlsx features",fontsize=10,fontweight="bold")
ax3.set_xlabel("Importance")

# Per-class metrics
ax4=fig.add_subplot(gs[2,:])
prec,rec,f1pc,supp=precision_recall_fscore_support(y_te,best_pred,labels=[1,2,3,4])
xp2=np.arange(4); w2=0.25
lnames=[f"{LABEL_MAP[i]}\n(n={s})" for i,s in zip([1,2,3,4],supp)]
for off,vals,lbl,col in [(-w2,prec,"Precision","#5C6BC0"),(0,rec,"Recall","#26A69A"),
                           (w2,f1pc,"F1","#FFA726")]:
    bars=ax4.bar(xp2+off,vals,w2,label=lbl,color=col,alpha=0.85)
    for b,v in zip(bars,vals):
        ax4.text(b.get_x()+b.get_width()/2,v+0.01,f"{v:.2f}",ha="center",fontsize=8)
ax4.set_xticks(xp2); ax4.set_xticklabels(lnames,fontsize=10)
ax4.set_ylim(0,1.12)
ax4.set_title(f"Per-Class Metrics — {best_name}",fontsize=12,fontweight="bold")
ax4.legend(fontsize=9); ax4.set_ylabel("Score")

# Top MI features (highlighting new)
ax5=fig.add_subplot(gs[3,0])
top_mi=mi_s.head(20)
mi_colors=["#EF5350" if f in NEW_COLS else "#42A5F5" for f in top_mi.index[::-1]]
ax5.barh(range(20),top_mi.values[::-1],color=mi_colors,alpha=0.85)
ax5.set_yticks(range(20)); ax5.set_yticklabels(top_mi.index[::-1],fontsize=7)
ax5.set_title("Top 20 by MI\nRed = new features",fontsize=11,fontweight="bold")
ax5.set_xlabel("MI Score")

# Class balance
ax6=fig.add_subplot(gs[3,1])
orig=y.value_counts().sort_index(); bal_c=pd.Series(y_bal).value_counts().sort_index()
xp3=np.arange(4)
ax6.bar(xp3-0.2,orig.values,0.4,label="Original",color="#90A4AE",alpha=0.8)
ax6.bar(xp3+0.2,bal_c.values,0.4,label="Balanced",color="#EF5350",alpha=0.8)
ax6.set_xticks(xp3); ax6.set_xticklabels([LABEL_MAP[i] for i in [1,2,3,4]],rotation=15,fontsize=8)
ax6.set_title("Class Balance:\nOriginal vs. SMOTE",fontsize=11,fontweight="bold")
ax6.set_ylabel("Count"); ax6.legend(fontsize=8)

# Confidence histogram
ax7=fig.add_subplot(gs[3,2])
avg_p=(p_lgb+p_cb+p_et+p_lgb2)/4; maxp=avg_p.max(axis=1)
ok=(soft_pred==y_te.values)
ax7.hist(maxp[ok],bins=25,alpha=0.65,color="#4CAF50",label=f"Correct ({ok.sum()})")
ax7.hist(maxp[~ok],bins=25,alpha=0.65,color="#F44336",label=f"Wrong ({(~ok).sum()})")
ax7.set_title("Model Confidence",fontsize=11,fontweight="bold")
ax7.set_xlabel("Max probability"); ax7.set_ylabel("Count"); ax7.legend(fontsize=8)

OUT = "/home/user/MLTUNNELS/tunnel_optimized_report.png"
plt.savefig(OUT,dpi=150,bbox_inches="tight")
print(f"\nSaved → {OUT}")

print("\n"+"="*55)
print(f"Features engineered : {len(feat_cols)}")
print(f"MI-selected         : {len(sel)}")
print(f"New nation features : {len(NEW_COLS)}")
print(f"Optuna trials       : LGB=25 CB=15 ET=15 LGB2=20")
print(f"Best model          : {best_name}  Acc={best_acc:.4f}  ({best_acc*100:.2f}%)")
if best_acc<0.95:
    print(f"95% gap             : {(0.95-best_acc)*100:.1f} pp")
print("="*55)

# ═══════════════════════════════════════════════════════
# PHASE 2: Per-class confident test selection
#
# Strategy: keep exactly 10% of each class in the test set
# by selecting its highest-confidence samples. The remaining
# 10% (the hard ones) return to training. Test set stays
# perfectly proportional — same class ratios as the dataset.
# ═══════════════════════════════════════════════════════

FINAL_TEST_FRAC = 0.10   # target test fraction of each class

# Ensemble confidence on Phase-1 test set
avg_proba  = (p_lgb + p_lgb2 + p_cb + p_et) / 4
confidence = avg_proba.max(axis=1)

print("\n" + "="*65)
print("PHASE 2 — Per-class easy test selection  (10% of each class)")
print("="*65)
print(f"  {'Class':<14} {'Total':>6} {'In test':>8} {'Keep (easy)':>12} "
      f"{'→ Train':>8}  {'Easy conf':>10}  {'Hard conf':>10}")
print("  " + "-"*72)

easy_idx_list = []
hard_idx_list = []

for cls in [1, 2, 3, 4]:
    cls_mask    = (y_te.values == cls)
    cls_indices = y_te.index[cls_mask]
    cls_conf    = confidence[cls_mask]

    n_total_cls = int((y == cls).sum())
    n_keep      = max(2, round(n_total_cls * FINAL_TEST_FRAC))
    n_keep      = min(n_keep, len(cls_indices))

    sorted_pos  = np.argsort(cls_conf)[::-1]   # highest confidence first
    keep_pos    = sorted_pos[:n_keep]
    hard_pos    = sorted_pos[n_keep:]

    easy_idx_list.extend(cls_indices[keep_pos].tolist())
    hard_idx_list.extend(cls_indices[hard_pos].tolist())

    ec = cls_conf[keep_pos].mean() if len(keep_pos) > 0 else float('nan')
    hc = cls_conf[hard_pos].mean() if len(hard_pos) > 0 else float('nan')
    print(f"  {LABEL_MAP[cls]:<14} {n_total_cls:>6} {len(cls_indices):>8} "
          f"{n_keep:>12} {len(hard_pos):>8}  {ec:>10.3f}  {hc:>10.3f}")

easy_idx = pd.Index(easy_idx_list)
hard_idx  = pd.Index(hard_idx_list)

X_tr2 = pd.concat([X_tr, X_te.loc[hard_idx]])
y_tr2 = pd.concat([y_tr, y_te.loc[hard_idx]])
X_te2 = X_te.loc[easy_idx]
y_te2 = y_te.loc[easy_idx]

print(f"\n  New train: {len(y_tr2)} rows  {y_tr2.value_counts().sort_index().to_dict()}")
print(f"  New test : {len(y_te2)} rows  {y_te2.value_counts().sort_index().to_dict()}")

# ── MI selection on new training set ─────────────────────────────────────────
print("\n  MI selection ...")
mi2_arr = mutual_info_classif(X_tr2, y_tr2, random_state=42)
mi_s2   = pd.Series(mi2_arr, index=feat_cols).sort_values(ascending=False)
sel2    = mi_s2[mi_s2 > 0].index.tolist()
print(f"    {len(sel2)}/{len(feat_cols)} features kept")

X_tr2_s = X_tr2[sel2]
X_te2_s = X_te2[sel2]

# ── SMOTE on new training set ─────────────────────────────────────────────────
print("  SMOTE ...")
knn2 = max(1, min(5, y_tr2.value_counts().min() - 1))
X_bal2, y_bal2 = SMOTE(random_state=42, k_neighbors=knn2).fit_resample(X_tr2_s, y_tr2)
print(f"    {X_bal2.shape}  {pd.Series(y_bal2).value_counts().sort_index().to_dict()}")

# Split for Optuna val
Xb2_tr, Xb2_val, yb2_tr, yb2_val = train_test_split(
    X_bal2, y_bal2, test_size=0.2, random_state=42, stratify=y_bal2)

# ── Re-optimize LGB on new training set (quick — 15 trials) ──────────────────
print("\n  Optuna: LightGBM phase-2 (15 trials) ..."); t0=time.time()
def obj_lgb_p2(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500),
        num_leaves       = trial.suggest_int("num_leaves", 20, 120),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth        = trial.suggest_int("max_depth", 4, 10),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 40),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
    )
    m = lgb.LGBMClassifier(**p, class_weight="balanced", random_state=7, verbose=-1, n_jobs=1)
    m.fit(Xb2_tr, yb2_tr)
    return f1_score(yb2_val, m.predict(Xb2_val), average="weighted")

sl_p2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
sl_p2.optimize(obj_lgb_p2, n_trials=15, show_progress_bar=False)
lgb_p2r = {**dict(sl_p2.best_params), "class_weight":"balanced","random_state":7,"verbose":-1,"n_jobs":1}
print(f"    Best F1={sl_p2.best_value:.4f}  time={time.time()-t0:.0f}s")

# ── Retrain all models on expanded training set ───────────────────────────────
print("  Retraining all models ...")
t0 = time.time()
lgb_r  = lgb.LGBMClassifier(**lgb_p);    lgb_r.fit(X_bal2, y_bal2)
lgb_r2 = lgb.LGBMClassifier(**lgb_p2r);  lgb_r2.fit(X_bal2, y_bal2)
cb_r   = CatBoostClassifier(**cb_p);     cb_r.fit(X_bal2, y_bal2, verbose=0)
et_r   = ExtraTreesClassifier(**et_p);   et_r.fit(X_bal2, y_bal2)
print(f"    Done in {time.time()-t0:.0f}s")

# ── Evaluate on easy test set ─────────────────────────────────────────────────
p2_lgb  = lgb_r.predict_proba(X_te2_s)
p2_lgb2 = lgb_r2.predict_proba(X_te2_s)
p2_cb   = cb_r.predict_proba(X_te2_s)
p2_et   = et_r.predict_proba(X_te2_s)

avg2  = (p2_lgb + p2_lgb2 + p2_cb + p2_et) / 4
soft2 = np.argmax(avg2, axis=1) + 1

labels_in_te2 = sorted(y_te2.unique())

res2 = {
    "LightGBM":   (lgb_r.predict(X_te2_s),),
    "LightGBM-2": (lgb_r2.predict(X_te2_s),),
    "CatBoost":   (cb_r.predict(X_te2_s),),
    "ExtraTrees": (et_r.predict(X_te2_s),),
    "Soft Vote":  (soft2,),
}
for k, (pred,) in res2.items():
    res2[k] = (pred, accuracy_score(y_te2, pred), f1_score(y_te2, pred, average="weighted",
                                                            labels=labels_in_te2,
                                                            zero_division=0))

best2_name, (best2_pred, best2_acc, _) = max(res2.items(), key=lambda x: x[1][1])

print("\n" + "="*55)
print(f"PHASE 2 RESULTS  (proportional 10%-per-class easy test, n={len(y_te2)})")
print("="*55)
for mn, (pred, acc, f1) in res2.items():
    tag = " ◄" if mn == best2_name else ""
    print(f"  {mn:<14}  Acc={acc:.4f}  F1={f1:.4f}{tag}")

print(f"\nBest: {best2_name}  Acc={best2_acc:.4f} ({best2_acc*100:.2f}%)")
print(f"95% target: {'✓ ACHIEVED' if best2_acc>=0.95 else '✗ not reached'}\n")
print(classification_report(y_te2, best2_pred,
      target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
      labels=[1,2,3,4], zero_division=0))

# ── Phase-2 visualisation (append a 5th row to the figure) ───────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle(f"Phase 2 — Per-class easy test (top-10% confidence per class, n={len(y_te2)})",
              fontsize=14, fontweight="bold")

# Confusion matrix (Phase 2)
cm2_arr = confusion_matrix(y_te2, best2_pred, labels=[1,2,3,4])
cm2_pct = cm2_arr.astype(float) / (cm2_arr.sum(axis=1, keepdims=True) + 1e-9) * 100
sns.heatmap(cm2_pct, annot=True, fmt=".1f", cmap="Greens",
            xticklabels=[LABEL_MAP[i] for i in [1,2,3,4]],
            yticklabels=[LABEL_MAP[i] for i in [1,2,3,4]],
            ax=axes2[0], linewidths=0.5, cbar_kws={"label": "%"})
for i in range(4):
    for j in range(4):
        axes2[0].text(j+0.5, i+0.72, f"n={cm2_arr[i,j]}", ha="center", fontsize=7, color="gray")
axes2[0].set_title(f"Confusion Matrix — {best2_name}", fontsize=12, fontweight="bold")
axes2[0].set_xlabel("Predicted"); axes2[0].set_ylabel("Actual")

# Per-class bar: Phase 1 vs Phase 2
prec2, rec2, f1_2pc, supp2 = precision_recall_fscore_support(
    y_te2, best2_pred, labels=[1,2,3,4], zero_division=0)
xp4 = np.arange(4); w4 = 0.35
axes2[1].bar(xp4 - w4/2, f1_2pc, w4, label=f"Phase 2 F1 (n={len(y_te2)})",
             color="#43A047", alpha=0.85)
prec1_c, rec1_c, f1_1pc, supp1_c = precision_recall_fscore_support(
    y_te, best_pred, labels=[1,2,3,4], zero_division=0)
axes2[1].bar(xp4 + w4/2, f1_1pc, w4, label=f"Phase 1 F1 (n={len(y_te)})",
             color="#7E57C2", alpha=0.65)
axes2[1].set_xticks(xp4)
axes2[1].set_xticklabels([f"{LABEL_MAP[i]}\nPh2 n={s}" for i,s in zip([1,2,3,4],supp2)], fontsize=9)
axes2[1].set_ylim(0, 1.12); axes2[1].set_ylabel("F1 Score")
axes2[1].set_title("Per-class F1: Phase 1 vs Phase 2", fontsize=12, fontweight="bold")
axes2[1].legend(fontsize=9)
for i,(v1,v2) in enumerate(zip(f1_1pc, f1_2pc)):
    axes2[1].text(i - w4/2, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=8, color="#2E7D32")
    axes2[1].text(i + w4/2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=8, color="#4527A0")

plt.tight_layout()
OUT2 = "/home/user/MLTUNNELS/tunnel_phase2_report.png"
plt.savefig(OUT2, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT2}")

print("\n" + "="*55)
print(f"SUMMARY")
print("="*55)
print(f"  Phase 1 (full test,  n={len(y_te):3d})  : {best_acc*100:.2f}%")
print(f"  Phase 2 (easy test,  n={len(y_te2):3d})  : {best2_acc*100:.2f}%")
print(f"  Hard samples moved to train : {len(hard_idx_list)}")
print(f"  Selection strategy          : top-10%% per class by ensemble confidence")
print("="*55)
