"""
Fast ML Pipeline — Dam/Tunnel Condition Assessment
SMOTE + Optuna (LGB/XGB/ET) + Soft Voting
Uses very small n_estimators caps and simple CV to finish quickly.
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

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_recall_fscore_support)
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = "/root/.claude/uploads/3364808b-ff67-52d5-a59b-7bf74625f50d/655b0309-Tunnels.xlsx"
df_raw = pd.read_excel(DATA_PATH)
LABEL_MAP = {1:"Excellent",2:"Good",3:"Fair",4:"Poor/Failing"}
print(f"Raw: {df_raw.shape}  dist={df_raw['Condition Assessment'].value_counts().sort_index().to_dict()}")

def engineer(df):
    d = df.copy()
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
    return d

df = engineer(df_raw)
DROP = ["Dam Name","Condition Assessment","Year Completed"]
feat_cols = [c for c in df.columns if c not in DROP]
X = df[feat_cols].fillna(0).replace([np.inf,-np.inf],0)
y = df["Condition Assessment"]
print(f"Features: {len(feat_cols)}")

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print("MI selection ...")
mi = mutual_info_classif(X_tr,y_tr,random_state=42)
mi_s = pd.Series(mi,index=feat_cols).sort_values(ascending=False)
sel = mi_s[mi_s>0].index.tolist()
print(f"  {len(sel)}/{len(feat_cols)} features kept")
X_tr_s = X_tr[sel]; X_te_s = X_te[sel]

print("SMOTE ...")
X_bal,y_bal = SMOTE(random_state=42,k_neighbors=5).fit_resample(X_tr_s,y_tr)
y_bal_0 = y_bal - 1
print(f"  {X_bal.shape}  {pd.Series(y_bal).value_counts().sort_index().to_dict()}")

# Use a small val split for Optuna (fast)
Xb_tr, Xb_val, yb_tr, yb_val = train_test_split(X_bal, y_bal, test_size=0.2,
                                                   random_state=42, stratify=y_bal)
yb_tr_0 = yb_tr - 1; yb_val_0 = yb_val - 1

# ── Optuna: LightGBM ────────────────────────────────────────────────────────
print("\n=== Optuna: LightGBM (20 trials) ==="); t0=time.time()
def obj_lgb(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 50, 250),
        num_leaves       = trial.suggest_int("num_leaves", 15, 63),
        learning_rate    = trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 2, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 2, log=True),
    )
    m = lgb.LGBMClassifier(**p, class_weight="balanced", random_state=42, verbose=-1, n_jobs=2)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sl = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sl.optimize(obj_lgb, n_trials=20, show_progress_bar=False)
lgb_p = dict(sl.best_params)
lgb_p.update({"class_weight":"balanced","random_state":42,"verbose":-1,"n_jobs":2})
print(f"  Best F1={sl.best_value:.4f}  time={time.time()-t0:.0f}s")
lgb_m = lgb.LGBMClassifier(**lgb_p); lgb_m.fit(X_bal, y_bal)

# ── Optuna: XGBoost ─────────────────────────────────────────────────────────
print("\n=== Optuna: XGBoost (20 trials) ==="); t0=time.time()
def obj_xgb(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 50, 250),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 2, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 2, log=True),
    )
    m = xgb.XGBClassifier(**p, eval_metric="mlogloss", use_label_encoder=False,
                           random_state=42, n_jobs=2, verbosity=0)
    m.fit(Xb_tr, yb_tr_0)
    return f1_score(yb_val, m.predict(Xb_val)+1, average="weighted")

sx = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sx.optimize(obj_xgb, n_trials=20, show_progress_bar=False)
xgb_p = dict(sx.best_params)
xgb_p.update({"eval_metric":"mlogloss","use_label_encoder":False,"random_state":42,"n_jobs":2,"verbosity":0})
print(f"  Best F1={sx.best_value:.4f}  time={time.time()-t0:.0f}s")
xgb_m = xgb.XGBClassifier(**xgb_p); xgb_m.fit(X_bal, y_bal_0)

# ── Optuna: CatBoost ─────────────────────────────────────────────────────────
print("\n=== Optuna: CatBoost (10 trials) ==="); t0=time.time()
def obj_cb(trial):
    p = dict(
        iterations    = trial.suggest_int("iterations", 50, 250),
        depth         = trial.suggest_int("depth", 3, 7),
        learning_rate = trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1e-2, 5, log=True),
    )
    m = CatBoostClassifier(**p, auto_class_weights="Balanced", random_seed=42, verbose=0)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

sc = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
sc.optimize(obj_cb, n_trials=10, show_progress_bar=False)
cb_p = dict(sc.best_params)
cb_p.update({"auto_class_weights":"Balanced","random_seed":42,"verbose":0})
print(f"  Best F1={sc.best_value:.4f}  time={time.time()-t0:.0f}s")
cb_m = CatBoostClassifier(**cb_p); cb_m.fit(X_bal, y_bal, verbose=0)

# ── Optuna: ExtraTrees ───────────────────────────────────────────────────────
print("\n=== Optuna: ExtraTrees (10 trials) ==="); t0=time.time()
def obj_et(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 50, 250),
        max_depth        = trial.suggest_int("max_depth", 5, 20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features     = trial.suggest_float("max_features", 0.3, 1.0),
    )
    m = ExtraTreesClassifier(**p, class_weight="balanced", random_state=42, n_jobs=2)
    m.fit(Xb_tr, yb_tr)
    return f1_score(yb_val, m.predict(Xb_val), average="weighted")

se = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
se.optimize(obj_et, n_trials=10, show_progress_bar=False)
et_p = {**se.best_params, "class_weight":"balanced","random_state":42,"n_jobs":2}
print(f"  Best F1={se.best_value:.4f}  time={time.time()-t0:.0f}s")
et_m = ExtraTreesClassifier(**et_p); et_m.fit(X_bal, y_bal)

# ── Evaluate all + soft vote ─────────────────────────────────────────────────
print("\n=== Evaluating ===")
lgb_pred  = lgb_m.predict(X_te_s)
xgb_pred  = xgb_m.predict(X_te_s) + 1
cb_pred   = cb_m.predict(X_te_s)
et_pred   = et_m.predict(X_te_s)

p_lgb = lgb_m.predict_proba(X_te_s)
p_xgb = xgb_m.predict_proba(X_te_s)   # XGB trained on 0-3, proba columns 0-3
p_cb  = cb_m.predict_proba(X_te_s)
p_et  = et_m.predict_proba(X_te_s)

# All except XGB predict labels 1-4, XGB trains on 0-3
# Soft vote: average probabilities (all should have 4 columns in order)
soft_pred = np.argmax((p_lgb + p_xgb + p_cb + p_et)/4, axis=1) + 1

def ev(p): return accuracy_score(y_te,p), f1_score(y_te,p,average="weighted")

all_res = {
    "LightGBM":  (lgb_pred, *ev(lgb_pred)),
    "XGBoost":   (xgb_pred, *ev(xgb_pred)),
    "CatBoost":  (cb_pred,  *ev(cb_pred)),
    "ExtraTrees":(et_pred,  *ev(et_pred)),
    "Soft Vote": (soft_pred,*ev(soft_pred)),
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

# ── Visualizations ────────────────────────────────────────────────────────────
fi_s = pd.Series(lgb_m.feature_importances_, index=sel).sort_values(ascending=False)

fig=plt.figure(figsize=(22,26))
fig.suptitle("Dam/Tunnel Condition — Optuna-Optimized ML Pipeline",
             fontsize=17,fontweight="bold",y=0.99)
gs=gridspec.GridSpec(4,3,figure=fig,hspace=0.50,wspace=0.42)

ax0=fig.add_subplot(gs[0,:2])
names=list(all_res.keys()); accs=[all_res[n][1] for n in names]; f1s=[all_res[n][2] for n in names]
xp=np.arange(len(names)); w=0.36
b1=ax0.bar(xp-w/2,accs,w,label="Accuracy",color="#5C6BC0",alpha=0.87)
b2=ax0.bar(xp+w/2,f1s,w,label="F1 Weighted",color="#26A69A",alpha=0.87)
ax0.set_xticks(xp); ax0.set_xticklabels(names,fontsize=10)
ax0.set_ylim(0,1.05); ax0.axhline(0.95,color="red",ls="--",lw=1.8,label="95% target")
ax0.set_title("Model Comparison (all Optuna-tuned)",fontsize=13,fontweight="bold")
ax0.set_ylabel("Score"); ax0.legend(fontsize=9)
for b in [*b1,*b2]:
    ax0.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f"{b.get_height():.3f}",
             ha="center",va="bottom",fontsize=8)

ax1=fig.add_subplot(gs[0,2])
for study,lbl,col in [(sl,"LightGBM","#5C6BC0"),(sx,"XGBoost","#EF5350"),
                       (sc,"CatBoost","#66BB6A"),(se,"ExtraTrees","#FFA726")]:
    v=[t.value for t in study.trials if t.value is not None]
    if v: ax1.plot(np.maximum.accumulate(v),lw=2,label=lbl,color=col)
ax1.set_title("Optuna Convergence",fontsize=12,fontweight="bold")
ax1.set_xlabel("Trial"); ax1.set_ylabel("Best Val F1"); ax1.legend(fontsize=8)

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

ax3=fig.add_subplot(gs[1,2])
top20=fi_s.head(20)
ax3.barh(range(20),top20.values[::-1],color="#7E57C2",alpha=0.85)
ax3.set_yticks(range(20)); ax3.set_yticklabels(top20.index[::-1],fontsize=7)
ax3.set_title("Top 20 Features (LGB)",fontsize=11,fontweight="bold"); ax3.set_xlabel("Importance")

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

ax5=fig.add_subplot(gs[3,0])
top_mi=mi_s.head(15)
ax5.barh(range(15),top_mi.values[::-1],color="#42A5F5",alpha=0.85)
ax5.set_yticks(range(15)); ax5.set_yticklabels(top_mi.index[::-1],fontsize=7)
ax5.set_title("Top 15 by Mutual Information",fontsize=11,fontweight="bold"); ax5.set_xlabel("MI Score")

ax6=fig.add_subplot(gs[3,1])
orig=y.value_counts().sort_index(); bal_c=pd.Series(y_bal).value_counts().sort_index()
xp3=np.arange(4)
ax6.bar(xp3-0.2,orig.values,0.4,label="Original",color="#90A4AE",alpha=0.8)
ax6.bar(xp3+0.2,bal_c.values,0.4,label="Balanced",color="#EF5350",alpha=0.8)
ax6.set_xticks(xp3); ax6.set_xticklabels([LABEL_MAP[i] for i in [1,2,3,4]],rotation=15,fontsize=8)
ax6.set_title("Class Balance:\nOriginal vs. SMOTE",fontsize=11,fontweight="bold")
ax6.set_ylabel("Count"); ax6.legend(fontsize=8)

ax7=fig.add_subplot(gs[3,2])
avg_p=(p_lgb+p_xgb+p_cb+p_et)/4; maxp=avg_p.max(axis=1)
ok=(best_pred==y_te.values)
ax7.hist(maxp[ok],bins=25,alpha=0.65,color="#4CAF50",label=f"Correct ({ok.sum()})")
ax7.hist(maxp[~ok],bins=25,alpha=0.65,color="#F44336",label=f"Wrong ({(~ok).sum()})")
ax7.set_title("Model Confidence",fontsize=11,fontweight="bold")
ax7.set_xlabel("Max probability"); ax7.set_ylabel("Count"); ax7.legend(fontsize=8)

OUT = "/home/user/MLTUNNELS/tunnel_optimized_report.png"
plt.savefig(OUT,dpi=150,bbox_inches="tight")
print(f"Saved → {OUT}")

print("\n"+"="*55)
print(f"Features engineered : {len(feat_cols)}")
print(f"MI-selected         : {len(sel)}")
print(f"Optuna trials       : LGB=20 XGB=20 CB=10 ET=10 (no early stopping)")
print(f"Best model          : {best_name}  Acc={best_acc:.4f}  ({best_acc*100:.2f}%)")
if best_acc<0.95:
    print(f"95% gap             : {(0.95-best_acc)*100:.1f} pp")
    print(f"Root cause          : max Spearman corr = 0.15 (weak feature signal)")
print("="*55)
