"""
ML Pipeline: Predicting Tunnel/Dam Condition Assessment
Target: Condition Assessment (1=Excellent, 2=Good, 3=Fair, 4=Poor)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

import xgboost as xgb
from imblearn.over_sampling import SMOTE

# ── 1. Load Data ──────────────────────────────────────────────────────────────
DATA_PATH = "/root/.claude/uploads/3364808b-ff67-52d5-a59b-7bf74625f50d/655b0309-Tunnels.xlsx"
df = pd.read_excel(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['Condition Assessment'].value_counts().sort_index()}\n")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
LABEL_MAP = {1: "Excellent", 2: "Good", 3: "Fair", 4: "Poor/Failing"}

feature_cols = [
    "Distance to Nearest City (Miles)",
    "Primary Dam Type",
    "Core Types",
    "Foundation",
    "Dam Height (Ft)",
    "Hydraulic Height (Ft)",
    "Structural Height (Ft)",
    "NID Height (Ft)",
    "Dam Length (Ft)",
    "Volume (Cubic Yards)",
    "Year Completed",
    "NID Storage (Acre-Ft)",
    "Max Storage (Acre-Ft)",
    "Normal Storage (Acre-Ft)",
    "Surface Area (Acres)",
    "Drainage Area (Sq Miles)",
    "Max Discharge (Cubic Ft/Second)",
    "Spillway Type",
    "Spillway Width (Ft)",
]

# Derived features
df["Age"] = 2024 - df["Year Completed"]
df["Height_Range"] = df["Dam Height (Ft)"] - df["Hydraulic Height (Ft)"]
df["Storage_Ratio"] = df["NID Storage (Acre-Ft)"] / (df["Max Storage (Acre-Ft)"] + 1e-6)
df["Discharge_per_Area"] = df["Max Discharge (Cubic Ft/Second)"] / (df["Surface Area (Acres)"] + 1e-6)
df["Volume_per_Height"] = df["Volume (Cubic Yards)"] / (df["Dam Height (Ft)"] + 1e-6)

feature_cols += ["Age", "Height_Range", "Storage_Ratio", "Discharge_per_Area", "Volume_per_Height"]

X = df[feature_cols]
y = df["Condition Assessment"]
# XGBoost requires 0-based labels; shift 1-4 → 0-3 for training, shift back for display
y_xgb = y - 1

print(f"Features used: {len(feature_cols)}")

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 0-based splits for XGBoost
_, _, y_train_xgb, y_test_xgb = train_test_split(
    X, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
)

# Handle class imbalance with SMOTE on training set
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
_, y_train_bal_xgb = smote.fit_resample(X_train, y_train_xgb)
print(f"After SMOTE — train shape: {X_train_bal.shape}")
print(f"Balanced class counts: {pd.Series(y_train_bal).value_counts().sort_index().to_dict()}\n")

# ── 4. Define Models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    ),
}

# ── 5. Train & Evaluate ───────────────────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 60)
print("MODEL EVALUATION (5-Fold CV on balanced training set)")
print("=" * 60)

for name, model in models.items():
    # XGBoost needs 0-based labels
    _y_train = y_train_bal_xgb if name == "XGBoost" else y_train_bal
    _y_test = y_test_xgb if name == "XGBoost" else y_test

    cv_scores = cross_val_score(model, X_train_bal, _y_train, cv=cv,
                                scoring="f1_weighted", n_jobs=-1)
    model.fit(X_train_bal, _y_train)
    y_pred_raw = model.predict(X_test)
    # Shift XGBoost predictions back to 1-based
    y_pred = y_pred_raw + 1 if name == "XGBoost" else y_pred_raw

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_acc": acc,
        "test_f1": f1,
    }
    print(f"\n{name}")
    print(f"  CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy   : {acc:.4f}")
    print(f"  Test F1 (weighted): {f1:.4f}")

# ── 6. Best Model ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["test_f1"])
best = results[best_name]
print(f"\n{'='*60}")
print(f"Best Model: {best_name}  (F1={best['test_f1']:.4f})")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(
    y_test, best["y_pred"],
    target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]
))

# ── 7. Feature Importance (best tree model) ───────────────────────────────────
best_model = best["model"]
if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
elif hasattr(best_model, "named_steps"):
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        fi = pd.Series(np.abs(clf.coef_).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    else:
        fi = None
else:
    fi = None

# ── 8. Visualizations ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.suptitle("Dam/Tunnel Condition Assessment — ML Model Report", fontsize=18, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

PALETTE = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800", 4: "#F44336"}
CLASS_COLORS = [PALETTE[i] for i in sorted(LABEL_MAP)]

# (a) Target distribution
ax0 = fig.add_subplot(gs[0, 0])
counts = y.value_counts().sort_index()
bars = ax0.bar([LABEL_MAP[i] for i in counts.index], counts.values,
               color=[PALETTE[i] for i in counts.index], edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax0.set_title("Target Class Distribution", fontsize=12, fontweight="bold")
ax0.set_xlabel("Condition")
ax0.set_ylabel("Count")
ax0.tick_params(axis="x", rotation=15)

# (b) Model comparison bar chart
ax1 = fig.add_subplot(gs[0, 1])
model_names = list(results.keys())
f1_scores = [results[n]["test_f1"] for n in model_names]
acc_scores = [results[n]["test_acc"] for n in model_names]
x = np.arange(len(model_names))
w = 0.35
b1 = ax1.bar(x - w/2, acc_scores, w, label="Accuracy", color="#5C6BC0", alpha=0.85)
b2 = ax1.bar(x + w/2, f1_scores, w, label="F1 Weighted", color="#26A69A", alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=8)
ax1.set_ylim(0, 1.0)
ax1.set_title("Model Comparison", fontsize=12, fontweight="bold")
ax1.set_ylabel("Score")
ax1.legend(fontsize=8)
for b in [*b1, *b2]:
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
             f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)

# (c) CV scores with std
ax2 = fig.add_subplot(gs[0, 2])
cv_means = [results[n]["cv_mean"] for n in model_names]
cv_stds = [results[n]["cv_std"] for n in model_names]
colors = ["#EF5350", "#42A5F5", "#66BB6A", "#FFA726"]
ax2.barh(model_names, cv_means, xerr=cv_stds, color=colors, alpha=0.85,
         capsize=5, edgecolor="white")
ax2.set_xlabel("CV F1 (weighted)")
ax2.set_title("5-Fold CV Performance", fontsize=12, fontweight="bold")
ax2.set_xlim(0, 1.0)
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax2.text(m + s + 0.01, i, f"{m:.3f}", va="center", fontsize=9)

# (d) Confusion matrix — best model
ax3 = fig.add_subplot(gs[1, :2])
cm = confusion_matrix(y_test, best["y_pred"])
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax3, linewidths=0.5, cbar_kws={"label": "%"})
ax3.set_title(f"Confusion Matrix — {best_name} (% of true class)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

# (e) Feature importance
if fi is not None:
    ax4 = fig.add_subplot(gs[1, 2])
    top_fi = fi.head(15)
    bars_fi = ax4.barh(range(len(top_fi)), top_fi.values[::-1], color="#7E57C2", alpha=0.85)
    ax4.set_yticks(range(len(top_fi)))
    ax4.set_yticklabels(top_fi.index[::-1], fontsize=8)
    ax4.set_title(f"Top 15 Feature Importances\n({best_name})", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Importance")

# (f) Age vs Condition box plot
ax5 = fig.add_subplot(gs[2, 0])
groups = [df[df["Condition Assessment"] == c]["Age"].values for c in sorted(LABEL_MAP)]
bp = ax5.boxplot(groups, patch_artist=True, notch=False,
                 medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], CLASS_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_xticklabels([LABEL_MAP[i] for i in sorted(LABEL_MAP)], rotation=15, fontsize=8)
ax5.set_title("Age vs Condition", fontsize=12, fontweight="bold")
ax5.set_ylabel("Age (years)")

# (g) Dam Height vs Condition
ax6 = fig.add_subplot(gs[2, 1])
groups_h = [df[df["Condition Assessment"] == c]["Dam Height (Ft)"].values for c in sorted(LABEL_MAP)]
bp2 = ax6.boxplot(groups_h, patch_artist=True, notch=False,
                  medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp2["boxes"], CLASS_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_xticklabels([LABEL_MAP[i] for i in sorted(LABEL_MAP)], rotation=15, fontsize=8)
ax6.set_title("Dam Height vs Condition", fontsize=12, fontweight="bold")
ax6.set_ylabel("Height (ft)")

# (h) Volume vs Condition (log scale)
ax7 = fig.add_subplot(gs[2, 2])
groups_v = [df[df["Condition Assessment"] == c]["Volume (Cubic Yards)"].values for c in sorted(LABEL_MAP)]
bp3 = ax7.boxplot(groups_v, patch_artist=True, notch=False,
                  medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp3["boxes"], CLASS_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax7.set_yscale("log")
ax7.set_xticklabels([LABEL_MAP[i] for i in sorted(LABEL_MAP)], rotation=15, fontsize=8)
ax7.set_title("Volume vs Condition (log scale)", fontsize=12, fontweight="bold")
ax7.set_ylabel("Volume (cu yd, log)")

plt.savefig("/home/user/MLTUNNELS/tunnel_ml_report.png", dpi=150, bbox_inches="tight")
print("\nVisualization saved to tunnel_ml_report.png")

# ── 9. Summary Table ──────────────────────────────────────────────────────────
print("\n=== FINAL SUMMARY ===")
summary = pd.DataFrame({
    "Model": model_names,
    "CV F1 (mean)": [f"{results[n]['cv_mean']:.4f}" for n in model_names],
    "CV F1 (std)": [f"{results[n]['cv_std']:.4f}" for n in model_names],
    "Test Accuracy": [f"{results[n]['test_acc']:.4f}" for n in model_names],
    "Test F1": [f"{results[n]['test_f1']:.4f}" for n in model_names],
})
print(summary.to_string(index=False))

if fi is not None:
    print(f"\nTop 10 Most Important Features ({best_name}):")
    for feat, imp in fi.head(10).items():
        print(f"  {feat:<45} {imp:.4f}")

print("\nDone.")
