import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================
# 1. Config
# =========================
TRAIN_PATH = "credit_risk_train.csv"
TEST_PATH = "credit_risk_test.csv"
TARGET_COL = "label"
REDUNDANT_TERM_FEATURES = ["term_months"]

OUTPUT_DIR = Path("deploy_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# 2. Helpers
# =========================
def export_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    return df.astype(np.float32)


def align_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train_aligned, X_test_aligned = X_train.align(
        X_test, join="outer", axis=1, fill_value=0
    )
    return X_train_aligned, X_test_aligned


def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC_AUC": np.nan,
        "PR_AUC": np.nan,
        "Recall_BadLoan": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision_BadLoan": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1_BadLoan": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "KS_Statistic": np.nan,
        "Gini": np.nan,
    }

    if y_proba is not None:
        try:
            prob_1 = y_proba[:, 1]
            result["ROC_AUC"] = roc_auc_score(y_true, prob_1)
            result["PR_AUC"] = average_precision_score(y_true, prob_1)
            fpr, tpr, _ = roc_curve(y_true, prob_1)
            result["KS_Statistic"] = float(np.max(tpr - fpr))
            result["Gini"] = float(2 * result["ROC_AUC"] - 1)
        except Exception:
            pass

    print(f"\n{'=' * 15} {model_name} {'=' * 15}")
    for k, v in result.items():
        if k != "Model":
            print(f"{k}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return result


def classify_score(metric_name, value):
    if pd.isna(value):
        return "N/A"

    thresholds = {
        "ROC_AUC": (0.70, 0.75, 0.85),
        "PR_AUC": (0.40, 0.55, 0.70),
        "Recall_BadLoan": (0.60, 0.70, 0.80),
        "Precision_BadLoan": (0.40, 0.55, 0.70),
        "F1_BadLoan": (0.45, 0.55, 0.70),
        "KS_Statistic": (0.30, 0.40, 0.55),
        "Gini": (0.40, 0.50, 0.70),
    }
    if metric_name not in thresholds:
        return "N/A"

    acceptable, good, excellent = thresholds[metric_name]
    if value > excellent:
        return "Xuat sac"
    if value >= good:
        return "Tot"
    if value >= acceptable:
        return "Tam chap nhan"
    return "Duoi nguong"


def print_scorecard(summary_df: pd.DataFrame):
    row = summary_df.iloc[0]
    print(f"Model: {row['Model']}")
    metrics_to_show = [
        ("Accuracy", "Accuracy", None),
        ("ROC_AUC", "AUC-ROC", "ROC_AUC_Rating"),
        ("PR_AUC", "AUC-PR", "PR_AUC_Rating"),
        ("Recall_BadLoan", "Recall (BadLoan)", "Recall_BadLoan_Rating"),
        ("Precision_BadLoan", "Precision (BadLoan)", "Precision_BadLoan_Rating"),
        ("F1_BadLoan", "F1 (BadLoan)", "F1_BadLoan_Rating"),
        ("KS_Statistic", "KS Statistic", "KS_Statistic_Rating"),
        ("Gini", "Gini", "Gini_Rating"),
    ]

    print("\n================ Scorecard ================\n")
    print(f"{'Chi so':<24} {'Gia tri':>12} {'Xep loai':>18}")
    print(f"{'-' * 24} {'-' * 12} {'-' * 18}")
    for metric_key, label, rating_key in metrics_to_show:
        value = row[metric_key]
        rating = row[rating_key] if rating_key else ""
        print(f"{label:<24} {value:>12.6f} {rating:>18}")


def build_feature_importance_df(feature_names, importances):
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False, ignore_index=True)
    total = fi["importance"].sum()
    if total > 0:
        fi["importance_pct"] = fi["importance"] / total
        fi["cumulative_importance_pct"] = fi["importance_pct"].cumsum()
    else:
        fi["importance_pct"] = 0.0
        fi["cumulative_importance_pct"] = 0.0
    return fi


def find_best_threshold_for_accuracy(y_true, y_prob):
    best_threshold = 0.5
    best_accuracy = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)
    return best_threshold, float(best_accuracy)


def build_feature_effect_summary(X_df, y_series, feature_importance_df):
    y_series = pd.Series(y_series).reset_index(drop=True)
    rows = []
    for feature in X_df.columns:
        col = pd.to_numeric(X_df[feature], errors="coerce")
        class0 = col[y_series == 0]
        class1 = col[y_series == 1]
        mean0 = float(class0.mean())
        mean1 = float(class1.mean())
        diff = mean1 - mean0
        if diff > 0:
            direction = "higher_value_more_risky"
        elif diff < 0:
            direction = "lower_value_more_risky"
        else:
            direction = "neutral"
        rows.append({
            "feature": feature,
            "mean_class_0": mean0,
            "mean_class_1": mean1,
            "difference_1_minus_0": float(diff),
            "direction_hint": direction,
        })

    effect_df = pd.DataFrame(rows)
    effect_df = effect_df.merge(
        feature_importance_df[["feature", "importance", "importance_pct"]],
        on="feature",
        how="left",
    )
    effect_df = effect_df.sort_values(
        ["importance", "difference_1_minus_0"],
        ascending=[False, False],
        ignore_index=True,
    )
    return effect_df


def train_xgb_model(X_fit_scaled, y_fit, params):
    try:
        model = XGBClassifier(**params)
        model.fit(X_fit_scaled, y_fit)
        return model, "cuda"
    except Exception as e:
        print(f"XGBoost CUDA không chạy được, fallback về CPU. Lý do: {e}")
        cpu_params = dict(params)
        cpu_params["device"] = "cpu"
        model = XGBClassifier(**cpu_params)
        model.fit(X_fit_scaled, y_fit)
        return model, "cpu"


def fit_and_apply_scaler(X_fit: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    scale_cols = [
        col
        for col in [
            "pub_rec",
            "num_tl_90g_dpd_24m",
            "annual_inc",
            "dti",
            "loan_amnt",
            "tot_cur_bal",
            "LTI",
        ]
        if col in X_fit.columns
    ]

    scaler = RobustScaler()
    X_fit_scaled = X_fit.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    if scale_cols:
        X_fit_scaled[scale_cols] = scaler.fit_transform(X_fit[scale_cols])
        X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])
        X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_fit_scaled, X_val_scaled, X_test_scaled, scaler, scale_cols


# =========================
# 3. Load data
# =========================
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

train_data.columns = train_data.columns.str.strip()
test_data.columns = test_data.columns.str.strip()

if TARGET_COL not in train_data.columns:
    raise ValueError(f"Không tìm thấy cột nhãn '{TARGET_COL}' trong file train.")
if TARGET_COL not in test_data.columns:
    raise ValueError(f"Không tìm thấy cột nhãn '{TARGET_COL}' trong file test.")

X_train = train_data.drop(columns=[TARGET_COL]).copy()
y_train = train_data[TARGET_COL].copy()

X_test = test_data.drop(columns=[TARGET_COL]).copy()
y_test = test_data[TARGET_COL].copy()

# optional state column is intentionally excluded in this setup
X_train = X_train.drop(columns=["addr_state"], errors="ignore")
X_test = X_test.drop(columns=["addr_state"], errors="ignore")

# =========================
# 4. Clean labels + features
# =========================
y_train = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)
y_test = pd.to_numeric(y_test, errors="coerce").fillna(0).astype(int)

if len(np.unique(y_train)) != 2:
    raise ValueError("Dữ liệu train phải là binary classification (2 class).")

X_train, X_test = align_features(X_train, X_test)
X_train = ensure_numeric(X_train)
X_test = ensure_numeric(X_test)

X_train = X_train.drop(columns=REDUNDANT_TERM_FEATURES, errors="ignore")
X_test = X_test.drop(columns=REDUNDANT_TERM_FEATURES, errors="ignore")
print(f"Đã loại khỏi training các cột trùng tín hiệu: {REDUNDANT_TERM_FEATURES}")

feature_names = X_train.columns.tolist()
print("Số lượng features:", len(feature_names))

# =========================
# 5. Imbalance handling
# =========================
class_counts = y_train.value_counts().sort_index()
neg_count = int(class_counts.get(0, 0))
pos_count = int(class_counts.get(1, 0))
scale_pos_weight = neg_count / max(pos_count, 1)

print("\nPhân bố class train gốc:")
print(class_counts)
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42,
)

X_fit_scaled, X_val_scaled, X_test_scaled, scaler, scaled_columns = fit_and_apply_scaler(
    X_fit, X_val, X_test
)
print(f"Scaled columns: {scaled_columns}")

# =========================
# 6. Train Logistic Regression
# =========================
# print("\nTraining Logistic Regression...")
# log_train_start = time.perf_counter()
# log_model = LogisticRegression(
#     max_iter=3000,
#     class_weight="balanced",
#     random_state=42,
#     solver="lbfgs",
# )
# log_model.fit(X_fit, y_fit)
# log_train_time = time.perf_counter() - log_train_start

# log_pred_start = time.perf_counter()
# log_probs = log_model.predict_proba(X_test)
# log_preds = (log_probs[:, 1] >= 0.5).astype(int)
# log_pred_time = time.perf_counter() - log_pred_start
# log_total_time = log_train_time + log_pred_time

# print(f"Logistic train time  : {log_train_time:.2f} sec")
# print(f"Logistic predict time: {log_pred_time:.2f} sec")
# print(f"Logistic total time  : {log_total_time:.2f} sec")
# print("Decision threshold   : 0.500")

# log_metrics = evaluate_model(y_test, log_preds, log_probs, "Logistic Regression")

# =========================
# 7. TabTransformer
# =========================
# Tam thoi tat TabTransformer de pipeline gon hon va tap trung vao
# bo feature 5C + Logistic Regression + XGBoost.

# =========================
# 8. Train XGBoost
# =========================
print("\nTraining XGBoost...")
xgb_train_start = time.perf_counter()

xgb_params = dict(
    n_estimators=1600,
    max_depth=2,
    learning_rate=0.010,
    subsample=0.55,
    colsample_bytree=0.30,
    min_child_weight=35,
    reg_lambda=32.0,
    reg_alpha=14.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    device="cuda",
    scale_pos_weight=scale_pos_weight * 0.82,
)
xgb_model, xgb_device_used = train_xgb_model(X_fit_scaled, y_fit, xgb_params)

X_fit_no_term = X_fit_scaled.drop(columns=["term_60"], errors="ignore")
X_val_no_term = X_val_scaled.drop(columns=["term_60"], errors="ignore")
X_test_no_term = X_test_scaled.drop(columns=["term_60"], errors="ignore")
xgb_model_no_term, xgb_no_term_device_used = train_xgb_model(
    X_fit_no_term,
    y_fit,
    xgb_params,
)

xgb_train_time = time.perf_counter() - xgb_train_start

xgb_val_probs = xgb_model.predict_proba(X_val_scaled)[:, 1]
best_threshold, best_val_accuracy = find_best_threshold_for_accuracy(y_val, xgb_val_probs)
print(f"Best validation threshold for accuracy: {best_threshold:.3f}")
print(f"Best validation accuracy: {best_val_accuracy:.6f}")

blend_weight_with_term = 0.50
decision_threshold = 0.48
print(f"Using evaluation threshold: {decision_threshold:.3f}")
print(f"Blend weight with term : {blend_weight_with_term:.2f}")

xgb_pred_start = time.perf_counter()
xgb_probs_with_term = xgb_model.predict_proba(X_test_scaled)
xgb_probs_no_term = xgb_model_no_term.predict_proba(X_test_no_term)
xgb_prob_1 = (
    blend_weight_with_term * xgb_probs_with_term[:, 1]
    + (1 - blend_weight_with_term) * xgb_probs_no_term[:, 1]
)
xgb_probs = np.column_stack([1 - xgb_prob_1, xgb_prob_1])
xgb_preds = (xgb_probs[:, 1] >= decision_threshold).astype(int)
xgb_pred_time = time.perf_counter() - xgb_pred_start
xgb_total_time = xgb_train_time + xgb_pred_time

print(f"XGBoost device       : {xgb_device_used}")
print(f"XGBoost no-term dev  : {xgb_no_term_device_used}")
print(f"XGBoost train time   : {xgb_train_time:.2f} sec")
print(f"XGBoost predict time : {xgb_pred_time:.2f} sec")
print(f"XGBoost total time   : {xgb_total_time:.2f} sec")
print(f"Decision threshold   : {decision_threshold:.3f}")

xgb_metrics = evaluate_model(y_test, xgb_preds, xgb_probs, "XGBoost")

# =========================
# 9. Feature importance
# =========================
with_term_importance = pd.Series(xgb_model.feature_importances_, index=feature_names)
no_term_importance = pd.Series(
    xgb_model_no_term.feature_importances_,
    index=X_fit_no_term.columns.tolist(),
)
blended_importance = (
    with_term_importance.mul(blend_weight_with_term, fill_value=0)
    .add(no_term_importance.mul(1 - blend_weight_with_term, fill_value=0), fill_value=0)
)
feature_importance_df = build_feature_importance_df(
    feature_names=blended_importance.index.tolist(),
    importances=blended_importance.values,
)
feature_effect_df = build_feature_effect_summary(X_train, y_train, feature_importance_df)

selected_features = feature_importance_df[
    feature_importance_df["cumulative_importance_pct"] <= 0.95
]["feature"].tolist()
if len(selected_features) < min(15, len(feature_importance_df)):
    selected_features = feature_importance_df.head(min(15, len(feature_importance_df)))["feature"].tolist()

print("\nTop 15 feature importance:")
print(feature_importance_df.head(15).to_string(index=False))
print("\nTop 15 feature effects:")
print(feature_effect_df.head(15).to_string(index=False))
print("\nFeature được khuyến nghị giữ lại:")
print(selected_features)

# =========================
# 10. Save outputs
# =========================
# joblib.dump(log_model, OUTPUT_DIR / "logistic_model.joblib")
xgb_model.save_model(str(OUTPUT_DIR / "xgboost_model.json"))
xgb_model_no_term.save_model(str(OUTPUT_DIR / "xgboost_model_no_term.json"))
joblib.dump(feature_names, OUTPUT_DIR / "feature_names.joblib")
joblib.dump(selected_features, OUTPUT_DIR / "selected_features.joblib")
joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

meta = {
    "target_col": TARGET_COL,
    "num_features": len(feature_names),
    "num_classes": 2,
    "xgboost_device_used": xgb_device_used,
    "scale_pos_weight": float(scale_pos_weight),
    "decision_threshold": float(decision_threshold),
    "blend_weight_with_term": float(blend_weight_with_term),
    "best_validation_threshold_for_accuracy": float(best_threshold),
    "best_validation_accuracy": float(best_val_accuracy),
    "feature_names_file": "feature_names.joblib",
    "selected_features_file": "selected_features.joblib",
    "scaler_file": "scaler.joblib",
    "scaled_columns": scaled_columns,
    "logistic_model_file": "logistic_model.joblib",
    "xgboost_model_file": "xgboost_model.json",
    "xgboost_model_no_term_file": "xgboost_model_no_term.json",
}
export_json(OUTPUT_DIR / "model_meta.json", meta)

summary = pd.DataFrame([
    # {
    #     "Model": "Logistic Regression",
    #     "Train_time_sec": round(log_train_time, 4),
    #     "Predict_time_sec": round(log_pred_time, 4),
    #     "Total_time_sec": round(log_total_time, 4),
    #     **log_metrics,
    # },
    {
        "Model": "XGBoost",
        "Train_time_sec": round(xgb_train_time, 4),
        "Predict_time_sec": round(xgb_pred_time, 4),
        "Total_time_sec": round(xgb_total_time, 4),
        **xgb_metrics,
    }
])

scorecard_metrics = [
    "ROC_AUC",
    "PR_AUC",
    "Recall_BadLoan",
    "Precision_BadLoan",
    "F1_BadLoan",
    "KS_Statistic",
    "Gini",
]
for metric in scorecard_metrics:
    summary[f"{metric}_Rating"] = summary[metric].apply(lambda v, m=metric: classify_score(m, v))

summary = summary[
    [
        "Model",
        "Train_time_sec",
        "Predict_time_sec",
        "Total_time_sec",
        "Accuracy",
        "ROC_AUC",
        "ROC_AUC_Rating",
        "PR_AUC",
        "PR_AUC_Rating",
        "Recall_BadLoan",
        "Recall_BadLoan_Rating",
        "Precision_BadLoan",
        "Precision_BadLoan_Rating",
        "F1_BadLoan",
        "F1_BadLoan_Rating",
        "KS_Statistic",
        "KS_Statistic_Rating",
        "Gini",
        "Gini_Rating",
    ]
]
summary.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
export_json(OUTPUT_DIR / "model_summary.json", summary.to_dict(orient="records"))
feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
export_json(
    OUTPUT_DIR / "feature_importance.json",
    feature_importance_df.to_dict(orient="records"),
)
export_json(
    OUTPUT_DIR / "selected_features.json",
    {"selected_features": selected_features},
)
feature_effect_df.to_csv(OUTPUT_DIR / "feature_effect_summary.csv", index=False)
export_json(
    OUTPUT_DIR / "feature_effect_summary.json",
    feature_effect_df.to_dict(orient="records"),
)

pred_df = pd.DataFrame({
    "y_true": y_test.reset_index(drop=True),
    # "logistic_pred": log_preds,
    # "logistic_prob_1": log_probs[:, 1],
    "xgboost_pred": xgb_preds,
    "xgboost_prob_1": xgb_probs[:, 1],
})
pred_df.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)

print("\n================ Model Comparison ================\n")
for _, summary_row in summary.iterrows():
    print_scorecard(pd.DataFrame([summary_row]))
    print("")
print("\nĐã xuất file vào thư mục:", OUTPUT_DIR.resolve())
print("- logistic_model.joblib")
print("- xgboost_model.json")
print("- xgboost_model_no_term.json")
print("- feature_names.joblib")
print("- selected_features.joblib")
print("- model_meta.json")
print("- model_summary.csv")
print("- model_summary.json")
print("- model_predictions.csv")
print("- feature_importance.csv")
print("- feature_importance.json")
print("- selected_features.json")
print("- feature_effect_summary.csv")
print("- feature_effect_summary.json")
