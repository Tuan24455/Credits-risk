from __future__ import annotations

import json
import mimetypes
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from policy_rules import evaluate_policy_rules
from preprocessor import preprocess_raw_input, validate_and_fill_raw_input

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "deploy_outputs"
INDEX_PATH = BASE_DIR / "index.html"


@lru_cache(maxsize=1)
def load_artifacts():
    feature_names = joblib.load(OUTPUT_DIR / "feature_names.joblib")
    meta = {}
    meta_path = OUTPUT_DIR / "model_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    logistic_model = None
    logistic_error = None
    logistic_path = OUTPUT_DIR / "logistic_model.joblib"
    if logistic_path.exists():
        try:
            candidate = joblib.load(logistic_path)
            expected = list(getattr(candidate, "feature_names_in_", []))
            if expected and expected != feature_names:
                raise ValueError("Logistic artifact khong khop feature schema hien tai.")
            logistic_model = candidate
        except Exception as exc:
            logistic_error = str(exc)

    xgb_model = None
    xgb_model_no_term = None
    xgb_error = None
    try:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier()
        xgb_model.load_model(str(OUTPUT_DIR / "xgboost_model.json"))
        no_term_path = OUTPUT_DIR / "xgboost_model_no_term.json"
        if no_term_path.exists():
            xgb_model_no_term = XGBClassifier()
            xgb_model_no_term.load_model(str(no_term_path))
    except Exception as exc:
        xgb_error = str(exc)

    feature_importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")
    effect_summary = pd.read_csv(OUTPUT_DIR / "feature_effect_summary.csv")
    return (
        feature_names,
        logistic_model,
        logistic_error,
        xgb_model,
        xgb_model_no_term,
        xgb_error,
        feature_importance,
        effect_summary,
        meta,
    )


def get_feature_importance_payload():
    _, _, _, _, _, _, fi, _, _ = load_artifacts()
    top = fi.head(8).copy()
    top["level"] = pd.cut(
        top["importance_pct"],
        bins=[-1, 0.02, 0.05, 0.15, 1],
        labels=["low", "medium", "high", "very-high"],
    ).astype(str)
    top["label"] = top["level"].map(
        {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
            "very-high": "Very High",
        }
    )
    max_importance = max(float(top["importance"].max()), 1e-9)
    top["value"] = (top["importance"] / max_importance * 100).round().astype(int)
    return top[["feature", "level", "label", "value"]].rename(
        columns={"feature": "name"}
    ).to_dict(orient="records")


RAW_FIELD_MAP = {
    "pub_rec": ["pub_rec", "flag_any_derog"],
    "num_tl_90g_dpd_24m": ["num_tl_90g_dpd_24m", "flag_recent_dpd"],
    "annual_inc": ["annual_inc"],
    "dti": ["dti", "flag_high_dti", "flag_very_high_dti"],
    "loan_amnt": ["loan_amnt", "LTI"],
    "term": ["term_months", "term_60"],
    "tot_cur_bal": ["tot_cur_bal"],
    "home_ownership": [
        "home_ownership_OTHER",
        "home_ownership_OWN",
        "home_ownership_RENT",
    ],
    "purpose": [
        "purpose_credit_card",
        "purpose_debt_consolidation",
        "purpose_home_improvement",
    ],
    "verification_status": [
        "verification_status_Source Verified",
        "verification_status_Verified",
    ],
}


def _raw_global_weight_map() -> dict[str, float]:
    _, _, _, _, _, _, fi, _, _ = load_artifacts()
    feature_weight = dict(zip(fi["feature"], fi["importance"]))
    raw_weights = {}
    for raw_field, processed_fields in RAW_FIELD_MAP.items():
        raw_weights[raw_field] = float(
            sum(feature_weight.get(processed_field, 0.0) for processed_field in processed_fields)
        )
    return raw_weights


FEATURE_LABELS = {
    "pub_rec": "Public records",
    "num_tl_90g_dpd_24m": "90+ ngày quá hạn",
    "annual_inc": "Thu nhập năm",
    "dti": "DTI",
    "loan_amnt": "Số tiền vay",
    "tot_cur_bal": "Tổng dư nợ hiện tại",
    "LTI": "Loan / Income",
    "flag_high_dti": "Cờ DTI cao",
    "flag_very_high_dti": "Cờ DTI rất cao",
    "flag_any_derog": "Cờ public record xấu",
    "flag_recent_dpd": "Cờ quá hạn gần đây",
    "term_months": "Kỳ hạn (tháng)",
    "term_60": "Cờ kỳ hạn 60 tháng",
    "home_ownership_OTHER": "Nhà ở: OTHER",
    "home_ownership_OWN": "Nhà ở: OWN",
    "home_ownership_RENT": "Nhà ở: RENT",
    "purpose_credit_card": "Mục đích: Credit card",
    "purpose_debt_consolidation": "Mục đích: Debt consolidation",
    "purpose_home_improvement": "Mục đích: Home improvement",
    "verification_status_Source Verified": "Xác minh: Source Verified",
    "verification_status_Verified": "Xác minh: Verified",
}


def build_global_model_feature_summary() -> list[dict]:
    _, _, _, _, _, _, fi, _, _ = load_artifacts()
    total = float(fi["importance"].sum()) or 1.0
    rows = []
    for _, record in fi.iterrows():
        field = str(record["feature"])
        pct = float(record["importance"]) / total * 100
        if pct >= 18:
            level = "Rất cao"
        elif pct >= 10:
            level = "Cao"
        elif pct >= 5:
            level = "Trung bình"
        else:
            level = "Thấp"
        rows.append(
            {
                "field": field,
                "label_vi": FEATURE_LABELS.get(field, field),
                "impact_pct": round(pct, 2),
                "level": level,
            }
        )
    rows.sort(key=lambda item: item["impact_pct"], reverse=True)
    return rows


def build_raw_feature_impacts(raw_filled: dict, processed_features: dict) -> list[dict]:
    global_weights = _raw_global_weight_map()
    annual_inc = float(raw_filled["annual_inc"])
    loan_amnt = float(raw_filled["loan_amnt"])
    dti = float(raw_filled["dti"])
    lti = loan_amnt / annual_inc if annual_inc > 0 else 0.0
    tot_cur_bal = float(raw_filled["tot_cur_bal"])
    balance_to_income = tot_cur_bal / annual_inc if annual_inc > 0 else 0.0

    field_status = {
        "pub_rec": (
            min(0.55, 0.15 + 0.15 * float(raw_filled["pub_rec"])) if raw_filled["pub_rec"] > 0 else 0.08,
            "Tăng rủi ro" if raw_filled["pub_rec"] > 0 else "Hỗ trợ",
            "Có public record tiêu cực, nhưng hiện chỉ dùng như tín hiệu hỗ trợ." if raw_filled["pub_rec"] > 0 else "Không có public record tiêu cực.",
        ),
        "num_tl_90g_dpd_24m": (
            0.6 if raw_filled["num_tl_90g_dpd_24m"] >= 2 else 0.35 if raw_filled["num_tl_90g_dpd_24m"] == 1 else 0.08,
            "Tăng rủi ro" if raw_filled["num_tl_90g_dpd_24m"] > 0 else "Hỗ trợ",
            "Có từ 2 tài khoản quá hạn 90+ ngày." if raw_filled["num_tl_90g_dpd_24m"] >= 2 else "Có 1 tài khoản quá hạn 90+ ngày, cần xem xét thêm." if raw_filled["num_tl_90g_dpd_24m"] == 1 else "Không có quá hạn 90+ ngày.",
        ),
        "annual_inc": (
            1.0 if annual_inc < 25000 else 0.7 if annual_inc < 45000 else 0.35 if annual_inc < 80000 else 0.12,
            "Tăng rủi ro" if annual_inc < 45000 else "Giảm rủi ro",
            "Thu nhập thấp hơn ngưỡng ưu tiên." if annual_inc < 45000 else "Thu nhập khá ổn định so với mặt bằng train.",
        ),
        "dti": (
            1.0 if dti > 45 else 0.8 if dti > 35 else 0.45 if dti > 25 else 0.15,
            "Tăng rủi ro" if dti > 25 else "Giảm rủi ro",
            "DTI cao." if dti > 35 else "DTI đang ở mức an toàn." if dti <= 25 else "DTI ở mức trung bình.",
        ),
        "loan_amnt": (
            1.0 if lti > 0.5 else 0.75 if lti > 0.3 else 0.35 if loan_amnt > 20000 else 0.15,
            "Tăng rủi ro" if lti > 0.3 or loan_amnt > 20000 else "Trung tính",
            "Khoản vay lớn so với thu nhập." if lti > 0.3 else "Khoản vay ở mức trung bình.",
        ),
        "term": (
            1.0 if processed_features["term_60"] == 1 else 0.25,
            "Tăng rủi ro" if processed_features["term_60"] == 1 else "Giảm rủi ro",
            "Kỳ hạn 60 tháng." if processed_features["term_60"] == 1 else "Kỳ hạn 36 tháng.",
        ),
        "tot_cur_bal": (
            0.9 if balance_to_income > 1.5 else 0.6 if balance_to_income > 1.0 else 0.35 if balance_to_income > 0.5 else 0.12,
            "Tăng rủi ro" if balance_to_income > 1.0 else "Trung tính" if balance_to_income > 0.5 else "Giảm rủi ro",
            "Tổng dư nợ cao so với thu nhập." if balance_to_income > 1.0 else "Tổng dư nợ ở mức chấp nhận được.",
        ),
        "home_ownership": (
            0.9 if raw_filled["home_ownership"] in {"OTHER", "NONE"} else 0.7 if raw_filled["home_ownership"] == "RENT" else 0.12,
            "Tăng rủi ro" if raw_filled["home_ownership"] in {"OTHER", "NONE", "RENT"} else "Giảm rủi ro",
            f"Trạng thái nhà ở hiện tại: {raw_filled['home_ownership']}.",
        ),
        "purpose": (
            0.85 if raw_filled["purpose"] in {"small_business", "medical"} else 0.18 if raw_filled["purpose"] in {"credit_card", "debt_consolidation"} else 0.4,
            "Tăng rủi ro" if raw_filled["purpose"] in {"small_business", "medical"} else "Giảm rủi ro" if raw_filled["purpose"] in {"credit_card", "debt_consolidation"} else "Trung tính",
            f"Mục đích vay: {raw_filled['purpose']}.",
        ),
        "verification_status": (
            0.75 if raw_filled["verification_status"] == "Not Verified" else 0.4 if raw_filled["verification_status"] == "Source Verified" else 0.12,
            "Tăng rủi ro" if raw_filled["verification_status"] == "Not Verified" else "Trung tính" if raw_filled["verification_status"] == "Source Verified" else "Giảm rủi ro",
            f"Trạng thái xác minh: {raw_filled['verification_status']}.",
        ),
    }

    impact_rows = []
    for raw_field in RAW_FIELD_MAP:
        severity, effect, detail = field_status[raw_field]
        base_weight = global_weights.get(raw_field, 0.01)
        impact_score = max(base_weight * (0.25 + severity), 1e-6)
        impact_rows.append(
            {
                "field": raw_field,
                "current_value": raw_filled[raw_field],
                "effect": effect,
                "detail": detail,
                "impact_score": impact_score,
            }
        )

    total = sum(row["impact_score"] for row in impact_rows)
    for row in impact_rows:
        pct = row["impact_score"] / total * 100 if total else 0.0
        row["impact_pct"] = round(pct, 2)
        if pct >= 18:
            row["level"] = "very-high"
            row["label"] = "Very High"
        elif pct >= 11:
            row["level"] = "high"
            row["label"] = "High"
        elif pct >= 7:
            row["level"] = "medium"
            row["label"] = "Medium"
        else:
            row["level"] = "low"
            row["label"] = "Low"
        row["value"] = max(3, int(round(pct)))

    impact_rows.sort(key=lambda item: item["impact_pct"], reverse=True)
    return impact_rows


def build_reason_signals(raw_filled: dict, processed_features: dict) -> list[dict]:
    signals = []

    def add_signal(feature: str, title: str, detail: str, severity: str):
        signals.append(
            {
                "feature": feature,
                "title": title,
                "detail": detail,
                "severity": severity,
            }
        )

    if raw_filled["num_tl_90g_dpd_24m"] >= 2:
        add_signal(
            "num_tl_90g_dpd_24m",
            "Quá hạn nghiêm trọng",
            "Có từ 2 tài khoản quá hạn trên 90 ngày trong 24 tháng gần đây, đây là tín hiệu phụ trợ cần chú ý.",
            "high",
        )
    elif raw_filled["num_tl_90g_dpd_24m"] == 1:
        add_signal(
            "num_tl_90g_dpd_24m",
            "Có quá hạn 90+ ngày",
            "Có 1 tài khoản quá hạn trên 90 ngày trong 24 tháng gần đây, cần thẩm định thêm.",
            "medium",
        )
    if raw_filled["pub_rec"] > 0:
        add_signal(
            "pub_rec",
            "Hồ sơ công khai tiêu cực",
            "Có public record tiêu cực, nên xem đây là tín hiệu hỗ trợ khi đánh giá.",
            "medium",
        )
    if raw_filled["dti"] > 45:
        add_signal(
            "dti",
            "DTI quá cao",
            "DTI vượt 45%, khả năng chịu đựng tài chính đang yếu.",
            "very-high",
        )
    elif raw_filled["dti"] > 35:
        add_signal(
            "dti",
            "DTI trung bình-cao",
            "DTI nằm trong nhóm 35%-45%, cần xem xét kỹ bổ sung.",
            "high",
        )
    if processed_features["LTI"] > 0.30:
        add_signal(
            "LTI",
            "Loan-to-income cao",
            "Tỷ lệ khoản vay/thu nhập vượt 30%.",
            "high",
        )
    if processed_features["term_60"] == 1:
        add_signal(
            "term_60",
            "Kỳ hạn 60 tháng",
            "Kỳ hạn dài hơn thường có rủi ro default cao hơn.",
            "medium",
        )
    if raw_filled["verification_status"] == "Not Verified":
        add_signal(
            "verification_status",
            "Chưa xác minh thu nhập",
            "Hồ sơ không có xác minh thu nhập, độ tin cậy thấp hơn.",
            "medium",
        )
    if raw_filled["home_ownership"] == "RENT":
        add_signal(
            "home_ownership_RENT",
            "Đang thuê nhà",
            "Nhóm RENT thường có rủi ro cao hơn OWN/MORTGAGE.",
            "medium",
        )
    if raw_filled["purpose"] in {"small_business", "medical"}:
        add_signal(
            "purpose",
            "Mục đích vay rủi ro cao",
            f"Mục đích vay '{raw_filled['purpose']}' thuộc nhóm cần thận trọng.",
            "high",
        )
    if raw_filled["annual_inc"] < 25000:
        add_signal(
            "annual_inc",
            "Thu nhập thấp",
            "Thu nhập dưới ngưỡng tối thiểu 25,000 USD/năm.",
            "high",
        )
    if raw_filled["tot_cur_bal"] > raw_filled["annual_inc"] * 1.5:
        add_signal(
            "tot_cur_bal",
            "Tổng dư nợ cao",
            "Tổng dư nợ hiện tại cao so với thu nhập, có dấu hiệu over-leveraged.",
            "medium",
        )

    return signals[:6]


def score_to_band(score: float, policy_decision: str) -> str:
    if policy_decision == "REJECT":
        return "REJECT"
    if score >= 0.65:
        return "HIGH RISK"
    if score >= 0.35:
        return "MEDIUM RISK"
    return "LOW RISK"


def to_binary_decision(score: float, policy_decision: str) -> tuple[str, str]:
    has_risk = score >= 0.5
    if has_risk:
        return "Có vỡ nợ", "Có dấu hiệu rủi ro cần lưu ý"
    return "Không vỡ nợ", "Chưa thấy dấu hiệu rủi ro nổi bật"


def run_prediction(raw_payload: dict):
    (
        feature_names,
        logistic_model,
        logistic_error,
        xgb_model,
        xgb_model_no_term,
        xgb_error,
        _,
        _,
        meta,
    ) = load_artifacts()

    raw_filled, imputation_notes, validation_errors = validate_and_fill_raw_input(raw_payload)
    if validation_errors:
        return {
            "ok": False,
            "errors": validation_errors,
        }

    X_input, processed_features = preprocess_raw_input(raw_filled, feature_names)
    policy = evaluate_policy_rules(raw_filled)

    logistic_prob = None
    if logistic_model is not None:
        logistic_prob = float(logistic_model.predict_proba(X_input)[0, 1])
    xgboost_prob = None
    if xgb_model is not None:
        xgboost_prob = float(xgb_model.predict_proba(X_input)[0, 1])
        if xgb_model_no_term is not None:
            X_input_no_term = X_input.drop(columns=["term_60"], errors="ignore")
            xgboost_prob_no_term = float(xgb_model_no_term.predict_proba(X_input_no_term)[0, 1])
            blend_weight = float(meta.get("blend_weight_with_term", 1.0))
            xgboost_prob = blend_weight * xgboost_prob + (1 - blend_weight) * xgboost_prob_no_term

    final_score = xgboost_prob if xgboost_prob is not None else logistic_prob
    if final_score is None:
        raise RuntimeError("Khong co model kha dung de du doan.")
    risk_band = score_to_band(final_score, policy["decision"])
    binary_label, binary_message = to_binary_decision(final_score, policy["decision"])

    reasons = build_reason_signals(raw_filled, processed_features)
    if not reasons:
        reasons = [
            {
                "feature": "overall_profile",
                "title": "Ho so tuong doi on",
                "detail": "Khong co tin hieu rui ro noi bat theo business rule chinh.",
                "severity": "low",
            }
        ]

    return {
        "ok": True,
        "raw_input": raw_filled,
        "policy": policy,
        "scores": {
            "final_score": final_score,
            "logistic_regression": logistic_prob,
            "xgboost": xgboost_prob,
        },
        "final_binary_label": binary_label,
        "final_binary_message": binary_message,
        "risk_band": risk_band,
        "imputation_notes": imputation_notes,
        "reasons": reasons,
        "raw_feature_impacts": build_raw_feature_impacts(raw_filled, processed_features),
        "global_feature_summary": build_global_model_feature_summary(),
        "processed_features": processed_features,
        "logistic_available": logistic_model is not None,
        "logistic_error": logistic_error,
        "xgboost_available": xgb_model is not None,
        "xgboost_error": xgb_error,
    }


class CreditRiskHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_file(INDEX_PATH)
            return
        if parsed.path == "/metadata":
            self._send_json({"feature_importance": get_feature_importance_payload()})
            return

        target = (BASE_DIR / parsed.path.lstrip("/")).resolve()
        if BASE_DIR not in target.parents and target != BASE_DIR:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        self._send_file(target)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "errors": {"request": "JSON khong hop le."}}, status=HTTPStatus.BAD_REQUEST)
            return

        result = run_prediction(payload)
        status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
        self._send_json(result, status=status)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), CreditRiskHandler)
    print("Credit risk app dang chay tai http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
