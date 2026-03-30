from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


NUMERIC_FILL_VALUES = {
    "pub_rec": 0.0,
    "num_tl_90g_dpd_24m": 0.0,
    "annual_inc": 65000.0,
    "dti": 17.62,
    "loan_amnt": 12000.0,
    "tot_cur_bal": 80283.0,
}

CATEGORICAL_FILL_VALUES = {
    "term": " 36 months",
    "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation",
    "verification_status": "Source Verified",
}

HOME_KEEP = {"MORTGAGE", "RENT", "OWN"}
PURPOSE_KEEP = {"debt_consolidation", "credit_card", "home_improvement"}
ALLOWED_HOME_VALUES = {"MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"}
ALLOWED_PURPOSE_VALUES = {
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "small_business",
    "medical",
    "other",
    "major_purchase",
    "car",
    "moving",
    "vacation",
    "house",
    "wedding",
    "renewable_energy",
    "educational",
}
ALLOWED_VERIFICATION_VALUES = {"Not Verified", "Source Verified", "Verified"}
BASE_DIR = Path(__file__).resolve().parent
SCALER_PATH = BASE_DIR / "deploy_outputs" / "scaler.joblib"


def load_scaler():
    if SCALER_PATH.exists():
        try:
            return joblib.load(SCALER_PATH)
        except Exception:
            return None
    return None


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return float(value)


def canonical_term(value: Any) -> str:
    if value is None:
        return CATEGORICAL_FILL_VALUES["term"]
    text = str(value).strip()
    if text in {"36", "36 months"}:
        return " 36 months"
    if text in {"60", "60 months"}:
        return " 60 months"
    return CATEGORICAL_FILL_VALUES["term"]


def validate_and_fill_raw_input(raw_data: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    errors: dict[str, str] = {}
    notes: list[str] = []
    filled: dict[str, Any] = {}

    numeric_fields = [
        "pub_rec",
        "num_tl_90g_dpd_24m",
        "annual_inc",
        "dti",
        "loan_amnt",
        "tot_cur_bal",
    ]

    for field in numeric_fields:
        try:
            parsed = _parse_optional_float(raw_data.get(field))
        except (TypeError, ValueError):
            parsed = None
            errors[field] = "Gia tri khong hop le."
            continue

        if parsed is None:
            parsed = NUMERIC_FILL_VALUES[field]
            notes.append(f"{field} trong, da bu theo median train = {parsed}.")

        if field in {"pub_rec", "num_tl_90g_dpd_24m", "tot_cur_bal", "dti"} and parsed < 0:
            errors[field] = "Gia tri phai >= 0."
        if field == "annual_inc" and parsed <= 0:
            errors[field] = "annual_inc phai > 0."
        if field == "loan_amnt" and parsed <= 0:
            errors[field] = "loan_amnt phai > 0."
        if field in {"pub_rec", "num_tl_90g_dpd_24m"}:
            parsed = int(parsed)

        filled[field] = parsed

    term = raw_data.get("term")
    if term in (None, ""):
        notes.append(f"term trong, da bu theo mode train = {CATEGORICAL_FILL_VALUES['term'].strip()}.")
    filled["term"] = canonical_term(term)

    home_ownership = raw_data.get("home_ownership") or CATEGORICAL_FILL_VALUES["home_ownership"]
    if home_ownership not in ALLOWED_HOME_VALUES:
        errors["home_ownership"] = "Gia tri home_ownership khong hop le."
    filled["home_ownership"] = str(home_ownership)

    purpose = raw_data.get("purpose") or CATEGORICAL_FILL_VALUES["purpose"]
    if purpose not in ALLOWED_PURPOSE_VALUES:
        errors["purpose"] = "Gia tri purpose khong hop le."
    filled["purpose"] = str(purpose)

    verification_status = raw_data.get("verification_status") or CATEGORICAL_FILL_VALUES["verification_status"]
    if verification_status not in ALLOWED_VERIFICATION_VALUES:
        errors["verification_status"] = "Gia tri verification_status khong hop le."
    filled["verification_status"] = str(verification_status)

    return filled, notes, errors


def preprocess_raw_input(raw_data: dict[str, Any], feature_names: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    filled = dict(raw_data)
    purpose_val = filled["purpose"] if filled["purpose"] in PURPOSE_KEEP else "OTHER"
    home_val = filled["home_ownership"] if filled["home_ownership"] in HOME_KEEP else "OTHER"
    term_months = 60 if "60" in str(filled["term"]) else 36

    features = {
        "pub_rec": float(filled["pub_rec"]),
        "num_tl_90g_dpd_24m": float(filled["num_tl_90g_dpd_24m"]),
        "annual_inc": float(np.log1p(filled["annual_inc"])),
        "dti": float(filled["dti"]),
        "loan_amnt": float(filled["loan_amnt"]),
        "tot_cur_bal": float(np.log1p(filled["tot_cur_bal"])),
        "LTI": float(filled["loan_amnt"] / filled["annual_inc"]) if filled["annual_inc"] > 0 else 0.0,
        "flag_high_dti": int(filled["dti"] > 35),
        "flag_very_high_dti": int(filled["dti"] > 45),
        "flag_any_derog": int(filled["pub_rec"] > 0),
        "flag_recent_dpd": int(filled["num_tl_90g_dpd_24m"] > 0),
        "term_60": int(term_months == 60),
        "home_ownership_OTHER": int(home_val == "OTHER"),
        "home_ownership_OWN": int(home_val == "OWN"),
        "home_ownership_RENT": int(home_val == "RENT"),
        "purpose_credit_card": int(purpose_val == "credit_card"),
        "purpose_debt_consolidation": int(purpose_val == "debt_consolidation"),
        "purpose_home_improvement": int(purpose_val == "home_improvement"),
        "verification_status_Source Verified": int(
            filled["verification_status"] == "Source Verified"
        ),
        "verification_status_Verified": int(filled["verification_status"] == "Verified"),
    }

    transformed = {feature: float(features.get(feature, 0.0)) for feature in feature_names}
    transformed_df = pd.DataFrame([transformed])

    scaler = load_scaler()
    scale_cols = [col for col in ["pub_rec", "num_tl_90g_dpd_24m", "annual_inc", "dti", "loan_amnt", "tot_cur_bal", "LTI"] if col in transformed_df.columns]
    if scaler is not None and scale_cols:
        transformed_df[scale_cols] = scaler.transform(transformed_df[scale_cols])

    return transformed_df, features
