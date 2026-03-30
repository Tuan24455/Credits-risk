from __future__ import annotations


def evaluate_policy_rules(raw_data: dict) -> dict:
    reasons = []
    notes = []
    decision = "PASS"

    annual_inc = float(raw_data["annual_inc"])
    dti = float(raw_data["dti"])
    loan_amnt = float(raw_data["loan_amnt"])
    lti = loan_amnt / annual_inc if annual_inc > 0 else 0.0
    term_60 = "60" in str(raw_data["term"])
    purpose = raw_data["purpose"]
    verification_status = raw_data["verification_status"]
    home_ownership = raw_data["home_ownership"]

    if raw_data["num_tl_90g_dpd_24m"] > 0:
        notes.append("Có tín hiệu quá hạn 90+ ngày trong 24 tháng gần đây, cần chú ý thêm khi thẩm định.")

    if raw_data["pub_rec"] > 0:
        notes.append("Có public record tiêu cực, nên kiểm tra hồ sơ kỹ hơn.")

    # --- Income ---
    if annual_inc < 25000:
        if decision != "REJECT":
            decision = "MANUAL_REVIEW"
        reasons.append("Thu nhập dưới ngưỡng tối thiểu 25,000 USD/năm.")

    # --- DTI ---
    if dti > 45:
        if decision != "REJECT":
            decision = "MANUAL_REVIEW"
        reasons.append("DTI vượt 45%, mức rủi ro cao.")
    elif dti > 35:
        notes.append("DTI nằm trong vùng rủi ro trung bình-cao (35%-45%).")

    if dti > 40 and verification_status != "Verified":
        if decision != "REJECT":
            decision = "MANUAL_REVIEW"
        reasons.append("DTI > 40 nhưng thu nhập chưa được Verified.")

    # --- LTI (Loan-to-Income) ---
    if lti >= 1.5:
        decision = "REJECT"
        reasons.append(f"LTI = {lti:.2f} (≥ 1.5) — Khoản vay vượt quá 150% thu nhập, từ chối tự động.")
    elif lti >= 0.8:
        if decision != "REJECT":
            decision = "MANUAL_REVIEW"
        reasons.append(f"LTI = {lti:.2f} (0.8–1.5) — Khoản vay chiếm 80%-150% thu nhập, cần thẩm định thủ công.")
    elif lti > 0.30:
        notes.append(f"LTI = {lti:.2f} — Tỷ lệ loan-to-income vượt 30%, lưu ý.")

    # --- Term ---
    if term_60:
        notes.append("Kỳ hạn 60 tháng thường có rủi ro cao hơn 36 tháng.")

    # --- Purpose ---
    if purpose in {"credit_card", "debt_consolidation"}:
        notes.append(f"Mục đích vay '{purpose}' thuộc nhóm được ưu tiên theo business rule.")

    if purpose in {"small_business", "medical"}:
        if decision != "REJECT":
            decision = "MANUAL_REVIEW"
        reasons.append(f"Mục đích vay '{purpose}' thuộc nhóm rủi ro cao.")

    # --- Verification ---
    if verification_status == "Not Verified":
        notes.append("Hồ sơ chưa xác minh thu nhập, nên cân nhắc penalty rate.")

    # --- Home ownership ---
    if home_ownership in {"OTHER", "NONE"}:
        if decision != "REJECT":
            decision = "MANUAL_REVIEW"
        reasons.append("Tình trạng sở hữu nhà thuộc nhóm rủi ro cao.")
    elif home_ownership == "RENT":
        notes.append("Khách hàng đang thuê nhà, rủi ro trung bình so với OWN/MORTGAGE.")
    else:
        notes.append("Tình trạng sở hữu nhà thuộc nhóm ưu tiên cao hơn về độ ổn định.")

    if not reasons:
        reasons.append("Không có business rule nghiêm trọng nào bị vi phạm.")

    return {
        "decision": decision,
        "reasons": reasons,
        "notes": notes,
        "lti": lti,
    }
