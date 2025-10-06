# =================== Inference Function ==================
# This is custom predict funtion that outputs (prediction + reliability_score)

# ======== Input Format =========
# 1. model from s3, 
# 2. feature vector / input vector (this should have consitent ordering as used in training)
# ===============================

def predict_with_reliability(model, x):
    x = np.atleast_2d(x)

    # Prediction from the whole ensemble
    y_pred = model.predict(x)[0]

    # Predictions from each individual tree
    all_preds = np.array([est.predict(x)[0] for est in model.estimators_])

    # Std deviation = uncertainty
    std = all_preds.std()

    # Convert std → reliability score (higher std = lower reliability)
    reliability = np.exp(-std)

    return float(y_pred), float(reliability)

# ======== Output Format =========
# 1. (Float Type) Remaining Useful Life percentage prediction (Like 60 - means 60% of tool life remains at this point) 
# 2. (Float Type) Reliability Score: will be from [0,1], higher value - higher reliability - higher confidence in predictions made. 
# ===============================

# ==========================================================
