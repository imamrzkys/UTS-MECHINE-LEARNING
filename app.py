import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cyber-attack-detection-2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# =================================================
# JSON-SAFE CONVERTER
# =================================================
def convert_to_native(obj):
    if isinstance(obj, (np.int32, np.int64)): 
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)): 
        return float(obj)
    if isinstance(obj, np.bool_): 
        return bool(obj)
    if isinstance(obj, dict): 
        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): 
        return [convert_to_native(v) for v in obj]
    if isinstance(obj, pd.Series): 
        return convert_to_native(obj.to_dict())
    return obj

# =================================================
# LOAD MODEL
# =================================================
MODEL_PATH = "random_forests_model.pkl"
model_artifact = joblib.load(MODEL_PATH)

model = model_artifact["model"]
feature_names = model_artifact["feature_names"]
fill_values = model_artifact["fill_values"]
label_encoder = model_artifact["label_encoder"]
metadata = convert_to_native(model_artifact["metadata"])

# =================================================
# FULL LIST OF CATEGORICAL FEATURES (MODEL-COMPATIBLE)
# =================================================
CATEGORICAL_FULL = [
    "proto", "service", "conn_state", "http_method",
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed",
    "ssl_established", "weird_notice"
]

# Safety: Keep only existing columns
CATEGORICAL_FULL = [c for c in CATEGORICAL_FULL if c in feature_names]

# =================================================
# AUTO-FILL SYSTEM
# =================================================
df = pd.read_csv("TON_Dataset.csv")
df_valid = df[feature_names].copy()

# --- NEW: Generate two real examples for auto-fill ---
normal_row = df[df["label"] == 0]
if not normal_row.empty:
    normal_row = normal_row[feature_names].sample(1, random_state=1).iloc[0]
else:
    normal_row = df_valid.sample(1, random_state=1).iloc[0]
# Ambil satu contoh attack dari baris kedua dataset (label=1) untuk percobaan
# Ambil satu contoh attack dari baris ke-3 dataset (label=1) untuk percobaan
attack_row = df[df['label'] == 1]
if len(attack_row) >= 2:
    attack_row = attack_row[feature_names].iloc[1]
elif not attack_row.empty:
    attack_row = attack_row[feature_names].iloc[0]
else:
    attack_row = df_valid.sample(1, random_state=2).iloc[0]

def row_to_example(row):
    ex = {}
    for f in feature_names:
        val = row[f]
        if isinstance(val, str):
            ex[f] = val.strip()
        elif pd.isna(val):
            ex[f] = fill_values.get(f, 0.0)
        else:
            ex[f] = convert_to_native(val)
    return ex

examples = {
    "normal_example": row_to_example(normal_row),
    "attack_example": row_to_example(attack_row),
}
# (Optionally keep old valid_example for backward compatibility)
row = df_valid.sample(1, random_state=42).iloc[0]
example_input = {}
for f in feature_names:
    val = row[f]
    if isinstance(val, str):
        example_input[f] = val.strip()
    elif pd.isna(val):
        example_input[f] = fill_values.get(f, 0.0)
    else:
        example_input[f] = convert_to_native(val)
examples["valid_example"] = example_input

# =================================================
# HOME
# =================================================
@app.route("/")
def index():
    # gambar visualisasi yang ada di folder static
    STATIC_IMAGES = [
        "confusion matrix-LR.png",
        "confusion matrix-RF.png",
        "feature importance.png",
        "grafik distribusi label.png",
        "heatmap korelasi.png",
        "ROC curve - LR.png",
        "ROC curve - RF.png"
    ]

    IMAGE_TITLES = {
        "confusion matrix-LR.png": "Confusion Matrix - Logistic Regression",
        "confusion matrix-RF.png": "Confusion Matrix - Random Forest",
        "feature importance.png": "Feature Importance (Random Forest)",
        "grafik distribusi label.png": "Distribusi Label (Normal vs Attack)",
        "heatmap korelasi.png": "Heatmap Korelasi Fitur",
        "ROC curve - LR.png": "ROC Curve - Logistic Regression",
        "ROC curve - RF.png": "ROC Curve - Random Forest"
    }

    return render_template(
        "index.html",
        metadata=metadata,
        feature_names=feature_names,
        images=STATIC_IMAGES,
        image_titles=IMAGE_TITLES,
        examples=examples
    )
# =================================================
# PREDICT (FINAL + CLEAN + SAFE)
# =================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = {}

        # ============================================
        # BACA INPUT USER
        # ============================================
        for feat in feature_names:
            val = data.get(feat, "")

            # ---------- KATEGORIKAL ----------
            if feat in CATEGORICAL_FULL:
                if isinstance(val, str) and val.strip() != "":
                    input_data[feat] = val.strip()
                else:
                    input_data[feat] = "-"      # Default kategori aman
                continue

            # ---------- NUMERIK ----------
            try:
                v = str(val).replace(",", ".")
                input_data[feat] = float(v)
            except:
                input_data[feat] = float(fill_values.get(feat, 0.0))

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # ============================================
        # PREDIKSI
        # ============================================
        pred_raw = model.predict(input_df)[0]
        pred_num = int(pred_raw)

        pred_proba = model.predict_proba(input_df)[0]
        prob_normal = float(pred_proba[0] * 100)
        prob_attack = float(pred_proba[1] * 100)

        # label via encoder
        try:
            pred_label = label_encoder.inverse_transform([pred_num])[0]
        except:
            pred_label = "Attack" if pred_num == 1 else "Normal"

        # pastikan label string
        if isinstance(pred_label, (int, np.integer)):
            pred_label = "Attack" if pred_label == 1 else "Normal"

        pred_label = str(pred_label).strip()

        # pilih confidence
        if pred_label.lower() == "attack":
            confidence = prob_attack
        else:
            confidence = prob_normal

        # ============================================
        # RETURN JSON
        # ============================================
        return jsonify({
            "success": True,
            "prediction": pred_label,
            "confidence": round(float(confidence), 2),
            "probability_normal": round(float(prob_normal), 2),
            "probability_attack": round(float(prob_attack), 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400

# =================================================
# RUN
# =================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
