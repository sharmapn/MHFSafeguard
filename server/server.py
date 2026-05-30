# server.py
# MHF Safeguard classifier API server
#
# This server receives one full cleaned XenForo message per submission,
# classifies it using a trained ktrain/BERT model, and returns a structured
# response that the MHFSafeguard XenForo plugin can understand.

import os
import traceback
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request
import ktrain


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Path to the saved ktrain predictor model.
# You can override this using the MHFS_MODEL_PATH environment variable.
MODEL_PATH = os.environ.get(
    "MHFS_MODEL_PATH",
    "../content/bert_model_Suicide"
)

# Optional API key.
# If this is set, XenForo must send:
# Authorization: Bearer YOUR_KEY
API_KEY = os.environ.get("MHFS_API_KEY", "")

# Server host and port.
HOST = os.environ.get("MHFS_HOST", "127.0.0.1")
PORT = int(os.environ.get("MHFS_PORT", "8000"))

# Score thresholds used by the API to recommend moderation actions.
MODERATE_THRESHOLD = float(os.environ.get("MHFS_MODERATE_THRESHOLD", "85"))
REVISE_THRESHOLD = float(os.environ.get("MHFS_REVISE_THRESHOLD", "95"))


# ------------------------------------------------------------
# Flask app and model loading
# ------------------------------------------------------------

app = Flask(__name__)

print(f"[INFO] Loading model from: {MODEL_PATH}")
predictor = ktrain.load_predictor(MODEL_PATH)
print("[INFO] Model loaded successfully.")


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def check_api_key() -> bool:
    """
    Check Bearer-token authentication only if MHFS_API_KEY is configured.
    If MHFS_API_KEY is empty, authentication is disabled.
    """

    if not API_KEY:
        return True

    auth_header = request.headers.get("Authorization", "")
    expected = f"Bearer {API_KEY}"

    return auth_header == expected


def extract_message(payload: Dict[str, Any]) -> str:
    """
    Extract the cleaned full message sent by the XenForo plugin.
    """

    message = payload.get("message", "")

    if message is None:
        message = ""

    return str(message).strip()


def normalise_label(raw_label: str) -> str:
    """
    Convert model-specific labels into the labels expected by the plugin.

    Adjust these mappings based on the exact labels used by the final model.
    """

    label = str(raw_label).strip().lower()

    if label in [
        "method",
        "method_or_action",
        "suicide method",
        "self-harm method",
        "self harm method",
        "harmful method"
    ]:
        return "method_or_action"

    if "method" in label or "action" in label:
        return "method_or_action"

    if "ideation" in label:
        return "ideation"

    if "not" in label or "none" in label or "safe" in label or "non" in label:
        return "not_harmful"

    return label.replace(" ", "_")


def get_prediction(message: str) -> Tuple[str, float]:
    """
    Run the trained model and return:
    - normalised label
    - confidence score from 0 to 100

    This first tries predict_proba(). If probability output is unavailable,
    it falls back to predictor.predict().
    """

    classes = []

    try:
        classes = list(predictor.get_classes())
    except Exception:
        classes = []

    # Try probability prediction first.
    try:
        probabilities = predictor.predict_proba([message])

        if hasattr(probabilities, "tolist"):
            probabilities = probabilities.tolist()

        if isinstance(probabilities, list) and len(probabilities) > 0:
            row = probabilities[0]

            best_index = max(range(len(row)), key=lambda i: row[i])
            raw_label = classes[best_index] if classes else str(best_index)
            score = float(row[best_index]) * 100.0

            return normalise_label(raw_label), round(score, 2)

    except Exception:
        # Some ktrain predictors may not expose predict_proba in the same way.
        # In that case, use the fallback below.
        pass

    # Fallback prediction.
    raw_prediction = predictor.predict(message)

    if isinstance(raw_prediction, list) and len(raw_prediction) > 0:
        raw_prediction = raw_prediction[0]

    label = normalise_label(str(raw_prediction))

    # If no confidence is available, return a cautious default.
    if label == "not_harmful":
        score = 20.0
    else:
        score = 90.0

    return label, score


def decide_action(label: str, score: float) -> Tuple[str, str]:
    """
    Convert label and score into:
    - risk_level
    - recommended_action
    """

    if label == "not_harmful":
        return "none", "allow"

    if score >= REVISE_THRESHOLD:
        return "critical", "revise"

    if score >= MODERATE_THRESHOLD:
        return "high", "moderate"

    if label in ["method_or_action", "ideation"]:
        return "medium", "review"

    return "low", "allow"


def build_flagged_parts(message: str, label: str, score: float) -> List[Dict[str, Any]]:
    """
    Return flagged span information.

    In this first version, if the model flags the message as risky, the full
    message is returned as the flagged span. Later, this can be improved by:
    - sentence splitting,
    - keyword/span extraction,
    - model attention-based span detection,
    - or a second sentence-level classifier.
    """

    if label == "not_harmful":
        return []

    return [
        {
            "text": message,
            "label": label,
            "score": score,
            "start_offset": 0,
            "end_offset": len(message)
        }
    ]


def classify_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main classification function.
    """

    message = extract_message(payload)

    if not message:
        return {
            "risk_level": "none",
            "recommended_action": "allow",
            "highest_label": "not_harmful",
            "highest_score": 0,
            "flagged_parts": [],
            "error": "Empty message."
        }

    label, score = get_prediction(message)
    risk_level, action = decide_action(label, score)
    flagged_parts = build_flagged_parts(message, label, score)

    return {
        "risk_level": risk_level,
        "recommended_action": action,
        "highest_label": label,
        "highest_score": score,
        "flagged_parts": flagged_parts
    }


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """
    Basic status endpoint.
    """

    return jsonify({
        "service": "MHF Safeguard Classifier API",
        "status": "running",
        "model_path": MODEL_PATH
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    """

    return jsonify({
        "ok": True,
        "message": "Classifier API is running."
    })


@app.route("/api/classify", methods=["POST"])
@app.route("/", methods=["POST"])
def classify():
    """
    Main API endpoint used by the XenForo plugin.
    """

    if not check_api_key():
        return jsonify({
            "risk_level": "unknown",
            "recommended_action": "moderate",
            "highest_label": "",
            "highest_score": 0,
            "flagged_parts": [],
            "error": "Unauthorized request."
        }), 401

    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "risk_level": "unknown",
                "recommended_action": "moderate",
                "highest_label": "",
                "highest_score": 0,
                "flagged_parts": [],
                "error": "Invalid JSON payload."
            }), 400

        result = classify_payload(payload)

        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "risk_level": "unknown",
            "recommended_action": "moderate",
            "highest_label": "",
            "highest_score": 0,
            "flagged_parts": [],
            "error": str(e)
        }), 500


# ------------------------------------------------------------
# Run server
# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=False,
        host=HOST,
        port=PORT
    )
