import os
import json
import logging
import sqlite3
import torch
import cv2
import numpy as np
from collections import Counter

from flask import Flask, render_template, request, jsonify, session, redirect
from datetime import datetime

from torchvision import transforms, models
import torch.nn as nn
from werkzeug.security import generate_password_hash, check_password_hash
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DEFAULT_CLASS_MAP = {
    0: "Healthy - Batch 1",
    1: "Healthy - Batch 2",
    2: "Rust - Batch 3",
    3: "Rust - Batch 4",
    4: "Wilt - Batch 5",
    5: "Wilt - Batch 6",
    6: "Healthy - Batch 7",
    7: "Rust - Batch 8",
    8: "Wilt - Batch 9",
    9: "Healthy - Batch 10"
}

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(BASE_DIR, "app.log")), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(STATIC_DIR, 'uploads')
app.config['HEATMAP_FOLDER'] = os.path.join(STATIC_DIR, 'heatmaps')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = "coffee_guard_ai_secret"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# =========================
# MODEL
# =========================
class CoffeeLeafCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=None)

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# =========================
# SAFE MODEL LOADING
# =========================
def load_model():
    metadata_path = os.path.join(BASE_DIR, "coffee_cnn_metadata.json")
    model_path = os.path.join(BASE_DIR, "coffee_cnn_best.pth")

    metadata = {
        "num_classes": 10,
        "class_map": DEFAULT_CLASS_MAP
    }

    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                metadata["class_map"] = {
                    int(key): value for key, value in metadata.get("class_map", DEFAULT_CLASS_MAP).items()
                }

        model = CoffeeLeafCNN(metadata["num_classes"]).to(device)

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=device)
                model.load_state_dict(state, strict=False)
                model.eval()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Model mismatch: {e}")
                return None, metadata, metadata["class_map"]
        else:
            logger.warning("Model file missing")
            return None, metadata, metadata["class_map"]

        return model, metadata, metadata["class_map"]

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None, metadata, metadata["class_map"]

model, metadata, class_map = load_model()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def estimate_yield(prediction, confidence):
    label = str(prediction).lower()

    if "healthy" in label:
        base_yield = 94
        message = "Strong yield outlook. Keep monitoring and maintain normal field care."
    elif "rust" in label:
        base_yield = 68
        message = "Moderate yield risk from rust. Scout nearby plants and consider treatment."
    elif "wilt" in label:
        base_yield = 52
        message = "High yield risk from wilt symptoms. Isolate affected plants and act quickly."
    else:
        base_yield = 75
        message = "Yield outlook is uncertain. Confirm with more leaf samples."

    confidence_adjustment = (confidence - 0.5) * 18
    estimate = max(20, min(98, base_yield + confidence_adjustment))
    return round(estimate, 1), message

def generate_heatmap(original_img, tensor, predicted_idx, filename):
    if model is None:
        return None

    activations = []
    gradients = []
    target_layer = model.backbone.layer4[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        score = output[0, predicted_idx]
        score.backward()

        if not activations or not gradients:
            return None

        activation = activations[0].detach()[0]
        gradient = gradients[0].detach()[0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activation).sum(dim=0)
        cam = F.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        if torch.isclose(cam_max, cam_min):
            return None

        cam = ((cam - cam_min) / (cam_max - cam_min)).cpu().numpy()
        cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
        cv2.imwrite(heatmap_path, overlay)
        return heatmap_filename
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        return None
    finally:
        forward_handle.remove()
        backward_handle.remove()

# =========================
# DATABASE
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'farmer',
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            filename TEXT,
            result TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)

    c.execute("PRAGMA table_info(predictions)")
    columns = {row[1] for row in c.fetchall()}
    if "heatmap_filename" not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN heatmap_filename TEXT")
    if "yield_estimate" not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN yield_estimate REAL")
    if "yield_message" not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN yield_message TEXT")

    conn.commit()
    conn.close()

init_db()

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return redirect('/login')

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect('/login')

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT * FROM predictions
        WHERE email=?
        ORDER BY id DESC
        LIMIT 30
    """, (session['email'],))

    history = c.fetchall()

    c.execute("SELECT COUNT(*) FROM predictions WHERE email=?", (session['email'],))
    total_scans = c.fetchone()[0]

    conn.close()

    conf = [row[4] for row in history if row[4]]
    avg_conf = float(np.mean(conf) * 100) if conf else 0
    disease_counts = Counter(row[3] for row in history if row[3])

    healthy_count = sum(v for k, v in disease_counts.items() if "healthy" in str(k).lower())
    at_risk_count = len(history) - healthy_count

    yields = [row[7] for row in history if row[7] is not None]
    avg_yield = round(float(np.mean(yields)), 1) if yields else 0
    best_yield = round(max(yields), 1) if yields else 0
    worst_yield = round(min(yields), 1) if yields else 0

    most_common_disease = disease_counts.most_common(1)[0][0] if disease_counts else "No data yet"

    chrono = list(reversed(history))
    trend_labels = [(row[5][5:16] if row[5] else "") for row in chrono]
    confidence_trend = [round((row[4] or 0) * 100, 1) for row in chrono]
    yield_trend = [row[7] or 0 for row in chrono]

    history_rows_json = [
        {
            "date": row[5],
            "result": row[3],
            "confidence": round((row[4] or 0) * 100, 1),
            "yieldEstimate": row[7] or 0
        }
        for row in history
    ]

    return render_template(
        "dashboard.html",
        history=history,
        fullname=session.get("fullname"),
        avg_confidence=round(avg_conf, 2),
        chart_labels=list(disease_counts.keys()),
        chart_values=list(disease_counts.values()),
        total_scans=total_scans,
        healthy_count=healthy_count,
        at_risk_count=at_risk_count,
        avg_yield=avg_yield,
        best_yield=best_yield,
        worst_yield=worst_yield,
        most_common_disease=most_common_disease,
        trend_labels=trend_labels,
        confidence_trend=confidence_trend,
        yield_trend=yield_trend,
        history_rows_json=history_rows_json
    )

@app.route('/upload')
def upload():
    if 'email' not in session:
        return redirect('/login')

    return render_template("upload.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# =========================
# REGISTER
# =========================
@app.route('/register_user', methods=['POST'])
def register_user():
    fullname = request.form.get('fullname')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    if password != confirm_password:
        return render_template("register.html", error="Passwords do not match")

    hashed = generate_password_hash(password)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute("""
            INSERT INTO users (fullname, email, password, created_at)
            VALUES (?, ?, ?, ?)
        """, (fullname, email, hashed,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        session['email'] = email
        session['fullname'] = fullname

        return redirect('/dashboard')

    except sqlite3.IntegrityError:
        return render_template("register.html", error="Email already exists")

    finally:
        conn.close()

# =========================
# LOGIN
# =========================
@app.route('/login_user', methods=['POST'])
def login_user():
    email = request.form.get('email')
    password = request.form.get('password')

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE email=?", (email,))
    user = c.fetchone()
    conn.close()

    if user and check_password_hash(user[3], password):
        session['email'] = email
        session['fullname'] = user[1]
        session['role'] = user[4]
        return redirect('/dashboard')

    return render_template("login.html", error="Invalid credentials")

# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({"error": "Not logged in"}), 401
        return redirect('/login')

    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image"}), 400

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return jsonify({"error": "Invalid image"}), 400

    display_img = img_bgr.copy()
    img = cv2.resize(img_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = transform(img).unsqueeze(0).to(device)

    out = model(tensor)
    prob = torch.softmax(out, dim=1)
    conf, idx = torch.max(prob, 1)

    idx = idx.item()
    conf = float(conf.item())

    pred = class_map.get(idx, f"Class_{idx}")
    heatmap_filename = generate_heatmap(display_img, tensor, idx, filename)
    yield_estimate, yield_message = estimate_yield(pred, conf)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        INSERT INTO predictions (
            email, filename, result, confidence, timestamp,
            heatmap_filename, yield_estimate, yield_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session['email'],
        filename,
        pred,
        conf,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        heatmap_filename,
        yield_estimate,
        yield_message
    ))

    conn.commit()
    conn.close()

    if request.accept_mimetypes.accept_html and not request.accept_mimetypes.accept_json:
        return redirect('/dashboard')

    return jsonify({
        "prediction": pred,
        "confidence": conf * 100,
        "heatmap": f"/static/heatmaps/{heatmap_filename}" if heatmap_filename else None,
        "yield_estimate": yield_estimate,
        "yield_message": yield_message
    })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("CoffeeGuard AI Starting...")
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
