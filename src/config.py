
import os

# --- MODEL SETTINGS ---
# Prefer a lightweight base model (YOLOv8 Nano) for edge/embedded inference.
# If you have finished training, the project will automatically load `models/best.pt`.
BASE_MODEL = "yolov8n.pt"  # lightweight base model (nano)
CUSTOM_WEIGHTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"))
MODEL_NAME = CUSTOM_WEIGHTS if os.path.exists(CUSTOM_WEIGHTS) else BASE_MODEL

# Inference Resolution:
# Standard YOLO is 640px. We use 1280px to see small cars/people clearly.
INFERENCE_SIZE = 1280
CONF_THRESHOLD = 0.25

# --- ATTENTION / ADAPTIVE FEATURE SETTINGS ---
# How strongly an object needs to stand out (size/motion/confidence) before we focus on it.
ATTENTION_THRESHOLD = 0.7
# Multiplier for lock duration when an object has high attention score.
ATTENTION_LOCK_MULTIPLIER = 1.5

# --- LOCKING LOGIC ---
# If an object covers >5% of the screen, we lock it by default.
LOCK_SIZE_THRESHOLD = 0.05

# How many frames to buffer before the AI learns what is "Normal"
ANOMALY_BUFFER_SIZE = 50

# How long (in frames) to keep a target locked
LOCK_DURATION_FRAMES = 60
