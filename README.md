# 🚁 YOLOv8 Drone Surveillance System

![YOLOv8](https://img.shields.io/badge/YOLO-v8-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

A professional computer vision project for aerial surveillance. This system automatically loads your trained weights from `models/best.pt` when available (falls back to a lightweight `yolov8n.pt` base model). It also includes adaptive feature enhancement and an attention-driven locking mechanism to focus on the most relevant objects.

## ✨ Key Features
* **Auto-loaded Trained Weights:** If `models/best.pt` exists, it will be used for inference automatically.
* **Lightweight Base Model:** Uses `yolov8n.pt` when no custom weights are present, keeping runtime fast and efficient.
* **Smart Locking with Attention:**
    * **Size Locking:** Targets large objects automatically.
    * **Attention Locking:** Considers motion, confidence, and size to focus on high-priority targets.
    * **Anomaly Locking:** Uses `IsolationForest` and adaptive z-score analysis to detect irregular behavior.
* **Modular Structure:** Clean separation of logic, config, and execution.

## � Model Evaluation

The `model_evaluation.py` script provides comprehensive evaluation metrics for your trained model:

- **TP, FP, FN Calculation:** Computes True Positives, False Positives, and False Negatives
- **Precision & Recall:** Calculates precision and recall at different confidence thresholds
- **P-R Curve Visualization:** Plots Precision-Recall curves for visual analysis
- **Average Precision (AP):** Computes AP for each class and overall mAP

### Usage
```bash
python model_evaluation.py
```

**Note:** Ensure the VisDrone dataset YAML file path is correct in the script (`D:/Visdrone2019/Task1/visdrone.yaml`). Modify the `data_path` variable if your dataset is located elsewhere.

The script will generate:
- Console output with detailed metrics
- P-R curve plots saved to `runs/val/exp/`
- Interactive matplotlib visualizations

## 🚀 Quick Start
