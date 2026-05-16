# AI Weapon Detection Console

Computer-vision project for real-time weapon detection using a trained Ultralytics YOLO model and a desktop Tkinter inference console.

## What It Shows

- Real-time webcam inference using `best.pt`
- Confidence threshold control for precision-oriented demos
- FPS and latency telemetry for practical ML evaluation
- Per-category detection breakdown for handguns, rifles, knives, explosives, and other weapon classes
- Still-image analysis for demos without a webcam
- Snapshot and CSV export for reviewing detection evidence

## Project Files

- `app.py` - latest polished desktop application
- `best.pt` - trained YOLO model weights
- `live.py` - lightweight OpenCV webcam inference script
- `ProML.ipynb` - training / experimentation notebook
- `metrics.png`, `train.png`, `val.png` - model evaluation and dataset visuals
- `gun.png` - sample image used by the app for quick inference

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Optional arguments:

```bash
python app.py --model best.pt --camera 0
```

## Demo Flow

1. Start the app with `python app.py`.
2. Use **Analyze Image** for a quick static-image demo.
3. Use **Start Live Detection** for webcam inference.
4. Adjust the confidence threshold to show how false positives are controlled.
5. Export a CSV after inference to show reviewer-friendly evidence logging.

## Responsible Use

This project is a computer-vision demonstration and should be validated on representative data before operational use. For safety-critical deployments, add human review, drift monitoring, bias testing, privacy controls, and clear escalation policies.
