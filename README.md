# Object Detection using YOLO (YOLOv8)

Intern ID: **CITS1758**

Project Title: **Object Detection using YOLO (YOLOv8)**

This project is a complete Python object detection pipeline built around the
Ultralytics YOLOv8 API. It supports dataset preparation, annotation validation,
augmentation, transfer learning, fine-tuning, evaluation, inference, export, and
a Gradio web demo for live predictions.

## Objectives

- Train YOLOv8 on a custom YOLO-format dataset or the default COCO128 demo.
- Validate model quality with mAP@0.5, mAP@0.5:0.95, precision, recall, and F1.
- Run inference on images, folders, video files, and webcam streams.
- Export trained models to ONNX and TorchScript for deployment.
- Visualize predictions, class balance, training curves, and Ultralytics plots.

## Model Architecture

YOLOv8 is a one-stage detector that predicts object boxes and classes directly
from an image in a single forward pass.

- **Backbone:** extracts visual features at multiple scales.
- **Neck:** fuses low-level and high-level features so small and large objects
  can be detected.
- **Head:** predicts bounding boxes, class probabilities, and confidence scores.

Default transfer learning starts from `yolov8n.pt`. You can switch to
`yolov8s.pt` or a custom `.pt` checkpoint with `--model` or `--weights`.

## Pipeline

```text
[Raw Images] -> [Preprocessing] -> [YOLOv8 Training] -> [Evaluation] -> [Inference]
```

## Dataset Format

The custom dataset template is located at `data/dataset.yaml`.

```text
data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
  dataset.yaml
```

Each image label file uses YOLO format:

```text
class_id x_center y_center width height
```

All box coordinates are normalized to `[0, 1]`. For example:

```text
0 0.5123 0.4810 0.2315 0.3372
```

The default config uses `coco128.yaml` so the project can run as a demo without
adding a dataset. To fine-tune custom classes, place images and labels in the
folders above, edit `data/dataset.yaml`, and set `dataset.active: custom` in
`config.yaml`.

## Hardware Note

This project is configured for automatic CPU/CUDA selection. During the smoke
training run, Ultralytics detected an **NVIDIA GeForce RTX 3060 12GB** with
PyTorch `2.12.0+cu130`; the scripts still fall back to CPU automatically when
CUDA is unavailable.

## Installation

### Pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Conda

```bash
conda create -n yolo-object-detection python=3.12 -y
conda activate yolo-object-detection
pip install -r requirements.txt
```

### CUDA

For NVIDIA GPUs, install a PyTorch build that matches your CUDA version before
installing the rest of the requirements. See the official PyTorch install
selector for the correct command, then run:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset and `dataset.yaml`

Use COCO128 for a demo, or set `dataset.active: custom` in `config.yaml` and
update `data/dataset.yaml`:

```yaml
path: .
train: images/train
val: images/val
test: images/test
nc: 2
names:
  0: helmet
  1: vest
```

### 2. Run Preprocessing

```bash
python src/preprocess.py --config config.yaml --validate-only
```

To create offline augmented training copies:

```bash
python src/preprocess.py --config config.yaml --augment
```

Ultralytics built-in augmentation is enabled by default during training.

### 3. Train YOLOv8

```bash
python src/train.py --config config.yaml --model yolov8n.pt --epochs 50
```

Use YOLOv8s for a stronger transfer-learning baseline:

```bash
python src/train.py --config config.yaml --model yolov8s.pt --epochs 50
```

Training outputs are saved under `runs/train/`, and `best.pt` / `last.pt` are
copied to `models/`.

### 4. Evaluate

```bash
python src/evaluate.py --config config.yaml --weights models/best.pt
```

Evaluation plots, including confusion matrix, PR curve, and F1 curve generated
by Ultralytics, are saved under `runs/evaluation/`.

### 5. Detect Objects

Image or folder:

```bash
python src/detect.py --config config.yaml --weights models/best.pt --source data/images/test
```

Video:

```bash
python src/detect.py --config config.yaml --weights models/best.pt --source input.mp4
```

Webcam:

```bash
python src/detect.py --config config.yaml --weights models/best.pt --source 0
```

Annotated outputs and `detections.csv` are saved under `runs/detect/`.

### 6. Export for Deployment

```bash
python src/export.py --config config.yaml --weights models/best.pt
```

By default, exports are generated for ONNX and TorchScript and copied to
`models/exported/`.

### 7. Launch Gradio Demo

```bash
python app.py
```

Open the printed local URL, upload an image or use webcam input, adjust
confidence/IoU thresholds, and run detection.

## Sample Detection Output

The inference scripts save annotated images with bounding boxes, class labels,
and confidence scores. A `detections.csv` row looks like this:

| source | class_name | confidence | x1 | y1 | x2 | y2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| image.jpg | person | 0.93 | 120.5 | 44.0 | 321.7 | 480.2 |

## Evaluation Metrics

| Metric | Description | Example Value |
| --- | --- | ---: |
| Precision | Share of predicted boxes that are correct | Run `evaluate.py` |
| Recall | Share of ground-truth boxes detected | Run `evaluate.py` |
| mAP@0.5 | Mean AP at IoU 0.5 | Run `evaluate.py` |
| mAP@0.5:0.95 | Mean AP across IoU thresholds 0.5 to 0.95 | Run `evaluate.py` |
| F1 | Harmonic mean of precision and recall | Run `evaluate.py` |

## Visualization

Class distribution:

```bash
python src/visualize.py --config config.yaml --plot classes
```

Training curves:

```bash
python src/visualize.py --config config.yaml --plot training \
  --results-csv runs/train/yolov8_custom/results.csv
```

Ultralytics automatically writes training plots, validation plots, confusion
matrix, PR curve, and F1 curve into the relevant `runs/` folders.

## Docker

Build and run the Gradio app:

```bash
docker build -t yolo-object-detection .
docker run --rm -p 7860:7860 yolo-object-detection
```

Mount local models and data when running trained weights:

```bash
docker run --rm -p 7860:7860 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data:/app/data" \
  yolo-object-detection
```

## Technologies Used

- YOLOv8
- Ultralytics
- PyTorch
- OpenCV
- Albumentations
- Gradio
- Python

## Project Structure

```text
.
├── app.py
├── config.yaml
├── data/
│   ├── dataset.yaml
│   ├── images/
│   └── labels/
├── Dockerfile
├── models/
├── notebooks/
├── requirements.txt
├── runs/
└── src/
    ├── detect.py
    ├── evaluate.py
    ├── export.py
    ├── preprocess.py
    ├── train.py
    ├── utils.py
    └── visualize.py
```
