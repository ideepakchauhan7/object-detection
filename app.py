"""Gradio web demo for live YOLOv8 object detection inference."""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gradio as gr

from src.utils import (
    get_nested,
    load_config,
    require_package,
    resolve_weights,
    ultralytics_device,
)


def parse_args() -> argparse.Namespace:
    """Parse Gradio server options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", help="Weights to load for inference")
    parser.add_argument("--server-name", help="Host interface for Gradio")
    parser.add_argument("--server-port", type=int, help="Port for Gradio")
    parser.add_argument("--share", action="store_true", help="Create Gradio share URL")
    return parser.parse_args()


@lru_cache(maxsize=2)
def load_model(weights: str) -> Any:
    """Load and cache a YOLO model by weights path/name."""

    ultralytics = require_package("ultralytics")
    return ultralytics.YOLO(weights)


def draw_predictions(
    image: np.ndarray,
    result: Any,
) -> tuple[np.ndarray, list[list[Any]]]:
    """Draw model predictions on an RGB image and build a table."""

    pil_image = Image.fromarray(image.astype("uint8"), mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    names = getattr(result, "names", {})
    rows: list[list[Any]] = []

    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return np.asarray(pil_image), rows

    for box in boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
        class_name = names.get(class_id, str(class_id))
        label = f"{class_name} {confidence:.2f}"

        draw.rectangle((ix1, iy1, ix2, iy2), outline=(34, 197, 94), width=3)
        text_bbox = draw.textbbox((ix1, iy1), label, font=font)
        draw.rectangle(text_bbox, fill=(34, 197, 94))
        draw.text((ix1, iy1), label, fill=(0, 0, 0), font=font)
        rows.append(
            [
                class_name,
                round(confidence, 4),
                round(x1, 1),
                round(y1, 1),
                round(x2, 1),
                round(y2, 1),
            ]
        )

    return np.asarray(pil_image), rows


def build_predict_fn(config: dict[str, Any], weights: str):
    """Create a Gradio-compatible prediction function."""

    device = ultralytics_device(get_nested(config, ["runtime", "device"], "auto"))

    def predict(
        image: np.ndarray | None,
        confidence: float,
        iou: float,
    ) -> tuple[np.ndarray | None, list[list[Any]]]:
        """Run YOLO prediction for one uploaded/webcam image."""

        if image is None:
            return None, []

        model = load_model(weights)
        results = model.predict(
            source=image,
            conf=float(confidence),
            iou=float(iou),
            device=device,
            verbose=False,
        )
        return draw_predictions(image, results[0])

    return predict


def create_demo(config: dict[str, Any], weights: str) -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    detection = config.get("detection", {})
    predict = build_predict_fn(config, weights)

    with gr.Blocks(title="YOLOv8 Object Detection") as demo:
        gr.Markdown("# Object Detection using YOLOv8")
        gr.Markdown(f"Model weights: `{weights}`")

        with gr.Row():
            input_image = gr.Image(
                label="Input Image",
                sources=["upload", "webcam"],
                type="numpy",
                image_mode="RGB",
            )
            output_image = gr.Image(label="Detections", type="numpy")

        with gr.Row():
            confidence = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=float(detection.get("conf", 0.25)),
                step=0.05,
                label="Confidence",
            )
            iou = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=float(detection.get("iou", 0.45)),
                step=0.05,
                label="IoU",
            )

        run_button = gr.Button("Run Detection", variant="primary")
        detections = gr.Dataframe(
            headers=["Class", "Confidence", "X1", "Y1", "X2", "Y2"],
            label="Detection Results",
            interactive=False,
        )

        run_button.click(
            fn=predict,
            inputs=[input_image, confidence, iou],
            outputs=[output_image, detections],
        )

    return demo


def main() -> None:
    """Launch the Gradio app."""

    args = parse_args()
    config = load_config(args.config)
    weights = resolve_weights(args.weights, config)
    demo = create_demo(config, weights)
    app_config = config.get("app", {})

    demo.launch(
        server_name=args.server_name or app_config.get("host", "0.0.0.0"),
        server_port=args.server_port or int(app_config.get("port", 7860)),
        share=args.share or bool(app_config.get("share", False)),
    )


if __name__ == "__main__":
    main()
