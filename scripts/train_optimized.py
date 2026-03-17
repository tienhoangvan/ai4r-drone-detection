from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO26 for drone detection with safe default settings for modest GPUs."
    )
    parser.add_argument("--model", type=str, default="../models/yolo26n.pt",
                        help="Path to pretrained YOLO26 weights")
    parser.add_argument("--data", type=str, default="../dataset/drone_round1/data.yaml",
                        help="Path to dataset YAML")
    parser.add_argument("--project", type=str, default="../runs/detect",
                        help="Project directory for training runs")
    parser.add_argument("--name", type=str, default="drone_r1_t1000_safe",
                        help="Run name")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device id, e.g. 0. Use 'cpu' for CPU training.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the last interrupted run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).resolve()
    data_path = Path(args.data).resolve()
    project_path = Path(args.project).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    project_path.mkdir(parents=True, exist_ok=True)

    print("=== YOLO26 TRAINING CONFIG ===")
    print(f"model    : {model_path}")
    print(f"data     : {data_path}")
    print(f"project  : {project_path}")
    print(f"name     : {args.name}")
    print(f"epochs   : {args.epochs}")
    print(f"imgsz    : {args.imgsz}")
    print(f"batch    : {args.batch}")
    print(f"device   : {args.device}")
    print(f"workers  : {args.workers}")
    print(f"patience : {args.patience}")
    print(f"resume   : {args.resume}")

    model = YOLO(str(model_path))

    # Safe defaults for Quadro T1000 / modest GPUs:
    # - imgsz=640 and batch=4 to avoid OOM
    # - amp=False because AMP check failed on this GPU
    # - cache=False for deterministic and lighter behavior
    # - mosaic disabled to reduce memory pressure and stabilize tiny-object training
    train_args = dict(
        data=str(data_path),
        project=str(project_path),
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        pretrained=True,
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        cache=False,
        amp=False,
        single_cls=True,
        close_mosaic=0,
        overlap_mask=False,
        rect=False,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
        exist_ok=True,
        resume=args.resume,
    )

    results = model.train(**train_args)

    print("\n=== TRAINING DONE ===")
    print(f"Results object: {results}")
    print(f"Run directory : {project_path / args.name}")
    print("Best weights  : see weights/best.pt inside the run directory")


if __name__ == "__main__":
    main()
