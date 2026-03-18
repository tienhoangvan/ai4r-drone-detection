from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    # Step: define and parse CLI arguments (keeps training reproducible/configurable)
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
    # Step 1: Parse CLI arguments
    args = parse_args()

    # Step 2: Resolve key paths and validate inputs
    model_path = Path(args.model).resolve()
    data_path = Path(args.data).resolve()
    project_path = Path(args.project).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    project_path.mkdir(parents=True, exist_ok=True)

    # Step 3: Print the effective configuration (useful for debugging runs)
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

    # Step 4: Load the model weights
    model = YOLO(str(model_path))

    # Step 5: Build training arguments (safe defaults for modest GPUs)
    # - imgsz=640 and batch=4 to reduce OOM risk
    # - amp=False because AMP may fail on some older GPUs
    # - cache=False for deterministic/lighter behavior
    # - mosaic disabled to reduce memory pressure and stabilize tiny-object training
    train_args = dict(
        data=str(data_path),              # Dataset config (YOLO data.yaml): paths + class names
        project=str(project_path),        # Root folder where Ultralytics writes runs (e.g., runs/detect)
        name=args.name,                   # Run name (subfolder under `project/`)
        epochs=args.epochs,               # Number of training epochs
        imgsz=args.imgsz,                 # Training image size (square input, e.g., 640)
        batch=args.batch,                 # Batch size (reduce if you hit OOM)
        device=args.device,               # Device selector: "0", "0,1", or "cpu"
        workers=args.workers,             # Dataloader worker processes (lower can be more stable on some systems)
        seed=args.seed,                   # Random seed for reproducibility
        patience=args.patience,           # Early stopping patience (epochs without improvement before stop)
        pretrained=True,                  # Start from pretrained weights instead of training from scratch
        optimizer="AdamW",                # Optimizer choice (AdamW is often stable for small-object datasets)
        lr0=5e-4,                         # Initial learning rate at the start of training
        lrf=0.01,                         # Final LR ratio (final_lr = lr0 * lrf) when using LR scheduling
        momentum=0.937,                   # Optimizer momentum (applies to SGD; for AdamW Ultralytics may ignore/translate)
        weight_decay=5e-4,                # Weight decay (L2 regularization strength)
        warmup_epochs=3.0,                # Warmup duration in epochs (ramps LR/momentum up at start)
        warmup_momentum=0.8,              # Warmup momentum value (used during warmup period)
        warmup_bias_lr=0.1,               # Warmup LR for bias parameters (can help stabilize early training)
        cos_lr=True,                      # Use cosine LR schedule (smoothly decays LR over epochs)
        cache=False,                      # Cache images in RAM/disk (False = lower memory, more deterministic I/O)
        amp=False,                        # Automatic Mixed Precision (False = safer on older GPUs / AMP issues)
        single_cls=True,                  # Treat dataset as single-class (all labels mapped to class 0)
        close_mosaic=0,                   # Epoch to disable mosaic augmentation (0 = effectively disabled)
        overlap_mask=False,               # Instance segmentation setting (False for pure detection)
        rect=False,                       # Rectangular training (keeps aspect ratios); False = standard square training
        hsv_h=0.01,                       # HSV hue augmentation gain
        hsv_s=0.5,                        # HSV saturation augmentation gain
        hsv_v=0.3,                        # HSV value/brightness augmentation gain
        degrees=3.0,                      # Random rotation degrees
        translate=0.05,                   # Random translation fraction
        scale=0.15,                       # Random scaling gain
        shear=0.0,                        # Random shear degrees
        perspective=0.0,                  # Random perspective transform magnitude
        flipud=0.0,                       # Vertical flip probability
        fliplr=0.5,                       # Horizontal flip probability
        mosaic=0.0,                       # Mosaic augmentation probability (0 disables mosaic)
        mixup=0.0,                        # MixUp augmentation probability (0 disables mixup)
        copy_paste=0.0,                   # Copy-paste augmentation probability (0 disables)
        erasing=0.0,                      # Random erasing augmentation probability (0 disables)
        save=True,                        # Save checkpoints and artifacts for the run
        save_period=10,                   # Save checkpoint every N epochs
        plots=True,                       # Save training curves/plots to the run directory
        val=True,                         # Run validation during training (per epoch)
        verbose=True,                     # Verbose logging during training
        exist_ok=True,                    # Allow overwriting an existing run directory with the same name
        resume=args.resume,               # Resume from last checkpoint if available
    )

    # Step 6: Start training
    results = model.train(**train_args)

    # Step 7: Print a short post-training summary
    print("\n=== TRAINING DONE ===")
    print(f"Results object: {results}")
    print(f"Run directory : {project_path / args.name}")
    print("Best weights  : see weights/best.pt inside the run directory")


if __name__ == "__main__":
    main()
