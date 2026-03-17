from pathlib import Path

LABEL_ROOT = Path("../dataset/drone_round1/labels").resolve()
EPS = 1e-9

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

fixed_count = 0
file_count = 0

for txt_path in LABEL_ROOT.rglob("*.txt"):
    file_count += 1
    lines = txt_path.read_text(encoding="utf-8").splitlines()

    if not lines:
        continue

    new_lines = []
    changed = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            new_lines.append(line)
            continue

        cls_id = parts[0]
        x, y, w, h = map(float, parts[1:])

        # YOLO -> xyxy
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2

        # clamp to image bounds
        x_min_c = clamp(x_min)
        y_min_c = clamp(y_min)
        x_max_c = clamp(x_max)
        y_max_c = clamp(y_max)

        # if box collapses after clamp, skip it
        new_w = x_max_c - x_min_c
        new_h = y_max_c - y_min_c
        if new_w <= EPS or new_h <= EPS:
            changed = True
            continue

        # xyxy -> YOLO
        new_x = (x_min_c + x_max_c) / 2
        new_y = (y_min_c + y_max_c) / 2

        if (
            abs(new_x - x) > EPS
            or abs(new_y - y) > EPS
            or abs(new_w - w) > EPS
            or abs(new_h - h) > EPS
        ):
            changed = True

        new_lines.append(f"{cls_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")

    if changed:
        txt_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        fixed_count += 1
        print(f"Fixed: {txt_path}")

print(f"\nDone. Scanned {file_count} files, fixed {fixed_count} files.")