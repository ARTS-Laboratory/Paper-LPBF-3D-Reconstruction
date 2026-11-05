import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# AUTOMATIC BOX TRACKING FOR 0.6 mm WALL
# ---------------------------------------------------------
# Keep this script in the same folder as:
#   reference.png
#   layer1.png
#   layer2.png
#   ...
#   layer7.png
#
# Then run:
#   python automatic_box_tracking.py
#
# what it does:
#   1) detects the blue reference box from reference.png
#   2) uses the INNER edges of the blue guide lines
#   3) tracks the wall using a height-averaged intensity profile using gaussian smoothing and gradient search
#   4) uses previous layer position as a prior
#   5) draws a rigid box with fixed reference width
#   6) prints exact box width/height in pixels and mm for every layer
#   =========================================================

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
input_dir = SCRIPT_DIR

reference_path = input_dir / "reference.png"
layer_paths = [input_dir / f"layer{i}.png" for i in range(1, 8)]

# -----------------------------
# Known physical dimensions
# -----------------------------
wall_width_mm = 0.6 #defined in the CAD model, used here to convert pixels to mm
wall_length_mm = 100.0

# -----------------------------
# Output folder
# -----------------------------
out_dir = input_dir / "automatic_box_tracking_outputs"
out_dir.mkdir(exist_ok=True)

# -----------------------------
# Tracking settings
# -----------------------------
blur_ksize = 5
trim_top = 15
trim_bottom = 15
profile_sigma = 4.0
width_tolerance_px = 12

# Search windows around expected edge locations
search_half_windows = [10, 16, 24, 36]

# -----------------------------
# Drawing settings
# -----------------------------
box_color = (255, 0, 0)   # blue in OpenCV BGR
line_thickness = 2
crop_margin_x = 28
crop_margin_y = 38


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def read_image_gray_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")

    if img.ndim == 2:
        gray = img
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        bgr = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray, bgr


def detect_reference_box_from_blue_lines(reference_bgr):
    hsv = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV)

    # Threshold for blue
    mask = cv2.inRange(hsv, (85, 40, 40), (140, 255, 255))

    # Clean up slightly
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    comps = []
    for i in range(1, nlab):
        x, y, w, h, area = stats[i]
        if h > 200 and area > 80:
            comps.append((x, y, w, h, area))

    if len(comps) < 2:
        raise RuntimeError("Could not detect the two blue guide lines in reference.png")

    comps = sorted(comps, key=lambda t: t[0])

    left_line = comps[0]
    right_line = comps[-1]

    xl, yl, wl, hl, _ = left_line
    xr, yr, wr, hr, _ = right_line

    # INNER edges of the vertical blue lines
    x_left_inner = xl + wl - 1
    x_right_inner = xr

    y_top = min(yl, yr)
    y_bottom = max(yl + hl, yr + hr)

    width_px = x_right_inner - x_left_inner
    center_px = 0.5 * (x_left_inner + x_right_inner)

    if width_px <= 0:
        raise RuntimeError("Detected reference width is invalid")

    return {
        "x_left": float(x_left_inner),
        "x_right": float(x_right_inner),
        "y_top": int(y_top),
        "y_bottom": int(y_bottom),
        "width_px": float(width_px),
        "center_px": float(center_px),
    }


def smooth_1d_profile(profile, sigma=4.0):
    prof2d = profile.reshape(1, -1).astype(np.float32)
    smoothed = cv2.GaussianBlur(prof2d, (0, 0), sigmaX=sigma, sigmaY=0)
    return smoothed.ravel()


def find_best_edges_from_profile(profile_smooth, ref_width_px, expected_center,
                                 width_tolerance_px=12,
                                 search_half_windows=(10, 16, 24, 36)):
    grad = np.gradient(profile_smooth)
    n = len(grad)

    expected_left = int(round(expected_center - 0.5 * ref_width_px))
    expected_right = int(round(expected_center + 0.5 * ref_width_px))

    best = None
    best_score = -1e18

    for search_half in search_half_windows:
        lmin = max(1, expected_left - search_half)
        lmax = min(n - 2, expected_left + search_half)

        rmin = max(1, expected_right - search_half)
        rmax = min(n - 2, expected_right + search_half)

        if lmax <= lmin or rmax <= rmin:
            continue

        left_rel = np.argmax(grad[lmin:lmax + 1])
        right_rel = np.argmin(grad[rmin:rmax + 1])

        left_edge = lmin + int(left_rel)
        right_edge = rmin + int(right_rel)

        width = right_edge - left_edge
        if width <= 2:
            continue

        width_err = abs(width - ref_width_px)
        if width_err > width_tolerance_px:
            continue

        edge_strength = grad[left_edge] - grad[right_edge]
        score = edge_strength - 0.25 * width_err

        if score > best_score:
            best_score = score
            best = (left_edge, right_edge, width, score)

    if best is None:
        grad = np.asarray(grad)
        pos_idx = np.argsort(-grad)[:80]
        neg_idx = np.argsort(grad)[:80]

        for left_edge in pos_idx:
            for right_edge in neg_idx:
                if right_edge <= left_edge:
                    continue
                width = right_edge - left_edge
                width_err = abs(width - ref_width_px)
                if width_err > width_tolerance_px:
                    continue

                center = 0.5 * (left_edge + right_edge)
                center_penalty = abs(center - expected_center)

                edge_strength = grad[left_edge] - grad[right_edge]
                score = edge_strength - 0.25 * width_err - 0.15 * center_penalty

                if score > best_score:
                    best_score = score
                    best = (left_edge, right_edge, width, score)

    if best is None:
        raise RuntimeError("Could not detect stable left/right edges from profile")

    left_edge, right_edge, width, score = best
    return float(left_edge), float(right_edge), float(width), float(score)


def detect_wall_box_automatically(gray, ref_box, prev_center=None):
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0).astype(np.float32)

    y0 = int(ref_box["y_top"] + trim_top)
    y1 = int(ref_box["y_bottom"] - trim_bottom)
    ref_width_px = float(ref_box["width_px"])

    if y1 <= y0:
        raise RuntimeError("Invalid y-range after trimming")

    roi = blur[y0:y1, :]
    profile = np.mean(roi, axis=0)
    profile_smooth = smooth_1d_profile(profile, sigma=profile_sigma)

    if prev_center is None:
        prev_center = ref_box["center_px"]

    left_edge, right_edge, width_detected_px, score = find_best_edges_from_profile(
        profile_smooth=profile_smooth,
        ref_width_px=ref_width_px,
        expected_center=prev_center,
        width_tolerance_px=width_tolerance_px,
        search_half_windows=search_half_windows,
    )

    center_final = 0.5 * (left_edge + right_edge)

    # Final rigid box
    x_left_box = center_final - 0.5 * ref_width_px
    x_right_box = center_final + 0.5 * ref_width_px

    return {
        "center_px": float(center_final),
        "width_detected_px": float(width_detected_px),
        "left_detected_px": float(left_edge),
        "right_detected_px": float(right_edge),
        "x_left_box": float(x_left_box),
        "x_right_box": float(x_right_box),
        "y_top": int(ref_box["y_top"]),
        "y_bottom": int(ref_box["y_bottom"]),
        "rows_used": int(y1 - y0),
        "score": float(score),
    }


def draw_box(image_bgr, x_left, x_right, y_top, y_bottom, label):
    out = image_bgr.copy()
    h, w = out.shape[:2]

    x_left = int(round(np.clip(x_left, 0, w - 1)))
    x_right = int(round(np.clip(x_right, 0, w - 1)))
    y_top = int(round(np.clip(y_top, 0, h - 1)))
    y_bottom = int(round(np.clip(y_bottom, 0, h - 1)))

    cv2.line(out, (x_left, y_top), (x_left, y_bottom), box_color, line_thickness)
    cv2.line(out, (x_right, y_top), (x_right, y_bottom), box_color, line_thickness)
    cv2.line(out, (x_left, y_top), (x_right, y_top), box_color, line_thickness)
    cv2.line(out, (x_left, y_bottom), (x_right, y_bottom), box_color, line_thickness)

    cv2.putText(
        out,
        label,
        (max(5, x_left - 5), max(25, y_top - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        box_color,
        2,
        cv2.LINE_AA,
    )

    return out


def crop_around_box(image_bgr, x_left, x_right, y_top, y_bottom):
    h, w = image_bgr.shape[:2]
    xa = max(0, int(np.floor(x_left - crop_margin_x)))
    xb = min(w, int(np.ceil(x_right + crop_margin_x)))
    ya = max(0, int(np.floor(y_top - crop_margin_y)))
    yb = min(h, int(np.ceil(y_bottom + crop_margin_y)))
    return image_bgr[ya:yb, xa:xb]


def save_panel(images_bgr, titles, save_path):
    fig, axes = plt.subplots(
        1, len(images_bgr),
        figsize=(2.5 * len(images_bgr), 11),
        constrained_layout=True
    )
    if len(images_bgr) == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images_bgr, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("Automatic 0.9 mm box tracking from Layer 1 to Layer 7", fontsize=14)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_graphs(layers, shifts_mm, shifts_px, widths_mm, save_dir):
    plt.figure(figsize=(7, 4.5))
    plt.plot(layers, shifts_mm, marker="o", linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.xticks(layers)
    plt.xlabel("Layer")
    plt.ylabel("Box shift relative to Layer 1 (mm)")
    plt.title("Automatic box shift vs layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "shift_vs_layer_mm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(layers, shifts_px, marker="s", linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.xticks(layers)
    plt.xlabel("Layer")
    plt.ylabel("Box shift relative to Layer 1 (pixels)")
    plt.title("Automatic box shift vs layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "shift_vs_layer_px.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(layers, widths_mm, marker="^", linewidth=2)
    plt.axhline(wall_width_mm, linewidth=1)
    plt.xticks(layers)
    plt.xlabel("Layer")
    plt.ylabel("Detected wall width (mm)")
    plt.title("Detected wall width vs layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "detected_width_vs_layer_mm.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    ref_gray, ref_bgr = read_image_gray_bgr(reference_path)
    ref_box = detect_reference_box_from_blue_lines(ref_bgr)

    mm_per_px = wall_width_mm / ref_box["width_px"]

    ref_box_width_px = ref_box["x_right"] - ref_box["x_left"]
    ref_box_height_px = ref_box["y_bottom"] - ref_box["y_top"]
    ref_box_width_mm = ref_box_width_px * mm_per_px
    ref_box_height_mm = ref_box_height_px * mm_per_px

    ref_drawn = draw_box(
        ref_bgr,
        ref_box["x_left"],
        ref_box["x_right"],
        ref_box["y_top"],
        ref_box["y_bottom"],
        "Reference",
    )
    cv2.imwrite(str(out_dir / "reference_detected_box.png"), ref_drawn)

    panel_imgs = []
    panel_titles = []
    layers = []
    shifts_px = []
    shifts_mm = []
    widths_mm = []
    summary_lines = []

    summary_lines.append(f"Reference width = {ref_box['width_px']:.2f} px")
    summary_lines.append(f"Scale = {mm_per_px:.6f} mm/px")
    summary_lines.append(f"Reference center = {ref_box['center_px']:.2f} px")
    summary_lines.append(f"Reference box y-range = {ref_box['y_top']} to {ref_box['y_bottom']}")
    summary_lines.append(
        f"Reference box size = "
        f"{ref_box_width_px:.2f} px × {ref_box_height_px:.2f} px "
        f"= {ref_box_width_mm:.4f} mm × {ref_box_height_mm:.4f} mm"
    )
    summary_lines.append("")

    print("\n================ REFERENCE BOX ================")
    print(f"Reference center      : {ref_box['center_px']:.2f} px")
    print(f"Reference width       : {ref_box_width_px:.2f} px  ({ref_box_width_mm:.4f} mm)")
    print(f"Reference height      : {ref_box_height_px:.2f} px  ({ref_box_height_mm:.4f} mm)")
    print(f"Reference y-range     : {ref_box['y_top']} to {ref_box['y_bottom']}")
    print(f"Scale                 : {mm_per_px:.6f} mm/px")
    print("================================================\n")

    center_layer1 = None
    prev_center = ref_box["center_px"]

    for i, layer_path in enumerate(layer_paths, start=1):
        gray, bgr = read_image_gray_bgr(layer_path)

        det = detect_wall_box_automatically(
            gray=gray,
            ref_box=ref_box,
            prev_center=prev_center,
        )

        prev_center = det["center_px"]

        if center_layer1 is None:
            center_layer1 = det["center_px"]

        shift_px = det["center_px"] - center_layer1
        shift_mm = shift_px * mm_per_px
        width_mm = det["width_detected_px"] * mm_per_px

        # exact final drawn box dimensions
        box_width_px = det["x_right_box"] - det["x_left_box"]
        box_height_px = det["y_bottom"] - det["y_top"]
        box_width_mm = box_width_px * mm_per_px
        box_height_mm = box_height_px * mm_per_px

        label = f"Layer {i}"

        boxed_full = draw_box(
            bgr,
            det["x_left_box"],
            det["x_right_box"],
            det["y_top"],
            det["y_bottom"],
            label,
        )

        boxed_crop = crop_around_box(
            boxed_full,
            det["x_left_box"],
            det["x_right_box"],
            det["y_top"],
            det["y_bottom"],
        )

        cv2.imwrite(str(out_dir / f"layer{i}_boxed_full.png"), boxed_full)
        cv2.imwrite(str(out_dir / f"layer{i}_boxed_crop.png"), boxed_crop)

        panel_imgs.append(boxed_crop)
        panel_titles.append(label)

        layers.append(i)
        shifts_px.append(shift_px)
        shifts_mm.append(shift_mm)
        widths_mm.append(width_mm)

        summary_line = (
            f"{label}: "
            f"center={det['center_px']:.2f} px, "
            f"shift_vs_Layer1={shift_px:+.2f} px ({shift_mm:+.4f} mm), "
            f"detected_width={det['width_detected_px']:.2f} px ({width_mm:.4f} mm), "
            f"box_size={box_width_px:.2f} px x {box_height_px:.2f} px "
            f"({box_width_mm:.4f} mm x {box_height_mm:.4f} mm), "
            f"left_detected={det['left_detected_px']:.2f}, "
            f"right_detected={det['right_detected_px']:.2f}, "
            f"box_left={det['x_left_box']:.2f}, "
            f"box_right={det['x_right_box']:.2f}, "
            f"profile_score={det['score']:.2f}"
        )
        summary_lines.append(summary_line)

        print(f"{label}")
        print(f"  center              : {det['center_px']:.2f} px")
        print(f"  shift vs Layer 1    : {shift_px:+.2f} px  ({shift_mm:+.4f} mm)")
        print(f"  detected wall width : {det['width_detected_px']:.2f} px  ({width_mm:.4f} mm)")
        print(f"  blue box width      : {box_width_px:.2f} px  ({box_width_mm:.4f} mm)")
        print(f"  blue box height     : {box_height_px:.2f} px  ({box_height_mm:.4f} mm)")
        print(f"  detected left edge  : {det['left_detected_px']:.2f} px")
        print(f"  detected right edge : {det['right_detected_px']:.2f} px")
        print(f"  box left            : {det['x_left_box']:.2f} px")
        print(f"  box right           : {det['x_right_box']:.2f} px")
        print(f"  profile score       : {det['score']:.2f}")
        print("")

    panel_path = out_dir / "boxed_panel_layers1to7.png"
    save_panel(panel_imgs, panel_titles, panel_path)

    summary_path = out_dir / "automatic_tracking_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    save_graphs(layers, shifts_mm, shifts_px, widths_mm, out_dir)

    print(f"Saved outputs in: {out_dir}")
    print(f"Reference box image: {out_dir / 'reference_detected_box.png'}")
    print(f"Panel: {panel_path}")
    print(f"Summary: {summary_path}")
    for i in range(1, 8):
        print(out_dir / f"layer{i}_boxed_crop.png")


if __name__ == "__main__":
    main()