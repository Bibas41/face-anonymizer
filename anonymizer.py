"""
Face Anonymizer
===============
Detects human faces in images, video files or a webcam stream using the
MediaPipe Face Detector (BlazeFace short-range model) and anonymizes every
detected face with blur, pixelation or a solid mask.

Examples
--------
    python anonymizer.py --input test_media/photo.jpg
    python anonymizer.py --input test_media/photo.jpg --method pixelate --show
    python anonymizer.py --input test_media/photo.jpg --method all
    python anonymizer.py --input test_media/clip.mp4 --method blur
    python anonymizer.py --input test_media            (whole folder)
    python anonymizer.py --webcam
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
METHODS = ["blur", "pixelate", "mask"]

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "blaze_face_short_range.tflite"
DEFAULT_OUTPUT = PROJECT_DIR / "output"

vision = mp.tasks.vision


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
def create_detector(model_path, min_confidence):
    """Create a MediaPipe FaceDetector.

    IMAGE running mode is used for photos and for every video frame. The
    face detector does not track faces between frames, so VIDEO mode would
    give no benefit - and IMAGE mode lets the same detector also run on the
    smaller tiles used for distant faces (see detect_faces).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Download it into the 'models' folder (see README.md)."
        )
    options = vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=min_confidence,
    )
    return vision.FaceDetector.create_from_options(options)


def _detect_once(detector, image_bgr, offset_x=0, offset_y=0):
    """Run the detector on one image (or tile) and return boxes in
    full-frame coordinates as (x, y, w, h, score)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)   # OpenCV=BGR, MediaPipe=RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    boxes = []
    for det in result.detections:
        bb = det.bounding_box
        score = det.categories[0].score if det.categories else 0.0
        boxes.append((bb.origin_x + offset_x, bb.origin_y + offset_y,
                      bb.width, bb.height, score))
    return boxes


def _iou(a, b):
    """Intersection-over-union of two (x, y, w, h, ...) boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _mostly_inside(inner, outer, ratio=0.6):
    """True if most of `inner` lies inside `outer` (e.g. half a face that
    was cut by a tile border, while the full face was found elsewhere)."""
    iw = max(0, min(inner[0] + inner[2], outer[0] + outer[2]) - max(inner[0], outer[0]))
    ih = max(0, min(inner[1] + inner[3], outer[1] + outer[3]) - max(inner[1], outer[1]))
    area = inner[2] * inner[3]
    return area > 0 and iw * ih / area > ratio


def _merge_duplicates(boxes, iou_threshold=0.3):
    """Non-maximum suppression: when the same face is found in the full
    frame and in a tile, keep only the highest-scoring box. Partial boxes
    that lie mostly inside an already kept box are dropped too."""
    kept = []
    for box in sorted(boxes, key=lambda b: b[4], reverse=True):
        if all(_iou(box, k) < iou_threshold and not _mostly_inside(box, k)
               for k in kept):
            kept.append(box)
    return kept


def detect_faces(detector, frame_bgr, tiles=2):
    """Return a list of (x, y, w, h, score) boxes for all faces in the frame.

    The BlazeFace short-range model shrinks the whole frame to 128x128
    pixels, so a face that is 3 m away becomes only a few pixels and is
    missed. To help with this, the frame is also split into a tiles x tiles
    grid of overlapping crops; each crop is scaled up by the model, so
    distant faces become large enough to detect. Results are merged.
    tiles=0 or 1 disables tiling (faster, fine for close-up webcam use).
    """
    boxes = _detect_once(detector, frame_bgr)

    if tiles >= 2:
        h, w = frame_bgr.shape[:2]
        tile_w = min(w, int(w / tiles * 1.4))       # 40 % larger -> overlap
        tile_h = min(h, int(h / tiles * 1.4))
        step_x = (w - tile_w) / (tiles - 1)
        step_y = (h - tile_h) / (tiles - 1)
        for row in range(tiles):
            for col in range(tiles):
                x0 = int(col * step_x)
                y0 = int(row * step_y)
                tile = frame_bgr[y0:y0 + tile_h, x0:x0 + tile_w]
                boxes += _detect_once(detector, tile, x0, y0)

    return _merge_duplicates(boxes)


def expand_box(box, padding, img_w, img_h):
    """Enlarge a face box so it also covers hair, forehead, ears and jaw.

    BlazeFace boxes are fairly tight (roughly eyebrows to chin), so extra
    padding is added, with a bit more at the top for the forehead/hairline.
    The result is clipped to the image borders so faces at the edge of the
    frame do not cause errors.
    """
    x, y, w, h = box[:4]
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - int(pad_y * 1.5))
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Anonymization methods
# ---------------------------------------------------------------------------
def blur_region(roi, strength):
    """Gaussian blur whose kernel size scales with the face size.

    A fixed small kernel would leave large faces recognisable, so the kernel
    is a fraction of the face width (strength 0..1). For speed the face is
    shrunk to 64 px, blurred there, and scaled back up - visually the same
    as a huge kernel on the full-size face but many times faster on video.
    """
    h, w = roi.shape[:2]
    small_w = min(64, w)
    small_h = max(1, int(h * small_w / w))
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    k = max(3, int(min(small_w, small_h) * strength) | 1)   # kernel must be odd
    small = cv2.GaussianBlur(small, (k, k), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def pixelate_region(roi, blocks):
    """Shrink the face to blocks x blocks pixels, then scale back up."""
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def mask_region(roi):
    """Replace the face with a solid dark rectangle (fully irreversible)."""
    return np.full_like(roi, 40)


def anonymize(frame, boxes, method, padding, blur_strength, pixel_blocks):
    """Anonymize every face box in the frame. Returns a new image."""
    out = frame.copy()
    img_h, img_w = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = expand_box(box, padding, img_w, img_h)
        if x2 - x1 < 2 or y2 - y1 < 2:          # skip degenerate boxes
            continue
        roi = out[y1:y2, x1:x2]
        if method == "blur":
            out[y1:y2, x1:x2] = blur_region(roi, blur_strength)
        elif method == "pixelate":
            out[y1:y2, x1:x2] = pixelate_region(roi, pixel_blocks)
        else:
            out[y1:y2, x1:x2] = mask_region(roi)
    return out


def draw_debug(frame, boxes, padding):
    """Draw detector boxes (green) and padded anonymization area (red)."""
    out = frame.copy()
    img_h, img_w = frame.shape[:2]
    for box in boxes:
        x, y, w, h, score = box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x1, y1, x2, y2 = expand_box(box, padding, img_w, img_h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"{score:.2f}", (x, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
def process_image(path, args):
    image = cv2.imread(str(path))
    if image is None:
        print(f"[ERROR] Could not read image (corrupt or unsupported): {path}")
        return

    with create_detector(args.model, args.confidence) as detector:
        boxes = detect_faces(detector, image, args.tiles)

    print(f"[IMAGE] {path.name}: {len(boxes)} face(s) detected"
          + ("" if boxes else " - saving unchanged copy, please check manually"))

    methods = METHODS if args.method == "all" else [args.method]
    for method in methods:
        result = anonymize(image, boxes, method, args.padding,
                           args.blur_strength, args.pixel_blocks)
        out_path = args.output / f"{path.stem}_{method}.png"
        cv2.imwrite(str(out_path), result)
        print(f"        saved {out_path}")

        if args.show:
            preview = np.hstack([draw_debug(image, boxes, args.padding), result]) \
                if args.debug else result
            show_scaled(f"{path.name} - {method} (press any key)", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if args.debug:
        debug_path = args.output / f"{path.stem}_debug.png"
        cv2.imwrite(str(debug_path), draw_debug(image, boxes, args.padding))
        print(f"        saved {debug_path}")


# ---------------------------------------------------------------------------
# Video / webcam processing
# ---------------------------------------------------------------------------
def process_stream(cap, args, writer=None, window_name="Face Anonymizer", fps=30.0):
    """Shared loop for video files and webcam.

    Flicker reduction: boxes from the last `hold` frames are kept and also
    anonymized. If the detector misses a face for a frame or two (motion
    blur, turning head), the previous box still covers it.
    """
    method = "blur" if args.method == "all" else args.method
    history = deque(maxlen=max(1, args.hold))
    stats = {"frames": 0, "frames_with_faces": 0, "detections": 0, "max_faces": 0}
    start = time.time()

    with create_detector(args.model, args.confidence) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            boxes = detect_faces(detector, frame, args.tiles)

            stats["frames"] += 1
            stats["detections"] += len(boxes)
            stats["max_faces"] = max(stats["max_faces"], len(boxes))
            if boxes:
                stats["frames_with_faces"] += 1

            history.append(boxes)
            boxes_to_hide = [b for frame_boxes in history for b in frame_boxes]
            result = anonymize(frame, boxes_to_hide, method, args.padding,
                               args.blur_strength, args.pixel_blocks)

            if writer is not None:
                writer.write(result)

            if args.show:
                view = np.hstack([draw_debug(frame, boxes, args.padding), result]) \
                    if args.debug else result
                show_scaled(window_name + " (press q to quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    elapsed = max(time.time() - start, 1e-6)
    print(f"        frames processed : {stats['frames']}")
    print(f"        frames with faces: {stats['frames_with_faces']}")
    print(f"        total detections : {stats['detections']}")
    print(f"        max faces/frame  : {stats['max_faces']}")
    print(f"        processing speed : {stats['frames'] / elapsed:.1f} FPS")


def create_writer(out_path, fps, width, height):
    """Create an MP4 writer, preferring H.264 so the video plays everywhere.

    H.264 ('avc1') plays in browsers, VS Code and GitHub, but is not
    available in every OpenCV build. If it fails, fall back to 'mp4v',
    which always works but only plays in desktop media players.
    """
    log = getattr(cv2, "utils", None) and getattr(cv2.utils, "logging", None)
    old_level = log.getLogLevel() if log else None
    if log:
        log.setLogLevel(log.LOG_LEVEL_SILENT)   # hide noise from the failed try
    try:
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec),
                                     fps, (width, height))
            if writer.isOpened():
                return writer, codec
            writer.release()
        return None, None
    finally:
        if log:
            log.setLogLevel(old_level)


def process_video(path, args):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    method = "blur" if args.method == "all" else args.method
    out_path = args.output / f"{path.stem}_{method}.mp4"
    writer, codec = create_writer(out_path, fps, width, height)
    if writer is None:
        print(f"[ERROR] Could not create output video: {out_path}")
        cap.release()
        return

    print(f"[VIDEO] {path.name}: {width}x{height} @ {fps:.1f} FPS, "
          f"method={method}, codec={codec}")
    process_stream(cap, args, writer, window_name=path.name, fps=fps)
    cap.release()
    writer.release()
    print(f"        saved {out_path}")


def process_webcam(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] No camera found at index {args.camera}. "
              "Is it connected and not used by another app?")
        return

    args.show = True                      # webcam always shows the live view
    writer = None
    if args.save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = args.output / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        writer, codec = create_writer(out_path, 30.0, width, height)
        if writer is None:
            print(f"[ERROR] Could not create output video: {out_path}")
        else:
            print(f"[WEBCAM] recording anonymized output to {out_path} ({codec})")

    print("[WEBCAM] live preview started - press q in the window to quit")
    process_stream(cap, args, writer, window_name="Webcam", fps=30.0)
    cap.release()
    if writer is not None:
        writer.release()


# ---------------------------------------------------------------------------
# Helpers and command line
# ---------------------------------------------------------------------------
def show_scaled(title, image, max_width=1400, max_height=800):
    """Show an image in a window, scaled down if it is larger than the screen."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(title, image)


def process_path(path, args):
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        process_image(path, args)
    elif ext in VIDEO_EXTS:
        process_video(path, args)
    else:
        print(f"[SKIP] Unsupported file type: {path.name} "
              f"(supported: {', '.join(sorted(IMAGE_EXTS | VIDEO_EXTS))})")


def parse_args():
    p = argparse.ArgumentParser(description="Detect and anonymize faces in images, videos or webcam.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="image, video, or folder of files")
    src.add_argument("--webcam", action="store_true", help="use the webcam live")

    p.add_argument("--method", choices=METHODS + ["all"], default="blur",
                   help="anonymization method ('all' saves one image per method)")
    p.add_argument("--confidence", type=float, default=0.4,
                   help="minimum detection confidence 0-1 (lower = catches more faces)")
    p.add_argument("--padding", type=float, default=0.25,
                   help="extra margin around each face as a fraction of its size")
    p.add_argument("--blur-strength", type=float, default=0.8,
                   help="blur kernel size as a fraction of face size (0.1 weak - 1.0 strong)")
    p.add_argument("--pixel-blocks", type=int, default=8,
                   help="number of pixel blocks across a face (fewer = stronger)")
    p.add_argument("--tiles", type=int, default=2,
                   help="also search an N x N grid of zoomed tiles for distant faces "
                        "(0 = off, faster; 2 = default; 3 = very small faces)")
    p.add_argument("--hold", type=int, default=5,
                   help="video: keep boxes from the last N frames to reduce flicker")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output folder")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to .tflite model")
    p.add_argument("--show", action="store_true", help="show results in a window")
    p.add_argument("--debug", action="store_true", help="also show/save detection boxes and scores")
    p.add_argument("--save", action="store_true", help="webcam: save the anonymized recording")
    return p.parse_args()


def main():
    args = parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        sys.exit("[ERROR] --confidence must be between 0 and 1")
    if args.tiles < 0:
        sys.exit("[ERROR] --tiles must be 0 or more")
    if args.pixel_blocks < 1:
        sys.exit("[ERROR] --pixel-blocks must be at least 1")
    if not Path(args.model).exists():
        sys.exit(f"[ERROR] Model file not found: {args.model}\n"
                 "        Download it into the 'models' folder (see README.md).")

    args.output.mkdir(parents=True, exist_ok=True)

    if args.webcam:
        process_webcam(args)
        return

    if not args.input.exists():
        sys.exit(f"[ERROR] Input not found: {args.input}")

    if args.input.is_dir():
        files = sorted(f for f in args.input.iterdir() if f.is_file())
        if not files:
            sys.exit(f"[ERROR] Folder is empty: {args.input}")
        for f in files:
            process_path(f, args)
    else:
        process_path(args.input, args)


if __name__ == "__main__":
    main()
