"""
Author: Nikesh Shah

This script loads a video, does drone detection using a YOLO finetuned model then outputs the inferred video file.
The script uses a built in tracker (through simple IoU tracker)
"""
# === Consolidated imports (from original notebook) ===
import sys
import sys, numpy as np, scipy, cv2, torch
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
from pathlib import Path
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
import time                    # NEW: timing
from collections import deque  # NEW: rolling FPS
from IPython.display import HTML
from base64 import b64encode
import cv2, torch
cv2.setNumThreads(0)
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # harmless for inference

## ---------------UPDATE THE PATHS----------------------
SOURCE_VIDEO_PATH = Path(HOME) / "anduril_swarm.mp4"
SOURCE_WEIGHTS_PATH = Path(HOME) / "yolov11n-UAV-finetune.pt"

try:
    get_ipython  # type: ignore
    IPYTHON_AVAILABLE = True
except Exception:
    IPYTHON_AVAILABLE = False

def _ipython_magic_safe(line: str):
    if not line.startswith("%"):
        return line
    # In script mode, ignore IPython magic
    return f"# [ignored IPython magic in .py] {line}"

HOME = os.getcwd()
print("HOME:", HOME)

MODEL = YOLO(SOURCE_WEIGHTS_PATH)
MODEL.to("cuda:0")
MODEL.fuse()                 # fuse Conv+BN where possible
try:
    MODEL.model.half()       # enforce FP16 inside the model
except Exception:
    pass
try:
    # optional: better matching if SciPy is available
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
## ---------- NEW TRACKER LOGIC -------------
# --- Simple IoU tracker (no external deps) 
@dataclass
class _Track:
    bbox: np.ndarray          # [x1,y1,x2,y2]
    class_id: Optional[int]
    hits: int = 0
    misses: int = 0

class SimpleIOUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30, class_aware: bool = True):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.class_aware = class_aware
        self._next_id = 1
        self.tracks: Dict[int, _Track] = {}

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # a: Mx4, b: Nx4 in xyxy
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        tl = np.maximum(a[:, None, :2], b[None, :, :2])
        br = np.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = np.clip(br - tl, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)

    def update(self, detections) -> "sv.Detections":
        # Prepare inputs
        det_boxes = detections.xyxy.astype(np.float32)
        det_cls   = getattr(detections, "class_id", None)

        # Step 1: build IoU cost between existing tracks and detections
        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[t].bbox for t in track_ids], dtype=np.float32) if track_ids else np.zeros((0,4), np.float32)
        iou = self._iou_matrix(track_boxes, det_boxes)

        # Optional class-aware gating
        if self.class_aware and det_cls is not None and len(track_ids) and len(det_boxes):
            gate = np.zeros_like(iou, dtype=bool)
            for r, tid in enumerate(track_ids):
                gate[r] = (det_cls == self.tracks[tid].class_id)
            iou = np.where(gate, iou, 0.0)

        # Step 2: assign
        matches, unmatched_tracks, unmatched_dets = [], set(range(len(track_ids))), set(range(len(det_boxes)))
        if len(track_boxes) and len(det_boxes):
            if _HAS_SCIPY:
                # Hungarian on 1-IOU, then filter by threshold
                cost = 1.0 - iou
                rr, cc = linear_sum_assignment(cost)
                for r, c in zip(rr, cc):
                    if iou[r, c] >= self.iou_thresh:
                        matches.append((r, c))
                        unmatched_tracks.discard(r)
                        unmatched_dets.discard(c)
            else:
                # Greedy fallback
                iou_copy = iou.copy()
                while True:
                    r, c = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
                    if iou_copy[r, c] < self.iou_thresh:
                        break
                    matches.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)
                    iou_copy[r, :] = -1
                    iou_copy[:, c] = -1

        # Step 3: update matched tracks
        for r, c in matches:
            tid = track_ids[r]
            self.tracks[tid].bbox = det_boxes[c]
            self.tracks[tid].hits += 1
            self.tracks[tid].misses = 0
            if det_cls is not None:
                self.tracks[tid].class_id = int(det_cls[c])

        # Step 4: increase miss count on unmatched tracks
        for r in unmatched_tracks:
            tid = track_ids[r]
            self.tracks[tid].misses += 1

        # Step 5: create new tracks for unmatched detections
        for c in unmatched_dets:
            cls_id = int(det_cls[c]) if det_cls is not None else None
            self.tracks[self._next_id] = _Track(bbox=det_boxes[c], class_id=cls_id, hits=1, misses=0)
            self._next_id += 1

        # Step 6: remove stale tracks
        for tid in [t for t, tr in self.tracks.items() if tr.misses > self.max_age]:
            del self.tracks[tid]

        # Step 7: write tracker_id back into Detections
        tracker_ids = np.full(len(det_boxes), -1, dtype=int)
        # we know IDs for matched + new (unmatched_dets just created)
        # Build map from bbox index to id
        # For matches:
        for r, c in matches:
            tracker_ids[c] = track_ids[r]
        # For newly created:
        # find det indices in unmatched_dets; they map to last created ids in insertion order
        # to avoid ambiguity, re-scan by IoU with each live track picking exact det bbox
        live_ids = list(self.tracks.keys())
        live_boxes = np.array([self.tracks[t].bbox for t in live_ids], dtype=np.float32)
        if len(det_boxes) and len(live_boxes):
            iou_live = self._iou_matrix(det_boxes, live_boxes)  # det x live
            best_live = iou_live.argmax(axis=1)
            for di in list(unmatched_dets):
                tid = live_ids[int(best_live[di])]
                # only assign if that live box equals the det box (avoid accidental reassignment)
                if np.allclose(self.tracks[tid].bbox, det_boxes[di], atol=1e-3):
                    tracker_ids[di] = tid

        detections.tracker_id = tracker_ids
        return detections

# ------- DRONE TRAIL LOGIC ---------
# --- NEW: simple per-ID trail memory + prediction -------------------------

@dataclass
class _TrailState:
    bbox: np.ndarray            # last bbox [x1,y1,x2,y2]
    center: Tuple[int, int]     # last center (cx, cy)
    vel: Tuple[float, float]    # EMA velocity (vx, vy) px/frame
    history: list               # list[(cx,cy)]
    age: int                    # missed frames

class TrailMemory:
    def __init__(self, history_len: int = 14, max_age: int = 20, ema_alpha: float = 0.3, max_step: int = 25):
        self.history_len = history_len
        self.max_age = max_age
        self.ema_alpha = ema_alpha
        self.max_step = max_step
        self.tracks: Dict[int, _TrailState] = {}

    @staticmethod
    def _center_xyxy(b: np.ndarray) -> Tuple[int, int]:
        x1, y1, x2, y2 = b
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(self, detections) -> Dict[int, _TrailState]:
        # mark all as missed
        for st in self.tracks.values():
            st.age += 1

        if len(detections) > 0 and getattr(detections, "tracker_id", None) is not None:
            boxes = detections.xyxy.astype(np.float32)
            tids  = detections.tracker_id.astype(int)

            for i, tid in enumerate(tids):
                if tid < 0:
                    # unassigned; skip to avoid jumpy ghost trails
                    continue
                b = boxes[i]
                cx, cy = self._center_xyxy(b)

                if tid not in self.tracks:
                    self.tracks[tid] = _TrailState(
                        bbox=b.copy(),
                        center=(cx, cy),
                        vel=(0.0, 0.0),
                        history=[(cx, cy)],
                        age=0
                    )
                else:
                    st = self.tracks[tid]
                    # delta since last
                    vx = float(cx - st.center[0])
                    vy = float(cy - st.center[1])
                    # clamp extreme leaps to reduce outliers
                    vx = max(-self.max_step, min(self.max_step, vx))
                    vy = max(-self.max_step, min(self.max_step, vy))
                    # simple reversal rejection: if huge reversal vs prior motion, keep previous vel
                    pvx, pvy = st.vel
                    prev_mag = (pvx*pvx + pvy*pvy) ** 0.5
                    cur_mag  = (vx*vx + vy*vy) ** 0.5
                    if prev_mag > 2.0 and cur_mag > 6.0:
                        dot = pvx*vx + pvy*vy
                        cosang = dot / (prev_mag * cur_mag + 1e-6)
                        if cosang < -0.34:  # ~ >110° reversal
                            vx, vy = pvx, pvy

                    # EMA smoothing
                    a = self.ema_alpha
                    st.vel = (a*vx + (1-a)*st.vel[0], a*vy + (1-a)*st.vel[1])
                    st.center = (cx, cy)
                    st.history.append(st.center)
                    st.bbox = b.copy()
                    st.age = 0
                    if len(st.history) > self.history_len:
                        st.history = st.history[-self.history_len:]

        # prune stale
        for tid in [t for t,s in self.tracks.items() if s.age > self.max_age]:
            del self.tracks[tid]
        return self.tracks

def draw_trails_and_prediction(
    frame: np.ndarray,
    tracks: Dict[int, _TrailState],
    predict_steps: int = 10,
    trail_color=(0, 255, 0)  # BGR
) -> np.ndarray:
    overlay = frame.copy()
    for tid, st in tracks.items():
        pts = st.history
        if len(pts) >= 2:
            steps = len(pts) - 1
            for i in range(1, len(pts)):
                p0, p1 = pts[i-1], pts[i]
                fade = i / (steps + 1)  # newer segments brighter
                c = (
                    int(trail_color[0] * fade),
                    int(trail_color[1] * fade),
                    int(trail_color[2] * fade),
                )
                thickness = 2 if i > steps // 2 else 1
                cv2.line(overlay, p0, p1, c, thickness, lineType=cv2.LINE_AA)

        # dashed prediction vector from last position along smoothed velocity
        cx, cy = st.center
        vx, vy = st.vel
        mag = max(1.0, (vx*vx + vy*vy) ** 0.5)
        ux, uy = vx/mag, vy/mag
        length = 10.0 * predict_steps
        dash_len, gap_len = 8, 6
        total = int(length)
        pos = 0
        while pos < total:
            seg = min(dash_len, total - pos)
            sx = int(cx + ux*pos); sy = int(cy + uy*pos)
            ex = int(cx + ux*(pos+seg)); ey = int(cy + uy*(pos+seg))
            # small shadow for readability
            cv2.line(overlay, (sx+1, sy+1), (ex+1, ey+1), (0,0,0), 2, cv2.LINE_AA)
            cv2.line(overlay, (sx, sy), (ex, ey), (0,255,160), 2, cv2.LINE_AA)
            pos += dash_len + gap_len
        # tiny arrowhead
        ex = int(cx + ux*length); ey = int(cy + uy*length)
        th = np.deg2rad(28); L = 14
        cos_t, sin_t = np.cos(th), np.sin(th)
        # wings
        wx1 = (-ux*cos_t + uy*sin_t) * L
        wy1 = (-ux*sin_t - uy*cos_t) * L
        wx2 = (-ux*cos_t - uy*sin_t) * L
        wy2 = ( ux*sin_t - uy*cos_t) * L
        p_end = (ex, ey); p_w1 = (int(ex+wx1), int(ey+wy1)); p_w2 = (int(ex+wx2), int(ey+wy2))
        cv2.line(overlay, (p_end[0]+1,p_end[1]+1), (p_w1[0]+1,p_w1[1]+1), (0,0,0), 2, cv2.LINE_AA)
        cv2.line(overlay, (p_end[0]+1,p_end[1]+1), (p_w2[0]+1,p_w2[1]+1), (0,0,0), 2, cv2.LINE_AA)
        cv2.line(overlay, p_end, p_w1, (0,255,160), 2, cv2.LINE_AA)
        cv2.line(overlay, p_end, p_w2, (0,255,160), 2, cv2.LINE_AA)

        # minimal ID tag
        #cv2.putText(overlay, f"ID {tid}", (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

# --- Rolling FPS + model-only time -------------------------------
class FPSMeter:
    def __init__(self, window: int = 60):
        self.window = window
        self.pipe_times = deque(maxlen=window)
        self.model_times = deque(maxlen=window)
        self.start = time.time()
        self.frames = 0

    def update(self, dt_pipeline: float, dt_model: float):
        self.pipe_times.append(dt_pipeline)
        self.model_times.append(dt_model)
        self.frames += 1

    @property
    def fps(self) -> float:
        return 0.0 if not self.pipe_times else len(self.pipe_times) / sum(self.pipe_times)

    @property
    def model_ms(self) -> float:
        return 0.0 if not self.model_times else 1000.0 * (sum(self.model_times) / len(self.model_times))

    def log(self, every: int = 60):
        if self.frames and self.frames % every == 0:
            elapsed = time.time() - self.start
            avg = self.frames / max(elapsed, 1e-6)
            print(f"[{self.frames} frames] pipeline FPS (avg): {avg:.2f} | window FPS: {self.fps:.2f} | model: {self.model_ms:.1f} ms")

FPS = FPSMeter(window=60)
# ----------------------------------------------------------------------

tracker = SimpleIOUTracker(iou_thresh=0.3, max_age=30, class_aware=True)
trails  = TrailMemory(history_len=14, max_age=20, ema_alpha=0.3, max_step=25)

TARGET_VIDEO_PATH = SOURCE_VIDEO_PATH.parent / f"{SOURCE_VIDEO_PATH.stem}-result{SOURCE_VIDEO_PATH.suffix}"
TARGET_VIDEO_PATH_COMPRESSED = TARGET_VIDEO_PATH.parent / f"{TARGET_VIDEO_PATH.stem}-compressed{TARGET_VIDEO_PATH.suffix}"

box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)

label_annotator = sv.LabelAnnotator(
    color=sv.Color.RED,
    text_color=sv.Color.WHITE,
    text_scale=0.45,
    text_thickness=1,
    text_padding=3,
    border_radius=3,
    text_position=sv.Position.TOP_CENTER
)

# def slicer_callback(image_slice: np.ndarray) -> sv.Detections:
#     result = MODEL(image_slice, conf=0.10, verbose=False)[0]
#     return sv.Detections.from_ultralytics(result)

# slicer = sv.InferenceSlicer(callback=slicer_callback, slice_wh=(640, 640), iou_threshold=0.3)

# ----------------- Batched tiler + merge (replaces InferenceSlicer) -----------------
def _tile_frame(frame: np.ndarray, tile_wh=(640, 640)):
    """Return list of 640x640 tiles (zero-padded at borders) and their (x,y) offsets."""
    H, W = frame.shape[:2]
    tw, th = tile_wh
    tiles, offsets = [], []
    for y in range(0, H, th):
        for x in range(0, W, tw):
            crop = frame[y:min(y+th, H), x:min(x+tw, W)]
            if crop.shape[0] != th or crop.shape[1] != tw:
                pad = np.zeros((th, tw, 3), dtype=frame.dtype)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            tiles.append(crop)
            offsets.append((x, y))
    return tiles, offsets

def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.30) -> np.ndarray:
    """Simple class-agnostic NMS on CPU."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=int)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        ovr = inter / (area_i + area_o - inter + 1e-9)
        order = order[1:][ovr <= iou_thr]
    return np.array(keep, dtype=int)

## ------------Inference-----------
def batched_infer(frame: np.ndarray, conf: float = 0.15, iou_merge: float = 0.30):
    """Tile → one batched forward → merge back to frame coords with NMS."""
    tiles, offsets = _tile_frame(frame, (640, 640))

    t0 = time.time()
    with torch.inference_mode():
        results = MODEL(tiles, device=0, half=True, conf=conf, verbose=False)
    strict_model_time = time.time() - t0

    all_boxes, all_scores, all_classes = [], [], []
    for r, (ox, oy) in zip(results, offsets):
        # r is already a `Results` object for this tile
        det = sv.Detections.from_ultralytics(r)
        if len(det) == 0:
            continue

        b = det.xyxy.copy()
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        all_boxes.append(b)
        all_scores.append(det.confidence)
        if getattr(det, "class_id", None) is not None:
            all_classes.append(det.class_id)

    if not all_boxes:
        empty = sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
        )
        return empty, strict_model_time

    boxes = np.vstack(all_boxes).astype(np.float32)
    scores = np.concatenate(all_scores).astype(np.float32)
    classes = np.concatenate(all_classes).astype(int) if all_classes else None

    keep = _nms_numpy(boxes, scores, iou_thr=iou_merge)
    merged = sv.Detections(
        xyxy=boxes[keep],
        confidence=scores[keep],
        class_id=(classes[keep] if classes is not None else None),
    )
    return merged, strict_model_time

    
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    t0_pipe = time.time()

    detections, t_model_strict = batched_infer(frame, conf=0.15, iou_merge=0.30)
    detections = detections[detections.area < 2000]
    detections = tracker.update(detections)

    if len(detections) == 0:
        out = frame.copy()
    else:
        track_states = trails.update(detections)
        trail_overlayed = draw_trails_and_prediction(frame, track_states, predict_steps=10)
        names = getattr(MODEL, "names", None) or getattr(MODEL.model, "names", {})
        def _name_from_id(cid: int) -> str:
            if isinstance(names, dict):
                return str(names.get(int(cid), "obj"))
            if isinstance(names, (list, tuple)) and 0 <= int(cid) < len(names):
                return str(names[int(cid)])
            return "obj"
        
        if getattr(detections, "class_id", None) is not None:
            labels = [
                f"{_name_from_id(cid).upper()} {str(f'{conf:.2f}').lstrip('0')}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]
        else:
            # no class ids – just show confidence
            labels = [f"OBJ {str(f'{conf:.2f}').lstrip('0')}" for conf in detections.confidence]
        out = box_annotator.annotate(scene=trail_overlayed, detections=detections)
        out = label_annotator.annotate(scene=out, detections=detections, labels=labels)

    # HUD
    FPS.update(time.time() - t0_pipe, t_model_strict)
    FPS.log(every=60)
    hud = f"FPS {FPS.fps:.1f} | model {FPS.model_ms:.0f} ms | objs {int(len(detections))}"
    (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(out, (8, 8), (8 + tw + 12, 8 + th + 12), (0, 0, 0), -1)
    cv2.putText(out, hud, (14, 8 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)

if __name__ == "__main__":
    try:
        main  # type: ignore
        main()
    except NameError:
        pass

