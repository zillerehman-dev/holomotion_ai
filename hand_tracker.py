from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL = Path("models/hand_landmarker.task")

# MediaPipe connections for skeleton drawing
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


class TrackerError(Exception):
    pass


@dataclass(slots=True)
class Hand:
    slot: int          # 0 or 1 (stable assignment)
    pts: np.ndarray    # (21, 3) float32 in pixel coords (x, y, z)
    handedness: str    # "Left" or "Right"
    score: float
    stale: bool = False


def _download_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("Download complete.")


class HandTracker:
    """MediaPipe HandLandmarker with stable slot assignment and light smoothing."""

    def __init__(self, model_path: str | None = None, max_hands: int = 2):
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.is_file():
            try:
                _download_model(path)
            except Exception as exc:
                raise TrackerError(
                    f"Model not found at {path} and download failed: {exc}"
                ) from exc

        base = mp_python.BaseOptions(model_asset_path=str(path))
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise TrackerError(f"Failed to create HandLandmarker: {exc}") from exc

        self._slots: list[Hand | None] = [None, None]
        self._smooth: list[np.ndarray | None] = [None, None]
        self._frame_ts = 0
        self._alpha = 0.45  # landmark smoothing

    def detect(self, frame_bgr: np.ndarray, now: float) -> list[Hand]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33  # ~30 fps timestamps in ms
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        detected: list[tuple[np.ndarray, str, float]] = []
        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                pts = np.array(
                    [[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks],
                    dtype=np.float32,
                )
                hand_info = result.handedness[i][0]
                label = hand_info.category_name
                score = hand_info.score
                detected.append((pts, label, score))

        # Assign to stable slots by nearest previous palm center
        assigned = self._assign_slots(detected)
        hands: list[Hand] = []
        for slot, item in enumerate(assigned):
            if item is None:
                if self._slots[slot] is not None:
                    # keep last position briefly as stale
                    old = self._slots[slot]
                    self._slots[slot] = Hand(
                        slot=slot, pts=old.pts, handedness=old.handedness,
                        score=old.score, stale=True,
                    )
                continue
            pts, label, score = item
            if self._smooth[slot] is None:
                self._smooth[slot] = pts.copy()
            else:
                self._smooth[slot] = (
                    self._alpha * pts + (1.0 - self._alpha) * self._smooth[slot]
                )
            hand = Hand(
                slot=slot,
                pts=self._smooth[slot].copy(),
                handedness=label,
                score=score,
                stale=False,
            )
            self._slots[slot] = hand
            hands.append(hand)

        # Drop truly missing slots after one stale frame
        for slot in range(2):
            if self._slots[slot] is not None and self._slots[slot].stale:
                if not any(h.slot == slot for h in hands):
                    self._slots[slot] = None
                    self._smooth[slot] = None

        return hands

    def _assign_slots(
        self, detected: list[tuple[np.ndarray, str, float]]
    ) -> list[tuple[np.ndarray, str, float] | None]:
        n = len(detected)
        if n == 0:
            return [None, None]
        if n == 1:
            pts, label, score = detected[0]
            # Prefer matching previous handedness or nearest slot
            best = 0
            if self._slots[1] is not None and self._slots[0] is None:
                best = 1
            elif self._slots[0] is not None and self._slots[1] is not None:
                d0 = np.linalg.norm(pts[0, :2] - self._slots[0].pts[0, :2])
                d1 = np.linalg.norm(pts[0, :2] - self._slots[1].pts[0, :2])
                best = 0 if d0 <= d1 else 1
            out: list = [None, None]
            out[best] = detected[0]
            return out

        # Two hands: Hungarian-style nearest to previous
        out = [None, None]
        used = set()
        for slot in (0, 1):
            if self._slots[slot] is None:
                continue
            best_i, best_d = -1, 1e9
            for i, (pts, _, _) in enumerate(detected):
                if i in used:
                    continue
                d = float(np.linalg.norm(pts[0, :2] - self._slots[slot].pts[0, :2]))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                out[slot] = detected[best_i]
                used.add(best_i)
        # Fill remaining
        for i, item in enumerate(detected):
            if i in used:
                continue
            for slot in (0, 1):
                if out[slot] is None:
                    out[slot] = item
                    break
        return out

    def close(self) -> None:
        self._landmarker.close()


def draw_skeleton(frame: np.ndarray, hand: Hand) -> None:
    if hand.stale:
        return
    pts = hand.pts
    color = (180, 220, 100) if hand.handedness == "Right" else (100, 200, 255)
    for a, b in HAND_CONNECTIONS:
        x0, y0 = int(pts[a, 0]), int(pts[a, 1])
        x1, y1 = int(pts[b, 0]), int(pts[b, 1])
        cv2.line(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    for i in range(21):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        r = 4 if i == 0 else 3
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, color, 1, cv2.LINE_AA)
