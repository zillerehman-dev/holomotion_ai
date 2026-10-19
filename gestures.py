from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np

from hand_tracker import Hand

FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
_PALM_IDS = np.array((0, 5, 9, 13, 17))
_TIP_IDS = (4, 8, 12, 16, 20)


class Gesture(IntEnum):
    OPEN = 0
    RELAXED = 1
    CLAW = 2
    FIST = 3
    PINCH = 4


class Mode(IntEnum):
    ENERGY = 0
    PORTAL = 1
    EARTH = 2
    BEAM = 3
    BLAST = 4          # cinematic charge → aim → launch projectile


MODE_NAMES = {
    Mode.ENERGY: "Energy Core",
    Mode.PORTAL: "Portal",
    Mode.EARTH: "Holo Earth",
    Mode.BEAM: "Energy Beam",
    Mode.BLAST: "Energy Blast",
}

MODE_LIST = (Mode.ENERGY, Mode.PORTAL, Mode.EARTH, Mode.BEAM, Mode.BLAST)


class InteractionState(IntEnum):
    MENU_CLOSED = 0
    MENU_OPEN = 1
    MODE_SELECTED = 2  # brief post-select animation before MENU_CLOSED


class BlastState(IntEnum):
    IDLE = 0
    CHARGING = 1
    READY = 2
    AIMING = 3
    FIRING = 4
    COOLDOWN = 5


@dataclass
class GestureConfig:
    boundaries: tuple[float, float, float] = (0.92, 0.78, 0.58)
    margin: float = 0.03
    pinch_ratio: float = 0.35          # tip distance / palm size
    summon_dist: float = 5.0
    release_dist: float = 14.0
    min_radius_palms: float = 0.8
    radius_scale: float = 1.0
    summon_hold: float = 0.12
    release_hold: float = 0.10
    fist_release_hold: float = 0.35
    cancel_hold: float = 0.10
    lost_hold: float = 0.70
    cooldown: float = 0.55
    # Menu / hands-free selection
    menu_pinch_ratio: float = 0.42     # slightly looser for cursor pinch
    menu_pinch_hold: float = 0.22      # debounce before confirm
    menu_select_cooldown: float = 1.20 # block reopen right after a pick
    menu_close_anim: float = 0.40
    menu_selected_hold: float = 0.85   # show mode name after pick
    # Double closed-hand (fist) to reopen menu
    menu_fist_min_hold: float = 0.12   # how long a close counts as one fist
    menu_double_fist_window: float = 2.0 # max gap between 1st and 2nd close
    menu_post_mode_lock: float = 0.50  # extra lock after entering MENU_CLOSED


def palm_center(pts: np.ndarray) -> np.ndarray:
    return pts[_PALM_IDS, :2].mean(axis=0)


def palm_size(pts: np.ndarray) -> float:
    length = float(np.linalg.norm(pts[9, :2] - pts[0, :2]))
    width = float(np.linalg.norm(pts[5, :2] - pts[17, :2]))
    return max(0.5 * (length + width), 1e-3)


def wrist_angle(pts: np.ndarray) -> float:
    """Approximate wrist orientation from wrist→middle-MCP vector (radians)."""
    v = pts[9, :2] - pts[0, :2]
    return math.atan2(float(v[1]), float(v[0]))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-6:
        return math.pi
    return math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / denom)))


def finger_straightness(pts: np.ndarray, ids: tuple[int, int, int, int]) -> float:
    mcp, pip, dip, tip = (pts[i] for i in ids)
    angles = (_angle(mcp, pip, dip) + _angle(pip, dip, tip)) / (2.0 * math.pi)
    chain = float(
        np.linalg.norm(mcp - pip) + np.linalg.norm(pip - dip) + np.linalg.norm(dip - tip)
    )
    reach = float(np.linalg.norm(tip - mcp)) / chain if chain > 1e-6 else 1.0
    return 0.5 * (angles + min(reach, 1.0))


def hand_straightness(pts: np.ndarray) -> float:
    return float(np.mean([finger_straightness(pts, ids) for ids in FINGERS]))


def is_pinch(pts: np.ndarray, threshold: float) -> bool:
    palm = palm_size(pts)
    dist = float(np.linalg.norm(pts[4, :2] - pts[8, :2]))
    return dist < threshold * palm


def index_tip(pts: np.ndarray) -> np.ndarray:
    return pts[8, :2].copy()


def thumb_tip(pts: np.ndarray) -> np.ndarray:
    return pts[4, :2].copy()


def pinch_distance_ratio(pts: np.ndarray) -> float:
    palm = palm_size(pts)
    dist = float(np.linalg.norm(pts[4, :2] - pts[8, :2]))
    return dist / palm


class GestureClassifier:
    def __init__(self, config: GestureConfig):
        self.config = config
        self.current: Gesture = Gesture.OPEN

    def reset(self) -> None:
        self.current = Gesture.OPEN

    def update(self, straightness: float, pinched: bool) -> Gesture:
        if pinched and straightness > 0.55:
            self.current = Gesture.PINCH
            return self.current
        bounds, margin = self.config.boundaries, self.config.margin
        g = int(self.current) if self.current != Gesture.PINCH else 1
        g = min(g, 3)
        while g < 3 and straightness < bounds[g] - margin:
            g += 1
        while g > 0 and straightness > bounds[g - 1] + margin:
            g -= 1
        self.current = Gesture(g)
        return self.current


@dataclass(slots=True)
class Control:
    mode: Mode = Mode.ENERGY
    active: bool = False
    summon: bool = False
    release: bool = False
    cancel: bool = False
    tracking: bool = False
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    palms: tuple[tuple[float, float], tuple[float, float]] | None = None
    fingertips: list[np.ndarray] = field(default_factory=list)
    wrist_angles: tuple[float, float] = (0.0, 0.0)
    intensity: float = 0.0
    pinch_scale: float = 1.0
    single_hand: bool = False
    hands_present: int = 0
    # Energy Blast (mode 5)
    blast_state: BlastState = BlastState.IDLE
    charge: float = 0.0                    # 0..1
    aim_origin: tuple[float, float] = (0.0, 0.0)
    aim_dir: tuple[float, float] = (0.0, -1.0)
    launch: bool = False


@dataclass
class MenuHit:
    """Hit-test result for holographic mode selector."""
    index: int = -1
    mode: Mode | None = None
    rect: tuple[int, int, int, int] | None = None  # x0,y0,x1,y1


class ModeMenu:
    """Hands-free holographic mode selector driven by index fingertip + pinch.

    Reopen only via double closed-hand (fist twice). After a selection the menu
    stays closed so the active mode can run until the user intentionally returns.
    """

    def __init__(self, config: GestureConfig | None = None):
        self.config = config or GestureConfig()
        self.state = InteractionState.MENU_OPEN
        self.cursor: tuple[float, float] | None = None
        self.hover = MenuHit()
        self.selected: Mode | None = None
        self.anim = 1.0          # 1 = fully open, 0 = closed
        self.pinch_progress = 0.0
        self.fist_count = 0      # 0, 1, or 2 (reopen at 2)
        self._pinch_since: float | None = None
        self._pinch_armed = True  # require release after select before next pinch
        self._fist_since: float | None = None
        self._fist_was = False
        self._fist_counted = False
        self._first_fist_at: float | None = None
        self._cooldown_until = 0.0
        self._selected_at = 0.0
        self._hand_lost_frames = 0
        self._layout: list[tuple[Mode, int, int, int, int]] = []  # mode,x0,y0,x1,y1
        self._frame_w = 640
        self._frame_h = 480

    def set_frame_size(self, w: int, h: int) -> None:
        self._frame_w, self._frame_h = w, h
        self._rebuild_layout()

    def _rebuild_layout(self) -> None:
        w, h = self._frame_w, self._frame_h
        n = len(MODE_LIST)
        box_w = max(140, int(w * 0.22))
        box_h = max(36, int(h * 0.07))
        gap = max(10, int(h * 0.018))
        total_h = n * box_h + (n - 1) * gap
        y0 = max(int(h * 0.18), (h - total_h) // 2 - int(h * 0.05))
        x0 = (w - box_w) // 2
        self._layout = []
        for i, mode in enumerate(MODE_LIST):
            y = y0 + i * (box_h + gap)
            self._layout.append((mode, x0, y, x0 + box_w, y + box_h))

    def _reset_fist_tracker(self) -> None:
        self.fist_count = 0
        self._fist_since = None
        self._fist_was = False
        self._fist_counted = False
        self._first_fist_at = None

    def open(self, now: float) -> None:
        if now < self._cooldown_until:
            return
        self.state = InteractionState.MENU_OPEN
        self.anim = 0.0
        self.selected = None
        self.hover = MenuHit()
        self._pinch_since = None
        self._pinch_armed = True
        self.pinch_progress = 0.0
        self._reset_fist_tracker()

    def force_open(self) -> None:
        self.state = InteractionState.MENU_OPEN
        self.anim = 1.0
        self.selected = None
        self.hover = MenuHit()
        self._pinch_since = None
        self._pinch_armed = True
        self.pinch_progress = 0.0
        self._cooldown_until = 0.0
        self._reset_fist_tracker()

    def update(
        self,
        hands: list[Hand],
        gestures: dict[int, Gesture],
        now: float,
        dt: float,
    ) -> Mode | None:
        """Returns a newly selected Mode once, else None. Owns gestures while open."""
        cfg = self.config
        live = [h for h in hands if not h.stale]
        newly: Mode | None = None

        # Animate open / close
        if self.state == InteractionState.MENU_OPEN:
            self.anim = min(1.0, self.anim + dt / 0.28)
        elif self.state == InteractionState.MODE_SELECTED:
            self.anim = max(0.0, self.anim - dt / max(cfg.menu_close_anim, 0.05))
            if now - self._selected_at >= cfg.menu_selected_hold:
                self.state = InteractionState.MENU_CLOSED
                self.anim = 0.0
                # Long lock so the chosen mode can run; reopen only via double fist later
                self._cooldown_until = now + cfg.menu_select_cooldown + cfg.menu_post_mode_lock
                self._reset_fist_tracker()
        else:
            self.anim = max(0.0, self.anim - dt / max(cfg.menu_close_anim, 0.05))

        # --- MENU_CLOSED: mode is active; reopen only with two closed-hand gestures ---
        if self.state == InteractionState.MENU_CLOSED:
            self.cursor = None
            self.hover = MenuHit()
            self._pinch_since = None
            self.pinch_progress = 0.0
            self._update_double_fist(live, gestures, now)
            if (
                self.fist_count >= 2
                and now >= self._cooldown_until
            ):
                self.open(now)
            return None

        # --- MODE_SELECTED: wait / animate only; no selection or reopen ---
        if self.state == InteractionState.MODE_SELECTED:
            self.cursor = None
            self._pinch_since = None
            self.pinch_progress = 0.0
            return None

        # --- MENU_OPEN: cursor + pinch (one-shot; must release before next pick) ---
        primary = None
        if live:
            primary = max(live, key=lambda h: h.score)

        if primary is None:
            self._hand_lost_frames += 1
            self.cursor = None
            self.hover = MenuHit()
            self._pinch_since = None
            self.pinch_progress = 0.0
            return None

        self._hand_lost_frames = 0
        tip = index_tip(primary.pts)
        self.cursor = (float(tip[0]), float(tip[1]))
        self.hover = self._hit_test(self.cursor[0], self.cursor[1])

        pinched = is_pinch(primary.pts, cfg.menu_pinch_ratio)
        if not pinched:
            self._pinch_armed = True
            self._pinch_since = None
            self.pinch_progress = 0.0
        elif self._pinch_armed and self.hover.mode is not None:
            if self._pinch_since is None:
                self._pinch_since = now
            held = now - self._pinch_since
            self.pinch_progress = min(1.0, held / max(cfg.menu_pinch_hold, 0.01))
            if held >= cfg.menu_pinch_hold:
                newly = self.hover.mode
                self.selected = newly
                self.state = InteractionState.MODE_SELECTED
                self._selected_at = now
                self._pinch_since = None
                self.pinch_progress = 0.0
                self._pinch_armed = False  # must fully release before another select
                self._cooldown_until = now + cfg.menu_select_cooldown + cfg.menu_post_mode_lock
                self._reset_fist_tracker()
        else:
            # Pinched but not armed or not over an item
            if not self._pinch_armed:
                self.pinch_progress = 0.0
            self._pinch_since = None

        return newly

    def _update_double_fist(
        self,
        live: list[Hand],
        gestures: dict[int, Gesture],
        now: float,
    ) -> None:
        """Count two separate closed-hand cycles (close → open twice) to reopen menu."""
        cfg = self.config
        # Expire partial sequence if user waited too long between closes
        if (
            self._first_fist_at is not None
            and self.fist_count == 1
            and now - self._first_fist_at > cfg.menu_double_fist_window
        ):
            self.fist_count = 0
            self._first_fist_at = None
            self._fist_since = None
            self._fist_was = False
            self._fist_counted = False

        any_fist = False
        if live:
            for h in live:
                if gestures.get(h.slot) is Gesture.FIST:
                    any_fist = True
                    break

        if any_fist:
            if not self._fist_was:
                self._fist_since = now
                self._fist_counted = False
            self._fist_was = True
            if (
                not self._fist_counted
                and self._fist_since is not None
                and now - self._fist_since >= cfg.menu_fist_min_hold
            ):
                self._fist_counted = True
                if self.fist_count == 0:
                    self.fist_count = 1
                    self._first_fist_at = now
                elif self.fist_count == 1:
                    self.fist_count = 2
        else:
            # Must open hand between the two closes
            self._fist_was = False
            self._fist_since = None
            self._fist_counted = False

    def _hit_test(self, x: float, y: float) -> MenuHit:
        if not self._layout:
            self._rebuild_layout()
        pad = 6
        for i, (mode, x0, y0, x1, y1) in enumerate(self._layout):
            if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
                return MenuHit(index=i, mode=mode, rect=(x0, y0, x1, y1))
        return MenuHit()

    @property
    def items(self) -> list[tuple[Mode, int, int, int, int]]:
        if not self._layout:
            self._rebuild_layout()
        return self._layout

    def reopen_progress(self, now: float) -> float:
        """0..1 progress toward reopening (double fist)."""
        if self.state != InteractionState.MENU_CLOSED:
            return 0.0
        if self.fist_count <= 0:
            return 0.0
        if self.fist_count >= 2:
            return 1.0
        # One fist registered: show half, plus hold progress on second if active
        base = 0.5
        if self._fist_was and self._fist_since is not None:
            held = now - self._fist_since
            base += 0.5 * min(1.0, held / max(self.config.menu_fist_min_hold, 0.01))
        return min(1.0, base)


class GestureController:
    def __init__(self, config: GestureConfig | None = None):
        self.config = config or GestureConfig()
        self._classifiers = [GestureClassifier(self.config), GestureClassifier(self.config)]
        self.gestures: dict[int, Gesture] = {}
        self.straightness: dict[int, float] = {}
        self.mode = Mode.ENERGY
        self._active = False
        self._since: dict[str, float | None] = {}
        self._cooldown_until = 0.0
        self._center = (0.0, 0.0)
        self._radius = 40.0
        self._pinch_scale = 1.0
        # Blast mode state
        self._blast = BlastState.IDLE
        self._charge = 0.0
        self._aim_origin = (0.0, 0.0)
        self._aim_dir = (0.0, -1.0)
        self._prev_lead = None  # previous lead-hand position for velocity
        self._charge_start = 0.0
        self.menu = ModeMenu(self.config)
        self.interaction = InteractionState.MENU_OPEN

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self._active = False
        self._since.clear()
        self._cooldown_until = 0.0
        self._blast = BlastState.IDLE
        self._charge = 0.0
        self._prev_lead = None

    def reset(self) -> None:
        self._active = False
        self._since.clear()
        self._cooldown_until = 0.0
        self._blast = BlastState.IDLE
        self._charge = 0.0
        self._prev_lead = None
        self.menu.force_open()
        self.interaction = InteractionState.MENU_OPEN

    def _held(self, key: str, condition: bool, now: float) -> float:
        if not condition:
            self._since[key] = None
            return 0.0
        start = self._since.get(key)
        if start is None:
            self._since[key] = now
            return 0.0
        return now - start

    def _end(self, now: float) -> None:
        self._active = False
        self._cooldown_until = now + self.config.cooldown
        self._since.clear()

    def _classify(self, hands: list[Hand]) -> None:
        present = {h.slot: h for h in hands if not h.stale}
        for slot, clf in enumerate(self._classifiers):
            hand = present.get(slot)
            if hand is None:
                clf.reset()
                self.gestures.pop(slot, None)
                self.straightness.pop(slot, None)
            else:
                s = hand_straightness(hand.pts)
                if math.isfinite(s):
                    self.straightness[slot] = s
                    pinched = is_pinch(hand.pts, self.config.pinch_ratio)
                    self.gestures[slot] = clf.update(s, pinched)

    def update(self, hands: list[Hand], now: float, max_radius: float, dt: float = 0.033) -> Control:
        cfg = self.config
        live = [h for h in hands if not h.stale]
        self._classify(live)

        # Menu owns gestures while open / animating selection
        selected = self.menu.update(live, self.gestures, now, dt)
        self.interaction = self.menu.state
        if selected is not None:
            self.set_mode(selected)

        menu_owns = self.menu.state in (InteractionState.MENU_OPEN, InteractionState.MODE_SELECTED)
        control = Control(mode=self.mode, hands_present=len(live))
        if menu_owns:
            # Explicitly idle control so effect gestures cannot fire from menu pinches
            control.active = False
            control.tracking = False
            control.intensity = 0.0
            control.blast_state = BlastState.IDLE
            control.charge = 0.0
            return control

        both = len(live) == 2 and all(h.slot in self.gestures for h in live)
        one = len(live) == 1 and live[0].slot in self.gestures
        dist_palms = math.inf
        palms = None
        intensity = 0.0
        wrist_a = (0.0, 0.0)
        tips: list[np.ndarray] = []

        if both:
            a, b = live[0], live[1]
            if a.slot > b.slot:
                a, b = b, a
            ca, cb = palm_center(a.pts), palm_center(b.pts)
            dist = float(np.linalg.norm(ca - cb))
            palm = 0.5 * (palm_size(a.pts) + palm_size(b.pts))
            dist_palms = dist / palm
            center = 0.5 * (ca + cb)
            radius = min(
                max(0.5 * dist * cfg.radius_scale, cfg.min_radius_palms * palm),
                max_radius,
            )
            if np.isfinite(center).all() and math.isfinite(radius):
                self._center = (float(center[0]), float(center[1]))
                self._radius = radius
                palms = ((float(ca[0]), float(ca[1])), (float(cb[0]), float(cb[1])))
                scores = [self.straightness.get(h.slot, 0.8) for h in (a, b)]
                claw_tight = max(0.0, min(1.0, (0.85 - sum(scores) / 2) / 0.35))
                prox = max(0.0, min(1.0, 1.0 - (dist_palms - 1.0) / max(cfg.summon_dist - 1.0, 0.1)))
                intensity = 0.30 + 0.70 * (0.55 * claw_tight + 0.45 * prox)
                wrist_a = (wrist_angle(a.pts), wrist_angle(b.pts))
                for h in (a, b):
                    for tid in _TIP_IDS:
                        tips.append(h.pts[tid, :2].copy())
                # Pinch scale (average)
                pinches = []
                for h in (a, b):
                    d = float(np.linalg.norm(h.pts[4, :2] - h.pts[8, :2]))
                    pinches.append(d / palm_size(h.pts))
                avg_pinch = sum(pinches) / 2
                target_scale = max(0.4, min(1.8, 0.5 + avg_pinch))
                self._pinch_scale += (target_scale - self._pinch_scale) * 0.15
            else:
                both = False

        elif one:
            h = live[0]
            c = palm_center(h.pts)
            self._center = (float(c[0]), float(c[1]))
            self._radius = max(palm_size(h.pts) * 2.5, 40.0)
            palms = ((float(c[0]), float(c[1])), (float(c[0]), float(c[1])))
            wrist_a = (wrist_angle(h.pts), wrist_angle(h.pts))
            for tid in _TIP_IDS:
                tips.append(h.pts[tid, :2].copy())
            intensity = 0.5
            control.single_hand = True

        states = [self.gestures[h.slot] for h in live] if both else []
        all_claw = both and all(g in (Gesture.CLAW, Gesture.PINCH) for g in states)
        all_fist = both and all(g is Gesture.FIST for g in states)
        any_fist = both and any(g is Gesture.FIST for g in states)
        all_open = both and all(g is Gesture.OPEN for g in states)

        # Mode-specific activation
        if self.mode in (Mode.ENERGY, Mode.PORTAL, Mode.EARTH, Mode.BEAM):
            if not self._active:
                ready = all_claw and dist_palms <= cfg.summon_dist and now >= self._cooldown_until
                if self.mode == Mode.BEAM:
                    ready = both and dist_palms <= cfg.release_dist and now >= self._cooldown_until
                if self._held("summon", ready, now) >= cfg.summon_hold:
                    self._active = True
                    self._since.clear()
                    control.summon = True
            else:
                if self._held("cancel", all_fist, now) >= cfg.cancel_hold:
                    control.cancel = True
                    self._end(now)
                else:
                    intact = True
                    if self.mode == Mode.BEAM:
                        intact = both
                    else:
                        intact = all_claw and dist_palms <= cfg.release_dist
                    grace = cfg.fist_release_hold if any_fist else cfg.release_hold
                    broken = self._held("broken", both and not intact, now) >= grace
                    lost = self._held("lost", not both, now) >= cfg.lost_hold
                    if broken or lost:
                        control.release = True
                        self._end(now)

        elif self.mode == Mode.BLAST:
            control.launch = False
            # --- Energy Blast state machine ---
            CHARGE_DIST = 3.8          # palm sizes to count as "hands together"
            MIN_CHARGE_TIME = 0.55     # seconds to reach full charge
            AIM_SEP = 2.2              # hands must separate this much to aim
            LAUNCH_SPEED = 380.0       # px/s forward velocity threshold
            COOLDOWN_T = 0.85

            if self._blast == BlastState.COOLDOWN:
                if now >= self._cooldown_until:
                    self._blast = BlastState.IDLE
                    self._charge = 0.0
                    self._prev_lead = None

            elif self._blast in (BlastState.IDLE, BlastState.CHARGING, BlastState.READY):
                charging_pose = both and dist_palms <= CHARGE_DIST
                if charging_pose and now >= self._cooldown_until:
                    if self._blast == BlastState.IDLE:
                        self._blast = BlastState.CHARGING
                        self._charge_start = now
                        self._charge = 0.0
                    # ramp charge
                    held = now - self._charge_start
                    self._charge = min(1.0, held / MIN_CHARGE_TIME)
                    if self._charge >= 1.0:
                        self._blast = BlastState.READY
                    self._active = True
                    # keep origin at midpoint while charging
                    if palms:
                        self._aim_origin = self._center
                else:
                    if self._blast == BlastState.CHARGING and self._charge < 0.35:
                        # cancelled early
                        self._blast = BlastState.IDLE
                        self._charge = 0.0
                        self._active = False
                    elif self._blast == BlastState.READY and both and dist_palms > AIM_SEP:
                        self._blast = BlastState.AIMING
                    elif self._blast == BlastState.CHARGING and not both:
                        self._blast = BlastState.IDLE
                        self._charge = 0.0
                        self._active = False

            elif self._blast == BlastState.AIMING:
                self._active = True
                if not both and not one:
                    # lost hands → cancel back to ready if still charged
                    self._blast = BlastState.READY if self._charge >= 1.0 else BlastState.IDLE
                else:
                    # Lead hand = the one farther from image center-bottom (more "forward")
                    # Use the hand with smaller y (higher in frame) as forward aim hand
                    if both and palms:
                        (p0, p1) = palms
                        # forward = higher in frame (smaller y) or farther from body mid
                        if p0[1] < p1[1]:
                            lead, rear = p0, p1
                        else:
                            lead, rear = p1, p0
                        origin = rear
                        dx = lead[0] - rear[0]
                        dy = lead[1] - rear[1]
                        # prefer upward bias if hands are level
                        if abs(dy) < 25 and abs(dx) < 40:
                            dy = -80
                        norm = math.hypot(dx, dy) + 1e-6
                        self._aim_origin = origin
                        self._aim_dir = (dx / norm, dy / norm)
                        self._center = lead
                        # launch detection: lead hand moves quickly in aim direction
                        if self._prev_lead is not None:
                            vx = (lead[0] - self._prev_lead[0]) / max(now - getattr(self, "_prev_t", now), 1e-3)
                            vy = (lead[1] - self._prev_lead[1]) / max(now - getattr(self, "_prev_t", now), 1e-3)
                            speed_along = vx * self._aim_dir[0] + vy * self._aim_dir[1]
                            # also allow open-palm thrust of lead hand
                            lead_slot = 0 if (palms[0][1] < palms[1][1]) else 1
                            lead_open = self.gestures.get(lead_slot) is Gesture.OPEN
                            if speed_along > LAUNCH_SPEED or (lead_open and speed_along > 120):
                                control.launch = True
                                self._blast = BlastState.FIRING
                                self._cooldown_until = now + COOLDOWN_T
                        self._prev_lead = lead
                        self._prev_t = now
                    elif one and palms:
                        # single hand aim from palm forward (upward default)
                        p = palms[0]
                        self._aim_origin = p
                        self._aim_dir = (0.0, -1.0)
                        self._center = p
                        if self._prev_lead is not None:
                            vy = (p[1] - self._prev_lead[1]) / max(now - getattr(self, "_prev_t", now), 1e-3)
                            if vy < -LAUNCH_SPEED * 0.6:  # fast upward
                                control.launch = True
                                self._blast = BlastState.FIRING
                                self._cooldown_until = now + COOLDOWN_T
                        self._prev_lead = p
                        self._prev_t = now

            elif self._blast == BlastState.FIRING:
                # brief state; effects owns the projectile after launch
                if now >= self._cooldown_until - COOLDOWN_T + 0.15:
                    self._blast = BlastState.COOLDOWN
                    self._active = False
                    self._charge = 0.0
                    self._prev_lead = None

            control.blast_state = self._blast
            control.charge = self._charge
            control.aim_origin = self._aim_origin
            control.aim_dir = self._aim_dir
            intensity = self._charge

        control.active = self._active
        control.tracking = both or (one and self.mode == Mode.BLAST)
        control.center = self._center
        control.radius = self._radius
        control.palms = palms
        control.fingertips = tips
        control.wrist_angles = wrist_a
        control.intensity = intensity if (control.active or self.mode == Mode.BLAST) else 0.0
        control.pinch_scale = self._pinch_scale
        if self.mode == Mode.BLAST:
            control.blast_state = self._blast
            control.charge = self._charge
            control.aim_origin = self._aim_origin
            control.aim_dir = self._aim_dir
        return control
