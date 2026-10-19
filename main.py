from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from effects import Effects
from gestures import (
    Control,
    GestureController,
    InteractionState,
    Mode,
    MODE_NAMES,
    palm_center,
)
from hand_tracker import Hand, HandTracker, TrackerError, draw_skeleton

WINDOW = "HoloMotion AI"
HUD_COLOR = (180, 185, 190)
TITLE_COLOR = (255, 235, 80)
ACCENT = (255, 220, 60)
WARN_COLOR = (60, 180, 255)
LABEL_COLOR = (230, 220, 90)
MENU_CYAN = (255, 220, 80)
MENU_GLOW = (255, 180, 40)
MENU_DIM = (90, 70, 40)
MENU_HL = (255, 240, 120)
LOW_LIGHT_LEVEL = 45.0
MAX_DETECT_ERRORS = 30
MAX_READ_FAILURES = 30


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open camera {index}. Check connection and permissions. "
            "Try --camera 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(
            "Camera opened but returned no frames. Try --width 640 --height 480."
        )
    return cap


def put_text(frame, text, org, color=HUD_COLOR, scale=0.5, thickness=1):
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def status_text(hands: list[Hand], ctrl: GestureController, control: Control) -> str:
    if ctrl.interaction == InteractionState.MENU_OPEN:
        return "Point with index · pinch to select mode"
    if ctrl.interaction == InteractionState.MODE_SELECTED:
        name = MODE_NAMES.get(ctrl.mode, "")
        return f"Selected  ·  {name}"
    mode = ctrl.mode
    if mode == Mode.BLAST:
        from gestures import BlastState

        st = control.blast_state
        if st == BlastState.CHARGING:
            return f"Charging… {int(control.charge * 100)}%  ·  keep hands together"
        if st == BlastState.READY:
            return "READY  ·  separate hands to aim"
        if st == BlastState.AIMING:
            return "Aim  ·  thrust lead hand forward to LAUNCH"
        if st == BlastState.FIRING:
            return "Firing!"
        if st == BlastState.COOLDOWN:
            return "Cooldown…"
        return "Bring both hands together to charge energy"
    if control.active:
        if mode == Mode.BEAM:
            return "Move hands  ·  fists to cancel  ·  release to discharge"
        return "Move to resize  ·  open claw to burst  ·  fists to cancel"
    if control.hands_present == 0:
        return "Show both hands  ·  close hand twice to open menu"
    if control.hands_present == 1:
        return "Show your other hand  ·  close hand twice to open menu"
    if mode == Mode.BEAM:
        return "Hold both hands in view to form the beam"
    return "Claw both hands and bring them together"


def draw_hud(frame, fps, hands, ctrl, control, low_light):
    h, w = frame.shape[:2]
    put_text(frame, "HoloMotion AI", (12, 24), TITLE_COLOR, 0.62)
    put_text(frame, MODE_NAMES[ctrl.mode], (12, 46), ACCENT, 0.48)
    put_text(frame, f"{fps:.0f} FPS", (w - 78, 22))

    hint = "Close hand ×2 · menu   Q quit"
    if ctrl.interaction == InteractionState.MENU_OPEN:
        hint = "Index tip · cursor   Pinch · select   Q quit"
    put_text(frame, hint, (12, h - 14), HUD_COLOR, 0.40)
    put_text(frame, status_text(hands, ctrl, control), (12, h - 34), HUD_COLOR, 0.42)

    if low_light:
        put_text(frame, "Low light", (12, 68), WARN_COLOR, 0.42)

    if ctrl.interaction == InteractionState.MENU_CLOSED:
        prog = ctrl.menu.reopen_progress(time.perf_counter())
        fc = ctrl.menu.fist_count
        if prog > 0.05 or fc > 0:
            bar_w, bar_h = 100, 5
            x0, y0 = 12, 72 if not low_light else 88
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (35, 35, 45), -1)
            fill = int(bar_w * max(prog, 0.5 * fc))
            cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bar_h), MENU_CYAN, -1)
            label = f"MENU {fc}/2" if fc > 0 else "MENU"
            put_text(frame, label, (x0 + bar_w + 8, y0 + 7), ACCENT, 0.38)

    if control.mode == Mode.BLAST and control.charge > 0.02 and ctrl.interaction == InteractionState.MENU_CLOSED:
        bar_w, bar_h = 110, 5
        x0, y0 = w - bar_w - 12, 36
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (35, 35, 45), -1)
        fill = int(bar_w * min(1.0, control.charge))
        col = (40, 220, 255) if control.charge < 1.0 else (40, 255, 180)
        cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bar_h), col, -1)
        put_text(frame, "CHG", (x0 - 36, y0 + 7), ACCENT, 0.38)
    elif control.active and control.intensity > 0.05 and ctrl.interaction == InteractionState.MENU_CLOSED:
        bar_w, bar_h = 110, 5
        x0, y0 = w - bar_w - 12, 36
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (35, 35, 45), -1)
        fill = int(bar_w * min(1.0, control.intensity))
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + fill, y0 + bar_h),
            (255, int(170 + 60 * control.intensity), 40),
            -1,
        )
        put_text(frame, "PWR", (x0 - 36, y0 + 7), ACCENT, 0.38)

    if ctrl.interaction == InteractionState.MENU_CLOSED:
        for hand in hands:
            g = ctrl.gestures.get(hand.slot)
            if g is None:
                continue
            x, y = hand.pts[0, :2]
            score = ctrl.straightness.get(hand.slot, 0.0)
            put_text(frame, f"{g.name} {score:.2f}", (int(x) - 36, int(y) + 26), LABEL_COLOR, 0.42)


def draw_mode_menu(frame: np.ndarray, ctrl: GestureController, now: float) -> None:
    menu = ctrl.menu
    if menu.anim < 0.02 and menu.state == InteractionState.MENU_CLOSED:
        return

    h, w = frame.shape[:2]
    menu.set_frame_size(w, h)
    a = menu.anim
    if a < 0.01:
        return

    # Soft holographic vignette panel
    overlay = frame.copy()
    cx, cy = w // 2, h // 2
    panel_w = max(180, int(w * 0.28))
    panel_h = max(220, int(h * 0.52))
    px0, py0 = cx - panel_w // 2, cy - panel_h // 2 - int(h * 0.04)
    px1, py1 = px0 + panel_w, py0 + panel_h
    cv2.rectangle(overlay, (px0, py0), (px1, py1), (20, 18, 12), -1)
    cv2.addWeighted(overlay, 0.35 * a, frame, 1.0 - 0.35 * a, 0, frame)

    # Title
    title = "SELECT MODE"
    ts = 0.55
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, ts, 1)
    put_text(
        frame,
        title,
        (cx - tw // 2, py0 + 28),
        sc_color(MENU_CYAN, a),
        ts,
        1,
    )
    # Decorative scan lines
    for i in range(3):
        y = py0 + 36 + i * 3
        cv2.line(frame, (px0 + 20, y), (px1 - 20, y), sc_color(MENU_DIM, a * 0.6), 1)

    items = menu.items
    hover_idx = menu.hover.index
    for i, (mode, x0, y0, x1, y1) in enumerate(items):
        # Slide-in from center
        mid = (y0 + y1) // 2
        offset = int((1.0 - a) * 40 * (1 if i % 2 == 0 else -1))
        rx0, rx1 = x0 + offset, x1 + offset
        highlighted = i == hover_idx and menu.state == InteractionState.MENU_OPEN
        selected = menu.selected is mode and menu.state == InteractionState.MODE_SELECTED

        # Box
        fill = (35, 30, 18) if not highlighted else (55, 48, 22)
        if selected:
            fill = (40, 70, 30)
        border = MENU_HL if highlighted or selected else MENU_CYAN
        alpha_box = a * (1.0 if highlighted or selected else 0.75)
        cv2.rectangle(frame, (rx0, y0), (rx1, y1), sc_color(fill, alpha_box), -1)
        cv2.rectangle(frame, (rx0, y0), (rx1, y1), sc_color(border, a), 1 + int(highlighted or selected))

        # Corner ticks
        tick = 8
        col = sc_color(border, a)
        for (tx, ty, dx, dy) in (
            (rx0, y0, 1, 1),
            (rx1, y0, -1, 1),
            (rx0, y1, 1, -1),
            (rx1, y1, -1, -1),
        ):
            cv2.line(frame, (tx, ty), (tx + dx * tick, ty), col, 1, cv2.LINE_AA)
            cv2.line(frame, (tx, ty), (tx, ty + dy * tick), col, 1, cv2.LINE_AA)

        name = MODE_NAMES[mode]
        scale = 0.48 if not highlighted else 0.52
        (nw, nh), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx = (rx0 + rx1 - nw) // 2
        ty = mid + nh // 2
        put_text(frame, name, (tx, ty), sc_color(MENU_HL if highlighted else MENU_CYAN, a), scale)

        # Index number
        put_text(
            frame,
            str(i + 1),
            (rx0 + 8, mid + 5),
            sc_color(MENU_DIM, a),
            0.40,
        )

    # Cursor from index fingertip
    if menu.cursor is not None and menu.state == InteractionState.MENU_OPEN:
        cx_i, cy_i = int(menu.cursor[0]), int(menu.cursor[1])
        r = 10
        cv2.circle(frame, (cx_i, cy_i), r, MENU_CYAN, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx_i, cy_i), 3, MENU_HL, -1, cv2.LINE_AA)
        # Crosshair
        cv2.line(frame, (cx_i - 16, cy_i), (cx_i - 6, cy_i), MENU_CYAN, 1, cv2.LINE_AA)
        cv2.line(frame, (cx_i + 6, cy_i), (cx_i + 16, cy_i), MENU_CYAN, 1, cv2.LINE_AA)
        cv2.line(frame, (cx_i, cy_i - 16), (cx_i, cy_i - 6), MENU_CYAN, 1, cv2.LINE_AA)
        cv2.line(frame, (cx_i, cy_i + 6), (cx_i, cy_i + 16), MENU_CYAN, 1, cv2.LINE_AA)

        # Pinch progress ring
        if menu.pinch_progress > 0.05:
            angle = int(360 * menu.pinch_progress)
            cv2.ellipse(
                frame,
                (cx_i, cy_i),
                (18, 18),
                0,
                -90,
                -90 + angle,
                MENU_HL,
                2,
                cv2.LINE_AA,
            )

    # Post-select banner
    if menu.state == InteractionState.MODE_SELECTED and menu.selected is not None:
        name = MODE_NAMES[menu.selected]
        msg = f"▶  {name}"
        scale = 0.72
        (nw, nh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        bx0 = (w - nw) // 2 - 16
        by0 = h // 2 - 28
        cv2.rectangle(frame, (bx0, by0), (bx0 + nw + 32, by0 + nh + 24), (20, 40, 25), -1)
        cv2.rectangle(frame, (bx0, by0), (bx0 + nw + 32, by0 + nh + 24), (80, 255, 120), 1)
        put_text(frame, msg, (bx0 + 16, by0 + nh + 8), (120, 255, 180), scale, 2)


def sc_color(color, k: float):
    k = max(0.0, min(1.0, k))
    return (int(color[0] * k), int(color[1] * k), int(color[2] * k))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HoloMotion AI — cinematic hand-gesture effects")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", default=None)
    p.add_argument("--no-skeleton", action="store_true")
    return p.parse_args()


def window_closed(n: int) -> bool:
    if n < 3:
        return False
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def get_window_size(default_w: int, default_h: int) -> tuple[int, int]:
    try:
        # OpenCV does not expose client size reliably on all backends; use rect when available
        x, y, ww, hh = cv2.getWindowImageRect(WINDOW)
        if ww > 10 and hh > 10:
            return int(ww), int(hh)
    except Exception:
        pass
    return default_w, default_h


def run(args: argparse.Namespace) -> None:
    cap = open_camera(args.camera, args.width, args.height)
    try:
        tracker = HandTracker(args.model)
    except TrackerError:
        cap.release()
        raise

    controller = GestureController()
    controller.menu.force_open()
    effects: Effects | None = None
    read_failures = detect_errors = frame_count = 0
    fps, low_light = 0.0, False
    prev = time.perf_counter()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, args.height)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                read_failures += 1
                if read_failures > MAX_READ_FAILURES:
                    raise RuntimeError("Camera stopped delivering frames.")
                if cv2.waitKey(10) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue
            read_failures = 0
            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Stretch capture to fill current window size
            win_w, win_h = get_window_size(args.width, args.height)
            if frame.shape[1] != win_w or frame.shape[0] != win_h:
                frame = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)

            now = time.perf_counter()
            dt = min(max(now - prev, 1e-4), 0.1)
            fps += (1.0 / dt - fps) * 0.1
            prev = now

            if effects is None or effects.layer.shape[:2] != frame.shape[:2]:
                effects = Effects(frame.shape[1], frame.shape[0])
                effects.set_mode(controller.mode)

            if frame_count % 10 == 1:
                low_light = float(frame[::8, ::8].mean()) < LOW_LIGHT_LEVEL

            try:
                hands = tracker.detect(frame, now)
                detect_errors = 0
            except Exception as exc:
                detect_errors += 1
                if detect_errors > MAX_DETECT_ERRORS:
                    raise RuntimeError(f"Hand tracking keeps failing: {exc}") from exc
                hands = []

            control = controller.update(hands, now, 0.48 * min(frame.shape[:2]), dt)

            # Sync effects mode after gesture selection
            if effects.mode != controller.mode:
                effects.set_mode(controller.mode)

            # Only drive effects when menu does not own gestures
            if controller.interaction == InteractionState.MENU_CLOSED:
                effects.update(dt, control)
                effects.update_trails(
                    {h.slot: palm_center(h.pts) for h in hands if not h.stale}
                )
            else:
                # Keep particle decay / idle ambient without activation signals
                idle = Control(mode=controller.mode, hands_present=0)
                effects.update(dt, idle)

            if not args.no_skeleton:
                for hand in hands:
                    draw_skeleton(frame, hand)
            effects.render(frame)
            draw_mode_menu(frame, controller, now)
            draw_hud(frame, fps, hands, controller, control, low_light)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or window_closed(frame_count):
                break
            # Emergency keyboard only: reset still useful, no mode keys
            if key in (ord("r"), ord("R")):
                controller.reset()
                effects.reset()
                effects.set_mode(controller.mode)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> int:
    try:
        run(parse_args())
    except (RuntimeError, TrackerError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
