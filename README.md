# HoloMotion AI

Cinematic hand-gesture visual effects for webcam. Five sci-fi modes controlled by real-time hand tracking — fully hands-free mode selection via holographic menu.

**Stack:** Python 3.11 · OpenCV · MediaPipe · NumPy only. Runs entirely locally in one window.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

On first run the MediaPipe hand landmarker model (~8 MB) is downloaded to `models/`.

Options: `--camera 0` `--width 1280` `--height 720` `--model PATH` `--no-skeleton`

The preview window is resizable; the camera feed scales to fill the window.

## Hands-free mode selection

On launch a **holographic mode selector** appears over the webcam feed:

1. Point with your **index fingertip** — a crosshair cursor tracks it and highlights the option under the tip.
2. **Pinch** (index + thumb together) and hold briefly (~0.18 s) to confirm. A progress ring shows the debounce.
3. The menu closes, the mode name is shown, and the effect activates.

To **reopen the menu**: **close your hand into a fist twice** (close → open → close). HUD shows `MENU 1/2` then opens the selector. After you pick a mode the menu stays closed so that mode can run; it will not keep re-selecting.

**Q / Esc** — quit · **R** — reset (emergency keyboard only; modes are gesture-selected)

## Modes

| Mode | How to use |
|------|------------|
| **Energy Core** | Claw both hands together → energy orb. Move to resize. Open claw → explosion. Fists → cancel. |
| **Portal** | Claw together → rotating vortex portal. Expand/shrink. Release → shockwave. |
| **Holo Earth** | Claw together → wireframe globe. Wrist rotation spins it. Pinch to zoom. |
| **Energy Beam** | Hold both hands → continuous luminous beam with sparks & branches. |
| **Energy Blast** | Superhero charge → aim → launch projectile (see below). |

---

## Mode 5 — Energy Blast (charge / aim / launch)

Cinematic projectile system:

1. **Charge** — Bring both hands together in front of you. Hold ~0.5 s.  
   Watch the core grow, electrical arcs jump between palms, charge bar fills (CHG).

2. **Ready** — When charge hits 100%, status shows READY.  
   Separate your hands to enter aiming.

3. **Aim** — One hand leads (higher / forward). A thin cyan aim line shows direction.  
   Move the lead hand to adjust aim.

4. **Launch** — Thrust the lead hand forward quickly (or open-palm push).  
   A bright energy projectile flies with a luminous trail, electrical branches, and sparks.

5. **Impact** — When the projectile hits the edge of the frame (or expires),  
   you get a flash, expanding rings, particle explosion, and a brief screen shake.

6. **Cooldown** — Short cooldown, then you can charge again.

States shown in the HUD: `Charging…` → `READY` → `Aim` → `Firing!` → `Cooldown…`

---

## Project layout

```
holomotion-ai/
├── main.py           # webcam loop, HUD, holographic menu render
├── hand_tracker.py   # MediaPipe landmarks, stable slots, smoothing
├── gestures.py       # classification, state machines, ModeMenu, Control
├── effects.py        # all five visual modes + particle pool + glow
├── requirements.txt
├── README.md
└── .gitignore
```

## Interaction states

- `MENU_OPEN` — menu owns index/pinch; effects receive idle control  
- `MODE_SELECTED` — short close animation + mode banner  
- `MENU_CLOSED` — effect gestures active; double closed-hand reopens menu  

