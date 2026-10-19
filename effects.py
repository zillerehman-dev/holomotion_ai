from __future__ import annotations

import math
from enum import IntEnum

import cv2
import numpy as np

from gestures import BlastState, Control, Mode, MODE_NAMES

# ── Colour palette (BGR) ──────────────────────────────────────────────
CYAN = (255, 235, 40)
BLUE = (255, 130, 20)
PURPLE = (230, 60, 180)
WHITE = (255, 255, 240)
GOLD = (40, 200, 255)
MAGENTA = (180, 40, 255)
DEEP_BLUE = (180, 60, 10)

SHIFT = 4
ONE = 1 << SHIFT
TAU = 2.0 * math.pi
MAX_PARTICLES = 1100


def sc(color, k: float):
    k = max(k, 0.0)
    return (min(255, int(color[0] * k)), min(255, int(color[1] * k)), min(255, int(color[2] * k)))


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def _fp(v: float) -> int:
    return int(v * ONE + 0.5)


# ── Shared particle pool ──────────────────────────────────────────────
class ParticlePool:
    def __init__(self, n: int = MAX_PARTICLES, seed: int | None = None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((n, 2), np.float32)
        self.vel = np.zeros((n, 2), np.float32)
        self.life = np.zeros(n, np.float32)
        self.maxn = np.ones(n, np.float32)
        self.mix = np.zeros(n, np.float32)   # colour mix 0=cyan 1=purple
        self.size = np.ones(n, np.float32)
        self.alive = 0

    def emit(self, center, count, speed_scale=1.0, life_range=(0.6, 1.6),
             radial=True, mix_prob=0.4, size_range=(1.0, 2.5)):
        free = np.flatnonzero(self.life <= 0)[:count]
        n = free.size
        if n == 0:
            return
        rng = self.rng
        ang = rng.uniform(0, TAU, n)
        rad = np.stack((np.cos(ang), np.sin(ang)), axis=1)
        if radial:
            spd = rng.uniform(0.5, 2.5, n) * speed_scale
            tan = np.stack((-rad[:, 1], rad[:, 0]), axis=1)
            swirl = rng.normal(0, 0.3, n) * spd
            self.vel[free] = rad * spd[:, None] + tan * swirl[:, None]
            ring = rng.uniform(0.1, 1.0, n)
            self.pos[free] = center + rad * (ring * speed_scale * 0.15)[:, None]
        else:
            self.pos[free] = center + rng.normal(0, 8, (n, 2)).astype(np.float32)
            self.vel[free] = rng.normal(0, speed_scale * 0.4, (n, 2)).astype(np.float32)
        life = rng.uniform(*life_range, n)
        self.life[free] = life
        self.maxn[free] = life
        self.mix[free] = (rng.random(n) < mix_prob).astype(np.float32) * rng.uniform(0.5, 1.0, n)
        self.size[free] = rng.uniform(*size_range, n)

    def step(self, dt: float, drag: float = 1.6):
        self.life -= dt
        np.maximum(self.life, -1.0, out=self.life)
        self.alive = int(np.count_nonzero(self.life > 0))
        if self.alive:
            self.vel *= math.exp(-drag * dt)
            self.pos += self.vel * dt

    def draw(self, layer: np.ndarray, gold_thresh: float = 0.8):
        idx = np.flatnonzero(self.life > 0)
        if idx.size == 0:
            return
        fade = np.clip(self.life[idx] / self.maxn[idx], 0, 1) ** 0.55
        mix = self.mix[idx][:, None]
        base = np.array(CYAN, np.float32) * (1 - mix) + np.array(PURPLE, np.float32) * mix
        gold = (mix > gold_thresh).astype(np.float32)
        base = base * (1 - gold) + np.array(GOLD, np.float32) * gold
        pos, vel = self.pos[idx], self.vel[idx]
        h, w = layer.shape[:2]
        for tail, wt in ((0.028, 0.25), (0.012, 0.55), (0.0, 1.0)):
            xy = pos - vel * tail
            colors = np.minimum(base * (fade * wt)[:, None], 255).astype(np.uint8)
            ix = np.rint(xy[:, 0]).astype(np.int32)
            iy = np.rint(xy[:, 1]).astype(np.int32)
            ok = (ix >= 0) & (ix < w - 1) & (iy >= 0) & (iy < h - 1)
            if not ok.any():
                continue
            ix, iy, c = ix[ok], iy[ok], colors[ok]
            layer[iy, ix] = c
            layer[iy, ix + 1] = c
            layer[iy + 1, ix] = c
            layer[iy + 1, ix + 1] = c


# ── Main effects engine ───────────────────────────────────────────────
class Effects:
    def __init__(self, width: int, height: int, seed: int | None = None):
        self.w, self.h = width, height
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.mode = Mode.ENERGY
        self.active = False
        self.appear = 0.0
        self.dissolve = 0.0
        self.center = np.zeros(2, np.float32)
        self.radius = 80.0
        self.presence = 0.0
        self.intensity = 1.0
        self.palms = None
        self.fingertips: list[np.ndarray] = []
        self.wrist_angles = (0.0, 0.0)
        self.pinch_scale = 1.0
        self.flash = 0.0
        self.shock = -1.0
        self.blast_c = np.zeros(2, np.float32)
        self.blast_r = 0.0

        self.pool = ParticlePool(MAX_PARTICLES, seed)
        # Ambient orbiters (shared look)
        n_orb = 40
        self.orb_r = self.rng.uniform(0.2, 1.3, n_orb)
        self.orb_a = self.rng.uniform(0, TAU, n_orb)
        self.orb_w = self.rng.choice((-1.0, 1.0), n_orb) * self.rng.uniform(0.3, 1.5, n_orb)
        self.orb_ph = self.rng.uniform(0, TAU, n_orb)

        # Portal spiral seeds
        n_sp = 120
        self.sp_a = self.rng.uniform(0, TAU, n_sp)
        self.sp_r = self.rng.uniform(0.05, 1.0, n_sp)
        self.sp_w = self.rng.uniform(0.8, 2.5, n_sp) * self.rng.choice((-1, 1), n_sp)
        self.sp_z = self.rng.uniform(0.3, 1.0, n_sp)

        # Earth satellites
        self.sat_a = self.rng.uniform(0, TAU, 8)
        self.sat_w = self.rng.uniform(0.4, 1.2, 8)
        self.sat_inc = self.rng.uniform(0.2, 0.8, 8)

        # Energy Blast projectile
        self.proj_alive = False
        self.proj_pos = np.zeros(2, np.float32)
        self.proj_vel = np.zeros(2, np.float32)
        self.proj_life = 0.0
        self.proj_trail = np.zeros((24, 2), np.float32)
        self.proj_trail_i = 0
        self.proj_charge = 0.0
        self.impact_t = -1.0
        self.impact_pos = np.zeros(2, np.float32)
        self.shake = 0.0
        self.blast_state = BlastState.IDLE
        self.charge_level = 0.0
        self.aim_origin = np.zeros(2, np.float32)
        self.aim_dir = np.array([0.0, -1.0], np.float32)

        # Trails
        self.trail_len = 20
        self.trail_xy = np.zeros((2, self.trail_len, 2), np.float32)
        self.trail_t = np.full((2, self.trail_len), -1e9)
        self.trail_i = [0, 0]

        self._alloc()

    def _alloc(self):
        w, h = self.w, self.h
        self.layer = np.zeros((h, w, 3), np.uint8)
        self._glow = np.zeros((h, w, 3), np.uint8)
        self._w4, self._h4 = max(w // 4, 1), max(h // 4, 1)
        self._w8, self._h8 = max(w // 8, 1), max(h // 8, 1)
        self._s4 = np.zeros((self._h4, self._w4, 3), np.uint8)
        self._b4 = np.zeros_like(self._s4)
        self._s8 = np.zeros((self._h8, self._w8, 3), np.uint8)
        self._b8 = np.zeros_like(self._s8)
        ys, xs = np.mgrid[0:h, 0:w]
        r = np.hypot((xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2)) / math.sqrt(2)
        shade = (0.84 * (1.0 - 0.42 * r ** 2))[..., None] * np.array((1.0, 0.93, 0.80))
        self._mask = np.clip(shade * 255, 0, 255).astype(np.uint8)
        self._white = np.full((h, w, 3), 255, np.uint8)
        self._tmp = np.zeros((h, w, 3), np.uint8)

    def set_mode(self, mode: Mode):
        self.mode = mode
        self.active = False
        self.appear = 0.0
        self.dissolve = 0.0
        self.flash = 0.0
        self.shock = -1.0
        self.pool.life[:] = 0
        self.pool.alive = 0
        self.proj_alive = False
        self.impact_t = -1.0
        self.charge_level = 0.0
        self.shake = 0.0

    def reset(self):
        self.active = False
        self.appear = 0.0
        self.dissolve = 0.0
        self.flash = 0.0
        self.shock = -1.0
        self.pool.life[:] = 0
        self.pool.alive = 0
        self.presence = 0.0
        self.proj_alive = False
        self.impact_t = -1.0
        self.charge_level = 0.0
        self.shake = 0.0

    @property
    def busy(self) -> bool:
        return (
            self.active or self.dissolve > 0 or self.flash > 0
            or self.shock >= 0 or self.pool.alive > 0
            or self.proj_alive or self.impact_t >= 0
            or (self.mode == Mode.BLAST and self.charge_level > 0.05)
        )

    def update(self, dt: float, control: Control):
        dt = min(max(dt, 0.0), 0.1)
        self.t += dt
        finite = math.isfinite(control.radius) and all(math.isfinite(v) for v in control.center)

        if control.mode != self.mode:
            self.set_mode(control.mode)

        if control.summon and finite:
            self.active = True
            self.appear = 0.0
            self.dissolve = 0.0
            self.center[:] = control.center
            self.radius = control.radius

        if self.active:
            if control.tracking and finite:
                a = 1.0 - math.exp(-dt / 0.06)
                self.center += a * (np.asarray(control.center, np.float32) - self.center)
                self.radius += (1.0 - math.exp(-dt / 0.08)) * (control.radius - self.radius)
                self.intensity += (control.intensity - self.intensity) * (1 - math.exp(-dt / 0.12))
            self.appear = min(1.0, self.appear + dt / 0.50)
            self.palms = control.palms
            self.fingertips = control.fingertips
            self.wrist_angles = control.wrist_angles
            self.pinch_scale = control.pinch_scale

            if control.release:
                self._explode()
                self.active = False
            elif control.cancel:
                self.active = False
                self.dissolve = 1.0
        elif self.dissolve > 0:
            self.dissolve = max(0.0, self.dissolve - dt / 0.38)
            self.palms = None

        # Energy Blast mode
        if self.mode == Mode.BLAST:
            self._update_blast(dt, control)

        self.pool.step(dt)
        if self.flash > 0:
            self.flash = max(0.0, self.flash - dt / 0.30)
        if self.shock >= 0:
            self.shock += dt / 0.75
            if self.shock > 1.0:
                self.shock = -1.0

        if self.active or self.proj_alive or self.impact_t >= 0 or (
            self.mode == Mode.BLAST and self.charge_level > 0.05
        ):
            target = 1.0
        else:
            tail = float(np.clip(self.pool.life.max(), 0, 1)) if self.pool.alive else 0.0
            target = min(1.0, self.dissolve + self.flash + tail)
        self.presence += (target - self.presence) * (1 - math.exp(-dt / 0.14))
        if not self.busy and self.presence < 0.02:
            self.presence = 0.0

    def update_trails(self, points: dict[int, np.ndarray]):
        for slot, xy in points.items():
            if slot not in (0, 1) or not np.isfinite(xy).all():
                continue
            i = self.trail_i[slot]
            last = (i - 1) % self.trail_len
            moved = float(np.hypot(*(xy - self.trail_xy[slot, last])))
            if self.trail_t[slot, last] < 0 or moved > 1.5:
                self.trail_xy[slot, i] = xy
                self.trail_t[slot, i] = self.t
                self.trail_i[slot] = (i + 1) % self.trail_len

    def render(self, frame: np.ndarray):
        trails = self.t - float(self.trail_t.max()) < 0.55
        if not (self.busy or trails):
            return
        self.layer.fill(0)
        if self.busy:
            self._darken(frame)
            self._draw_blast()
            if self.active or self.dissolve > 0 or self.mode == Mode.BLAST:
                if self.mode == Mode.ENERGY:
                    self._draw_energy()
                elif self.mode == Mode.PORTAL:
                    self._draw_portal()
                elif self.mode == Mode.EARTH:
                    self._draw_earth()
                elif self.mode == Mode.BEAM:
                    self._draw_beam()
                elif self.mode == Mode.BLAST:
                    self._draw_blast_mode()
            self.pool.draw(self.layer)
        self._draw_trails()
        self._composite(frame)
        # Screen shake on impact
        if self.shake > 0:
            dx = int(self.rng.normal(0, 6 * self.shake))
            dy = int(self.rng.normal(0, 6 * self.shake))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            cv2.warpAffine(frame, M, (self.w, self.h), dst=frame,
                           borderMode=cv2.BORDER_REPLICATE)

    # ── helpers ───────────────────────────────────────────────────────
    def _circle(self, cx, cy, r, color, th=1):
        if r < 0.8:
            return
        cv2.circle(self.layer, (_fp(cx), _fp(cy)), _fp(r), color, th, cv2.LINE_AA, SHIFT)

    def _line(self, x0, y0, x1, y1, color, th=1):
        cv2.line(self.layer, (_fp(x0), _fp(y0)), (_fp(x1), _fp(y1)), color, th, cv2.LINE_AA, SHIFT)

    def _arcs(self, cx, cy, r, count, span, rot, color, th):
        if r < 1:
            return
        axes = (_fp(r), _fp(r))
        for i in range(count):
            start = rot + i * 360.0 / count
            cv2.ellipse(self.layer, (_fp(cx), _fp(cy)), axes, 0, start, start + span,
                        color, th, cv2.LINE_AA, SHIFT)

    def _poly(self, cx, cy, r, n, rot, color, th=1):
        if r < 2:
            return
        ang = rot + np.arange(n) * (TAU / n)
        pts = np.empty((n, 2), np.int32)
        pts[:, 0] = (cx + r * np.cos(ang)) * ONE
        pts[:, 1] = (cy + r * np.sin(ang)) * ONE
        cv2.polylines(self.layer, [pts], True, color, th, cv2.LINE_AA, SHIFT)

    def _splat(self, xy, colors):
        h, w = self.layer.shape[:2]
        ix = np.rint(xy[:, 0]).astype(np.int32)
        iy = np.rint(xy[:, 1]).astype(np.int32)
        ok = (ix >= 0) & (ix < w - 1) & (iy >= 0) & (iy < h - 1)
        if not ok.any():
            return
        ix, iy, c = ix[ok], iy[ok], colors[ok]
        self.layer[iy, ix] = c
        self.layer[iy, ix + 1] = c
        self.layer[iy + 1, ix] = c
        self.layer[iy + 1, ix + 1] = c

    def _explode(self):
        self.flash = 1.0
        self.shock = 0.0
        self.blast_c = self.center.copy()
        self.blast_r = max(self.radius, 28.0)
        self.pool.emit(
            self.center, 380,
            speed_scale=self.blast_r * 1.4 + 100,
            life_range=(0.7, 1.9),
            mix_prob=0.45,
        )

    def _draw_blast(self):
        cx, cy = float(self.blast_c[0]), float(self.blast_c[1])
        R = self.blast_r
        if self.flash > 0:
            f = self.flash
            grow = 0.55 + 0.9 * (1 - f)
            for frac, level in ((1.1, 0.08), (0.9, 0.15), (0.65, 0.28), (0.4, 0.5), (0.2, 0.8), (0.08, 1.0)):
                col = WHITE if frac < 0.2 else (GOLD if frac < 0.4 else CYAN)
                self._circle(cx, cy, R * frac * grow, sc(col, level * f * f), -1)
        if self.shock >= 0:
            p = self.shock
            r = R * (0.85 + 1.9 * ease_out(p))
            k = (1 - p) ** 1.4
            self._circle(cx, cy, r, sc(CYAN, 0.95 * k), 3)
            self._circle(cx, cy, r * 0.85, sc(PURPLE, 0.7 * k), 2)
            self._circle(cx, cy, r * 0.68, sc(GOLD, 0.35 * k), 1)

    # ── Mode A: Energy Core ───────────────────────────────────────────
    def _draw_energy(self):
        cx, cy = float(self.center[0]), float(self.center[1])
        k = ease_out(self.appear) if self.active else self.dissolve
        R = self.radius * (0.3 + 0.7 * k)
        t = self.t
        intens = 0.7 + 0.3 * self.intensity
        b = k * intens * (0.8 + 0.2 * math.sin(t * 3.5))
        s = min(max(R / 140, 0.55), 1.7)

        def vis(i):
            return min(max((self.appear if self.active else 1.0) * 1.7 - 0.15 * i, 0), 1) * b

        # Beams from palms
        if self.palms and self.active:
            for px, py in self.palms:
                for frac, lv in ((0.0, 0.15), (0.4, 0.4), (0.75, 0.65), (1.0, 0.9)):
                    x = px + (cx - px) * frac
                    y = py + (cy - py) * frac
                    self._circle(x, y, (3 + 5 * (1 - abs(frac - 0.5) * 1.5)) * k, sc(CYAN, lv * k), -1 if frac > 0.5 else 1)
                self._line(px, py, cx, cy, sc(CYAN, 0.5 * k), 1)
                self._line(px, py, cx, cy, sc(BLUE, 0.18 * k), 3)
                self._circle(px, py, 8 * k, sc(CYAN, 0.35 * k), 1)
                self._circle(px, py, 3.5 * k, sc(WHITE, 0.7 * k), -1)

        # Sweep arcs
        r78 = _fp(R * 0.80)
        sweep = (t * 110) % 360
        for width, lv in ((50, 0.04), (32, 0.08), (18, 0.12), (6, 0.35)):
            cv2.ellipse(self.layer, (_fp(cx), _fp(cy)), (r78, r78), 0,
                        sweep - width, sweep, sc(CYAN, lv * vis(2)), -1, cv2.LINE_AA, SHIFT)

        # Rings
        for i, (frac, lv) in enumerate(((1.05, 0.2), (0.92, 0.3), (0.78, 0.28), (0.62, 0.3), (0.45, 0.25), (0.28, 0.2))):
            self._circle(cx, cy, R * frac, sc(CYAN, lv * vis(i)), 1)

        self._arcs(cx, cy, R * 1.02, 4, 55, t * 45, sc(CYAN, 0.95 * vis(0)), max(1, int(3 * s)))
        self._arcs(cx, cy, R * 0.88, 6, 28, -t * 70, sc(BLUE, 0.9 * vis(1)), max(1, int(2 * s)))
        self._arcs(cx, cy, R * 0.70, 3, 100, t * 115, sc(PURPLE, 0.85 * vis(2)), max(1, int(3 * s)))
        self._arcs(cx, cy, R * 0.50, 5, 40, -t * 160, sc(CYAN, 0.8 * vis(3)), max(1, int(2 * s)))

        # Polyhedra
        self._poly(cx, cy, R * 0.75, 6, t * 0.5, sc(CYAN, 0.35 * vis(2)))
        self._poly(cx, cy, R * 0.55, 8, -t * 0.7, sc(BLUE, 0.3 * vis(3)))
        self._poly(cx, cy, R * 0.40, 3, t * 1.1, sc(PURPLE, 0.4 * vis(3)), 2)

        # Nodes
        for i in range(10):
            a = t * 0.95 + i * TAU / 10
            x, y = cx + R * 0.90 * math.cos(a), cy + R * 0.90 * math.sin(a)
            self._circle(x, y, 6.5 * s, sc(CYAN, 0.35 * vis(2)), 1)
            self._circle(x, y, 2.5 * s, sc(WHITE, vis(2)), -1)
        for i in range(6):
            a = -t * 1.3 + i * TAU / 6
            x, y = cx + R * 0.65 * math.cos(a), cy + R * 0.65 * math.sin(a)
            self._circle(x, y, 5 * s, sc(PURPLE, 0.35 * vis(3)), 1)
            self._circle(x, y, 2 * s, sc(PURPLE, vis(3)), -1)

        # Core
        cp = 1.0 + 0.12 * math.sin(t * 6.5)
        for frac, col, lv in ((0.36, BLUE, 0.1), (0.26, BLUE, 0.2), (0.18, CYAN, 0.4),
                               (0.11, CYAN, 0.7), (0.06, WHITE, 1.0)):
            self._circle(cx, cy, R * frac * cp, sc(col, lv * vis(4)), -1)
        self._circle(cx, cy, R * 0.40 * cp, sc(PURPLE, 0.5 * vis(4)), max(1, int(2 * s)))

        # Spokes
        for i in range(12):
            a = t * 0.65 + i * TAU / 12
            length = R * (0.3 + 0.15 * math.sin(t * 7 + i * 1.7))
            ca, sa = math.cos(a), math.sin(a)
            col = PURPLE if i % 3 == 0 else (GOLD if i % 2 == 0 else CYAN)
            self._line(cx + R * 0.08 * ca, cy + R * 0.08 * sa,
                       cx + length * ca, cy + length * sa, sc(col, 0.7 * vis(4)))

        # Orbiters
        ang = self.orb_a + self.orb_w * t
        rad = self.orb_r * R * (1 + 0.05 * np.sin(t * 2.2 + self.orb_a))
        xy = np.stack((cx + rad * np.cos(ang), cy + rad * np.sin(ang)), 1).astype(np.float32)
        tw = (0.4 + 0.6 * np.sin(t * 4.5 + self.orb_ph) ** 2) * vis(2)
        cols = np.minimum(np.array(CYAN, np.float32) * tw[:, None], 255).astype(np.uint8)
        self._splat(xy, cols)

        # Scan lines
        y_scan = R * 0.9 * math.sin(t * 1.8)
        direction = 1 if math.cos(t * 1.8) >= 0 else -1
        for j, lv in enumerate((0.6, 0.3, 0.15)):
            y = y_scan - direction * j * 5.5 * s
            half = math.sqrt(max(R * R * 0.95 - y * y, 0))
            if half > 2:
                self._line(cx - half, cy + y, cx + half, cy + y, sc(CYAN, lv * vis(1)))

    # ── Mode B: Portal ────────────────────────────────────────────────
    def _draw_portal(self):
        cx, cy = float(self.center[0]), float(self.center[1])
        k = ease_out(self.appear) if self.active else self.dissolve
        R = self.radius * (0.3 + 0.7 * k)
        t = self.t
        b = k * (0.75 + 0.25 * self.intensity)

        # Spiral particles
        ang = self.sp_a + self.sp_w * t * 0.6
        # Spiral radius grows with angle offset for vortex feel
        rad = self.sp_r * R * (0.15 + 0.85 * ((ang * 0.15) % 1.0))
        # Depth scaling
        depth = self.sp_z
        sz = (0.6 + 1.4 * depth) * b
        xy = np.stack((cx + rad * np.cos(ang), cy + rad * np.sin(ang)), 1).astype(np.float32)
        fade = (0.3 + 0.7 * depth) * b * (0.5 + 0.5 * np.sin(t * 3 + self.sp_a))
        mix = (self.sp_r > 0.6).astype(np.float32)
        base = np.array(CYAN, np.float32) * (1 - mix[:, None]) + np.array(PURPLE, np.float32) * mix[:, None]
        colors = np.minimum(base * fade[:, None], 255).astype(np.uint8)
        self._splat(xy, colors)

        # Concentric rings with rotation
        for i, frac in enumerate((1.05, 0.9, 0.75, 0.58, 0.4, 0.22)):
            self._arcs(cx, cy, R * frac, 3 + i % 3, 70 - i * 8,
                       t * (40 + i * 25) * (1 if i % 2 == 0 else -1),
                       sc(CYAN if i % 2 == 0 else PURPLE, (0.7 - i * 0.08) * b),
                       2 if i < 2 else 1)

        # Outer glow ring
        self._circle(cx, cy, R * 1.12, sc(BLUE, 0.2 * b), 2)
        self._circle(cx, cy, R * 1.18, sc(PURPLE, 0.1 * b), 1)

        # Radial streaks
        for i in range(16):
            a = t * 0.3 + i * TAU / 16
            r0, r1 = R * 0.15, R * (0.7 + 0.25 * math.sin(t * 2 + i))
            ca, sa = math.cos(a), math.sin(a)
            self._line(cx + r0 * ca, cy + r0 * sa, cx + r1 * ca, cy + r1 * sa,
                       sc(CYAN if i % 2 == 0 else GOLD, 0.45 * b))

        # Bright core
        cp = 1 + 0.15 * math.sin(t * 5)
        for frac, col, lv in ((0.2, PURPLE, 0.3), (0.12, CYAN, 0.6), (0.05, WHITE, 1.0)):
            self._circle(cx, cy, R * frac * cp, sc(col, lv * b), -1)

        # Beams
        if self.palms and self.active:
            for px, py in self.palms:
                self._line(px, py, cx, cy, sc(PURPLE, 0.4 * k), 1)
                self._line(px, py, cx, cy, sc(CYAN, 0.2 * k), 2)

    # ── Mode C: Holographic Earth ──────────────────────────────────────
    def _draw_earth(self):
        cx, cy = float(self.center[0]), float(self.center[1])
        k = ease_out(self.appear) if self.active else self.dissolve
        scale = self.pinch_scale
        R = self.radius * (0.3 + 0.7 * k) * scale * 0.85
        t = self.t
        # Rotation from wrist average
        rot = 0.5 * sum(self.wrist_angles) + t * 0.4
        b = k * (0.8 + 0.2 * self.intensity)

        # Latitude lines (ellipses with perspective)
        for lat in np.linspace(-0.85, 0.85, 9):
            ry = R * math.sqrt(max(1 - lat ** 2, 0.01))
            cy_off = cy + R * lat * 0.55
            # Perspective squash
            axes = (_fp(R * 0.95), _fp(ry * 0.35))
            color = sc(CYAN, (0.25 + 0.35 * (1 - abs(lat))) * b)
            cv2.ellipse(self.layer, (_fp(cx), _fp(cy_off)), axes, 0, 0, 360,
                        color, 1, cv2.LINE_AA, SHIFT)

        # Longitude lines
        for i in range(12):
            a0 = rot + i * TAU / 12
            pts = []
            for lat in np.linspace(-1, 1, 24):
                x = cx + R * math.cos(lat * math.pi / 2) * math.cos(a0) * 0.95
                y = cy + R * lat * 0.55
                # Simple back-face cull
                if math.cos(a0) > -0.15 or abs(lat) > 0.85:
                    pts.append((_fp(x), _fp(y)))
            if len(pts) > 2:
                arr = np.array(pts, np.int32)
                cv2.polylines(self.layer, [arr], False, sc(CYAN, 0.4 * b), 1, cv2.LINE_AA, SHIFT)

        # Outer sphere outline
        self._circle(cx, cy, R, sc(CYAN, 0.55 * b), 2)
        self._circle(cx, cy, R * 1.04, sc(BLUE, 0.2 * b), 1)

        # Equator highlight
        axes = (_fp(R * 0.97), _fp(R * 0.18))
        cv2.ellipse(self.layer, (_fp(cx), _fp(cy)), axes, 0, 0, 360,
                    sc(GOLD, 0.45 * b), 1, cv2.LINE_AA, SHIFT)

        # Orbiting rings
        for ri, (frac, speed, col) in enumerate(((1.25, 0.5, BLUE), (1.4, -0.35, PURPLE), (1.55, 0.25, CYAN))):
            self._arcs(cx, cy, R * frac, 2, 100, t * speed * 60 + ri * 40,
                       sc(col, 0.5 * b), 1)

        # Satellites
        for i in range(8):
            a = self.sat_a[i] + self.sat_w[i] * t
            inc = self.sat_inc[i]
            x = cx + R * 1.35 * math.cos(a)
            y = cy + R * 1.35 * math.sin(a) * inc
            self._circle(x, y, 4, sc(GOLD, 0.7 * b), -1)
            self._circle(x, y, 7, sc(CYAN, 0.25 * b), 1)

        # Scan arc
        sweep = (t * 80) % 360
        cv2.ellipse(self.layer, (_fp(cx), _fp(cy)), (_fp(R * 0.98), _fp(R * 0.98)),
                    0, sweep - 40, sweep, sc(CYAN, 0.35 * b), 2, cv2.LINE_AA, SHIFT)

        # Core glow
        self._circle(cx, cy, R * 0.15, sc(CYAN, 0.3 * b), -1)
        self._circle(cx, cy, R * 0.08, sc(WHITE, 0.6 * b), -1)

        # Hand beams
        if self.palms and self.active:
            for px, py in self.palms:
                self._line(px, py, cx, cy, sc(CYAN, 0.3 * k), 1)

    # ── Mode D: Energy Beam ───────────────────────────────────────────
    def _draw_beam(self):
        if not self.palms or len(self.palms) < 2:
            return
        k = ease_out(self.appear) if self.active else self.dissolve
        if k < 0.05:
            return
        (x0, y0), (x1, y1) = self.palms
        t = self.t
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy) + 1e-3
        nx, ny = -dy / dist, dx / dist  # perpendicular

        # Thickness based on distance (closer = thicker)
        thick = max(2.0, min(14.0, 180.0 / (dist * 0.15 + 10))) * k
        intens = 0.7 + 0.3 * self.intensity

        # Core beam layers
        for frac, lv, th_mul in ((0.0, 0.15, 3.5), (0.3, 0.35, 2.2), (0.6, 0.55, 1.3), (1.0, 0.95, 0.6)):
            self._line(x0, y0, x1, y1, sc(CYAN, lv * k * intens), max(1, int(thick * th_mul)))

        # Traveling pulses
        for p in range(5):
            frac = ((t * 1.8 + p * 0.22) % 1.0)
            px = x0 + dx * frac
            py = y0 + dy * frac
            self._circle(px, py, thick * 0.9, sc(WHITE, 0.8 * k), -1)
            self._circle(px, py, thick * 1.6, sc(CYAN, 0.3 * k), 1)

        # Electrical branches
        n_branch = 8
        for i in range(n_branch):
            frac = (i + 0.5) / n_branch
            bx = x0 + dx * frac
            by = y0 + dy * frac
            # Jitter direction
            j = math.sin(t * 12 + i * 2.7) * thick * 2.5
            j2 = math.cos(t * 9 + i * 1.9) * thick * 1.8
            ex = bx + nx * j + dx / dist * j2 * 0.3
            ey = by + ny * j + dy / dist * j2 * 0.3
            col = PURPLE if i % 2 == 0 else GOLD
            self._line(bx, by, ex, ey, sc(col, 0.55 * k), 1)
            self._circle(ex, ey, 2.5, sc(col, 0.7 * k), -1)

        # Hand nodes
        for px, py in self.palms:
            self._circle(px, py, thick * 1.2, sc(CYAN, 0.4 * k), 1)
            self._circle(px, py, thick * 0.5, sc(WHITE, 0.85 * k), -1)
            # Outer arc
            self._arcs(px, py, thick * 2.5, 3, 50, t * 90, sc(CYAN, 0.35 * k), 1)

        # Sparks along beam
        n_spark = 20
        ang = self.rng.uniform(0, TAU, n_spark)  # static-ish
        for i in range(n_spark):
            frac = (i + 0.3 * math.sin(t * 5 + i)) / n_spark
            sx = x0 + dx * frac + nx * math.sin(t * 8 + i * 1.3) * thick * 1.5
            sy = y0 + dy * frac + ny * math.sin(t * 8 + i * 1.3) * thick * 1.5
            self._circle(sx, sy, 1.5 + 1.5 * abs(math.sin(t * 6 + i)), sc(GOLD, 0.6 * k), -1)

    # ── Mode E: Particle Field ────────────────────────────────────────

    # ── Mode E: Energy Blast (charge → aim → launch → impact) ─────────
    def _update_blast(self, dt, control: Control):
        self.blast_state = control.blast_state
        self.charge_level = control.charge
        self.aim_origin[:] = control.aim_origin
        self.aim_dir[:] = control.aim_dir
        self.palms = control.palms
        self.intensity = control.intensity

        if control.blast_state in (BlastState.CHARGING, BlastState.READY, BlastState.AIMING):
            self.active = True
            self.appear = min(1.0, self.appear + dt / 0.35)
            if control.palms:
                self.center[:] = control.center
        elif control.blast_state == BlastState.IDLE:
            self.active = False
            self.appear = max(0.0, self.appear - dt / 0.25)

        # Launch projectile
        if control.launch and not self.proj_alive:
            self.proj_alive = True
            self.proj_pos[:] = control.aim_origin
            speed = 520.0 + 280.0 * control.charge
            self.proj_vel[:] = np.asarray(control.aim_dir, np.float32) * speed
            self.proj_life = 1.6
            self.proj_charge = max(0.5, control.charge)
            self.proj_trail[:] = self.proj_pos
            self.proj_trail_i = 0
            self.flash = 0.9
            self.shock = 0.0
            self.blast_c[:] = control.aim_origin
            self.blast_r = 40.0 + 30.0 * control.charge
            self.pool.emit(self.proj_pos, 60, speed_scale=90, life_range=(0.3, 0.7), mix_prob=0.5)
            self.active = False

        # Advance projectile (independent of hands)
        if self.proj_alive:
            self.proj_vel *= math.exp(-0.15 * dt)  # slight drag
            # slight acceleration feel at start
            self.proj_pos += self.proj_vel * dt
            self.proj_life -= dt
            self.proj_trail[self.proj_trail_i % 24] = self.proj_pos
            self.proj_trail_i += 1
            # continuous spark trail
            if self.rng.random() < 0.55:
                self.pool.emit(self.proj_pos, 3, speed_scale=40, life_range=(0.15, 0.45),
                               radial=True, mix_prob=0.4, size_range=(1.0, 2.0))
            # off-screen or life over → impact or fade
            x, y = float(self.proj_pos[0]), float(self.proj_pos[1])
            margin = 40
            if (self.proj_life <= 0 or x < -margin or x > self.w + margin
                    or y < -margin or y > self.h + margin):
                if 0 <= x <= self.w and 0 <= y <= self.h:
                    self._trigger_impact(self.proj_pos.copy())
                self.proj_alive = False
                self.proj_life = 0.0

        # Impact animation
        if self.impact_t >= 0:
            self.impact_t += dt / 0.7
            if self.impact_t > 1.0:
                self.impact_t = -1.0
        if self.shake > 0:
            self.shake = max(0.0, self.shake - dt * 4.0)

    def _trigger_impact(self, pos):
        self.impact_t = 0.0
        self.impact_pos[:] = pos
        self.flash = 1.0
        self.shock = 0.0
        self.blast_c[:] = pos
        self.blast_r = 55.0 + 40.0 * self.proj_charge
        self.shake = 0.55
        self.pool.emit(pos, 220, speed_scale=140 + 80 * self.proj_charge,
                       life_range=(0.5, 1.4), mix_prob=0.5)

    def _draw_blast_mode(self):
        st = self.blast_state
        ch = self.charge_level
        t = self.t

        # Charging / ready core between hands
        if st in (BlastState.CHARGING, BlastState.READY, BlastState.AIMING) and self.palms:
            (p0, p1) = self.palms
            cx = 0.5 * (p0[0] + p1[0])
            cy = 0.5 * (p0[1] + p1[1])
            if st == BlastState.AIMING:
                cx, cy = float(self.aim_origin[0]), float(self.aim_origin[1])
            k = ease_out(min(1.0, ch + 0.15))
            R = (18 + 42 * ch) * k

            # Electrical arcs between palms while charging
            if st in (BlastState.CHARGING, BlastState.READY):
                for i in range(5):
                    mid_x = cx + math.sin(t * 14 + i * 1.7) * (8 + 10 * ch)
                    mid_y = cy + math.cos(t * 11 + i * 2.1) * (6 + 8 * ch)
                    col = PURPLE if i % 2 else CYAN
                    self._line(p0[0], p0[1], mid_x, mid_y, sc(col, 0.45 * k), 1)
                    self._line(mid_x, mid_y, p1[0], p1[1], sc(col, 0.45 * k), 1)
                # palm nodes
                for px, py in (p0, p1):
                    self._circle(px, py, 7 * k, sc(CYAN, 0.4 * k), 1)
                    self._circle(px, py, 3 * k, sc(WHITE, 0.8 * k), -1)

            # Swirling energy strands
            for i in range(8):
                a = t * (2.5 + ch) + i * TAU / 8
                rr = R * (0.6 + 0.4 * math.sin(t * 5 + i))
                x = cx + rr * math.cos(a)
                y = cy + rr * math.sin(a)
                self._circle(x, y, 2.5, sc(CYAN if i % 2 == 0 else PURPLE, 0.7 * k), -1)

            # Core
            pulse = 0.85 + 0.15 * math.sin(t * (6 + 8 * ch))
            for frac, col, lv in ((1.1, BLUE, 0.15), (0.75, CYAN, 0.35),
                                   (0.45, CYAN, 0.7), (0.2, WHITE, 1.0)):
                self._circle(cx, cy, R * frac * pulse, sc(col, lv * k), -1)
            self._circle(cx, cy, R * 1.25, sc(PURPLE, 0.25 * k), 2)
            self._arcs(cx, cy, R * 1.35, 3, 50, t * 90, sc(CYAN, 0.5 * k), 2)

            # Charge ring progress
            if st == BlastState.CHARGING:
                angle = 360 * ch
                cv2.ellipse(self.layer, (_fp(cx), _fp(cy)), (_fp(R * 1.5), _fp(R * 1.5)),
                            -90, 0, angle, sc(GOLD, 0.7 * k), 2, cv2.LINE_AA, SHIFT)

        # Aiming line
        if st == BlastState.AIMING:
            ox, oy = float(self.aim_origin[0]), float(self.aim_origin[1])
            dx, dy = float(self.aim_dir[0]), float(self.aim_dir[1])
            length = 180 + 80 * ch
            # subtle dashed aim line
            for i in range(12):
                f0, f1 = i / 12, (i + 0.55) / 12
                self._line(ox + dx * length * f0, oy + dy * length * f0,
                           ox + dx * length * f1, oy + dy * length * f1,
                           sc(CYAN, 0.25 + 0.15 * math.sin(t * 8 + i)), 1)
            self._circle(ox, oy, 10, sc(CYAN, 0.4), 1)
            self._circle(ox, oy, 4, sc(WHITE, 0.7), -1)

        # Projectile in flight
        if self.proj_alive:
            px, py = float(self.proj_pos[0]), float(self.proj_pos[1])
            ch = self.proj_charge
            # long luminous trail
            n_trail = min(self.proj_trail_i, 24)
            for i in range(n_trail - 1):
                a = self.proj_trail[(self.proj_trail_i - 1 - i) % 24]
                b = self.proj_trail[(self.proj_trail_i - 2 - i) % 24]
                fade = 1.0 - i / max(n_trail, 1)
                self._line(float(a[0]), float(a[1]), float(b[0]), float(b[1]),
                           sc(CYAN, 0.55 * fade), max(1, int(3 * fade * ch)))
                self._line(float(a[0]), float(a[1]), float(b[0]), float(b[1]),
                           sc(WHITE, 0.25 * fade), 1)
            # electrical branches off the projectile
            for i in range(4):
                a = t * 20 + i * 1.7
                bx = px + math.cos(a) * (12 + 8 * ch)
                by = py + math.sin(a) * (12 + 8 * ch)
                self._line(px, py, bx, by, sc(PURPLE, 0.55), 1)
            # core
            r = 6 + 8 * ch
            self._circle(px, py, r * 2.2, sc(BLUE, 0.25), -1)
            self._circle(px, py, r * 1.4, sc(CYAN, 0.5), -1)
            self._circle(px, py, r * 0.7, sc(WHITE, 0.95), -1)

        # Impact
        if self.impact_t >= 0:
            p = self.impact_t
            ix, iy = float(self.impact_pos[0]), float(self.impact_pos[1])
            k = (1.0 - p) ** 1.2
            R = self.blast_r * (0.4 + 1.2 * ease_out(p))
            self._circle(ix, iy, R, sc(CYAN, 0.7 * k), 3)
            self._circle(ix, iy, R * 0.75, sc(PURPLE, 0.5 * k), 2)
            self._circle(ix, iy, R * 0.45, sc(GOLD, 0.35 * k), 1)
            if p < 0.35:
                self._circle(ix, iy, 30 * (1 - p / 0.35), sc(WHITE, 0.8 * (1 - p / 0.35)), -1)

    # ── Trails & composite ────────────────────────────────────────────
    # ── Trails & composite ────────────────────────────────────────────
    def _draw_trails(self):
        order = np.arange(self.trail_len)
        for slot in range(2):
            idx = (self.trail_i[slot] - 1 - order) % self.trail_len
            pts, times = self.trail_xy[slot, idx], self.trail_t[slot, idx]
            age = self.t - times
            for j in range(self.trail_len - 1):
                if age[j] > 0.55 or age[j + 1] > 0.55 or times[j] - times[j + 1] > 0.12:
                    break
                lv = 0.6 * (1 - age[j] / 0.55)
                self._line(pts[j, 0], pts[j, 1], pts[j + 1, 0], pts[j + 1, 1],
                           sc(CYAN if j < 10 else BLUE, lv), 2 if j < 6 else 1)

    def _darken(self, frame):
        p = self.presence
        if p < 0.01:
            return
        cv2.addWeighted(self._white, 1 - p, self._mask, p, 0, dst=self._tmp)
        cv2.multiply(frame, self._tmp, dst=frame, scale=1 / 255)

    def _composite(self, frame):
        cv2.resize(self.layer, (self._w4, self._h4), dst=self._s4, interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(self._s4, (0, 0), 1.6, dst=self._b4)
        cv2.resize(self._s4, (self._w8, self._h8), dst=self._s8, interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(self._s8, (0, 0), 1.6, dst=self._b8)
        cv2.resize(self._b4, (self.w, self.h), dst=self._glow, interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(frame, 1.0, self._glow, 2.5, 0, dst=frame)
        cv2.resize(self._b8, (self.w, self.h), dst=self._glow, interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(frame, 1.0, self._glow, 2.9, 0, dst=frame)
        cv2.add(frame, self.layer, dst=frame)
