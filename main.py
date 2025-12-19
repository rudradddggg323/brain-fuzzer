#!/usr/bin/env python3
"""
Brain Fuzzer: Visual Perception Research Tool

A psychedelic visual effects generator designed to explore the boundaries
of human visual perception through rapidly changing geometric patterns,
color cycling, and perceptual illusions.

WARNING: May cause disorientation, afterimages, or discomfort.
Not recommended for individuals with photosensitive epilepsy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pygame
from numpy.typing import NDArray


@dataclass
class Config:
    """Global configuration for the fuzzer."""

    width: int = 800
    height: int = 800
    fullscreen: bool = False
    target_fps: int = 60
    session_duration: float = 120.0
    intensity: float = 0.7
    chaos: float = 0.5
    enable_tesseract: bool = True
    enable_fractals: bool = True
    enable_spirals: bool = True
    enable_waves: bool = True
    enable_particles: bool = True
    enable_flicker: bool = True
    enable_distortion: bool = True
    enable_chromatic: bool = True
    enable_scanlines: bool = True
    enable_crt: bool = True
    color_cycle_speed: float = 0.02
    high_contrast_mode: bool = False
    color_mode: str = "psychedelic"
    log_level: int = logging.INFO
    preset_file: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["log_level"] = logging.getLevelName(data["log_level"])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        if "log_level" in data and isinstance(data["log_level"], str):
            data["log_level"] = getattr(logging, data["log_level"], logging.INFO)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_preset(self, filepath: str) -> None:
        Path(filepath).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_preset(cls, filepath: str) -> "Config":
        data = json.loads(Path(filepath).read_text())
        return cls.from_dict(data)


@dataclass
class State:
    """Mutable runtime state."""

    frame: int = 0
    time_elapsed: float = 0.0
    delta_time: float = 0.0
    running: bool = True
    paused: bool = False
    rotation_xy: float = 0.0
    rotation_zw: float = 0.0
    rotation_speed_xy: float = 0.05
    rotation_speed_zw: float = 0.03
    hue_offset: float = 0.0
    message: str = ""
    message_timer: float = 0.0


class ColorMode(Enum):
    PSYCHEDELIC = auto()
    MONOCHROME = auto()
    COMPLEMENTARY = auto()
    FIRE = auto()
    ICE = auto()
    MATRIX = auto()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain Fuzzer: Visual Perception Research Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --fullscreen --intensity 0.9
  %(prog)s --duration 60 --chaos 0.8
  %(prog)s --preset my_config.json
  %(prog)s --width 1920 --height 1080 --save-preset default.json
        """,
    )

    display = parser.add_argument_group("Display")
    display.add_argument("-W", "--width", type=int, default=800, help="Window width")
    display.add_argument("-H", "--height", type=int, default=800, help="Window height")
    display.add_argument(
        "-f", "--fullscreen", action="store_true", help="Fullscreen mode"
    )
    display.add_argument("--fps", type=int, default=60, help="Target FPS")

    session = parser.add_argument_group("Session")
    session.add_argument(
        "-d",
        "--duration",
        type=float,
        default=120.0,
        help="Session duration (0=infinite)",
    )
    session.add_argument(
        "-i", "--intensity", type=float, default=0.7, help="Effect intensity (0.1-1.0)"
    )
    session.add_argument(
        "-c", "--chaos", type=float, default=0.5, help="Chaos level (0.0-1.0)"
    )
    session.add_argument(
        "--color-mode",
        choices=["psychedelic", "fire", "ice", "matrix", "monochrome"],
        default="psychedelic",
    )
    session.add_argument(
        "--high-contrast", action="store_true", help="High contrast flicker mode"
    )

    effects = parser.add_argument_group("Effects (disable with --no-<effect>)")
    effects.add_argument(
        "--no-tesseract", action="store_true", help="Disable tesseract effect"
    )
    effects.add_argument(
        "--no-fractals", action="store_true", help="Disable fractal effect"
    )
    effects.add_argument(
        "--no-spirals", action="store_true", help="Disable spiral effect"
    )
    effects.add_argument("--no-waves", action="store_true", help="Disable wave effect")
    effects.add_argument(
        "--no-particles", action="store_true", help="Disable particle effect"
    )
    effects.add_argument(
        "--no-flicker", action="store_true", help="Disable flicker effect"
    )
    effects.add_argument(
        "--no-distortion", action="store_true", help="Disable distortion effect"
    )
    effects.add_argument(
        "--no-chromatic", action="store_true", help="Disable chromatic aberration"
    )
    effects.add_argument(
        "--no-scanlines", action="store_true", help="Disable scanline effect"
    )
    effects.add_argument("--no-crt", action="store_true", help="Disable CRT effect")

    presets = parser.add_argument_group("Presets")
    presets.add_argument("-p", "--preset", type=str, help="Load preset from JSON file")
    presets.add_argument(
        "--save-preset", type=str, metavar="FILE", help="Save config to JSON and exit"
    )
    presets.add_argument(
        "--list-presets", action="store_true", help="List built-in presets"
    )

    debug = parser.add_argument_group("Debug")
    debug.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    debug.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode (errors only)"
    )
    debug.add_argument(
        "--skip-intro", action="store_true", help="Skip startup animation"
    )

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    if args.preset and Path(args.preset).exists():
        config = Config.load_preset(args.preset)
    else:
        config = Config()

    config.width = args.width
    config.height = args.height
    config.fullscreen = args.fullscreen
    config.target_fps = args.fps
    config.session_duration = args.duration
    config.intensity = max(0.1, min(1.0, args.intensity))
    config.chaos = max(0.0, min(1.0, args.chaos))
    config.color_mode = args.color_mode
    config.high_contrast_mode = args.high_contrast

    config.enable_tesseract = not args.no_tesseract
    config.enable_fractals = not args.no_fractals
    config.enable_spirals = not args.no_spirals
    config.enable_waves = not args.no_waves
    config.enable_particles = not args.no_particles
    config.enable_flicker = not args.no_flicker
    config.enable_distortion = not args.no_distortion
    config.enable_chromatic = not args.no_chromatic
    config.enable_scanlines = not args.no_scanlines
    config.enable_crt = not args.no_crt

    if args.verbose:
        config.log_level = logging.DEBUG
    elif args.quiet:
        config.log_level = logging.ERROR

    return config


BUILTIN_PRESETS = {
    "mellow": {"intensity": 0.4, "chaos": 0.2, "color_mode": "ice"},
    "intense": {"intensity": 1.0, "chaos": 0.8, "color_mode": "fire"},
    "matrix": {
        "intensity": 0.6,
        "chaos": 0.3,
        "color_mode": "matrix",
        "enable_flicker": False,
    },
    "minimal": {
        "intensity": 0.5,
        "chaos": 0.1,
        "enable_fractals": False,
        "enable_particles": False,
    },
    "chaos": {"intensity": 1.0, "chaos": 1.0, "high_contrast_mode": True},
}


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (0-1 range) to RGB (0-255 range)."""
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]

    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


_current_color_mode: str = "psychedelic"


def set_color_mode(mode: str) -> None:
    global _current_color_mode
    _current_color_mode = mode


def get_color(
    offset: float = 0.0, saturation: float = 1.0, value: float = 1.0
) -> tuple[int, int, int]:
    hue = (time.time() * 0.5 + offset) % 1.0
    mode = _current_color_mode

    if mode == "psychedelic":
        return hsv_to_rgb(hue, saturation, value)
    elif mode == "fire":
        r = int(255 * value)
        g = int((100 + 155 * hue) * value)
        b = int(50 * (1 - hue) * value)
        return (r, g, b)
    elif mode == "ice":
        r = int((100 + 100 * (1 - hue)) * value)
        g = int((180 + 75 * hue) * value)
        b = int(255 * value)
        return (r, min(255, g), b)
    elif mode == "matrix":
        intensity = int((100 + 155 * hue) * value)
        return (0, intensity, int(intensity * 0.3))
    elif mode == "monochrome":
        gray = int(255 * hue * value)
        return (gray, gray, gray)
    else:
        return hsv_to_rgb(hue, saturation, value)


def get_psychedelic_color(
    offset: float = 0.0, saturation: float = 1.0, value: float = 1.0
) -> tuple[int, int, int]:
    return get_color(offset, saturation, value)


def get_glitch_color() -> tuple[int, int, int]:
    mode = _current_color_mode
    if mode == "matrix":
        return (0, random.randint(150, 255), random.randint(0, 50))
    elif mode == "fire":
        return (255, random.randint(100, 200), random.randint(0, 50))
    elif mode == "ice":
        return (random.randint(100, 200), random.randint(180, 230), 255)
    elif mode == "monochrome":
        g = random.randint(100, 255)
        return (g, g, g)
    return (random.randint(180, 255), random.randint(0, 255), random.randint(0, 255))


def get_palette(mode: ColorMode, offset: float = 0.0) -> list[tuple[int, int, int]]:
    if mode == ColorMode.PSYCHEDELIC:
        return [hsv_to_rgb((offset + i * 0.1) % 1.0, 1.0, 1.0) for i in range(10)]
    elif mode == ColorMode.MONOCHROME:
        return [
            (int(255 * i / 9), int(255 * i / 9), int(255 * i / 9)) for i in range(10)
        ]
    elif mode == ColorMode.FIRE:
        return [(255, int(100 + i * 15), 0) for i in range(10)]
    elif mode == ColorMode.ICE:
        return [(int(100 + i * 15), int(200 + i * 5), 255) for i in range(10)]
    elif mode == ColorMode.MATRIX:
        return [(0, int(100 + i * 15), 0) for i in range(10)]
    else:
        return [get_glitch_color() for _ in range(10)]


def create_tesseract() -> NDArray[np.float64]:
    """Create a 4D tesseract (hypercube) with vertices at +/-1."""
    vertices = []
    for i in range(16):
        vertex = [
            ((i >> 0) & 1) * 2 - 1,
            ((i >> 1) & 1) * 2 - 1,
            ((i >> 2) & 1) * 2 - 1,
            ((i >> 3) & 1) * 2 - 1,
        ]
        vertices.append(vertex)
    return np.array(vertices, dtype=np.float64)


def create_tesseract_edges() -> list[tuple[int, int]]:
    """Create edges connecting tesseract vertices that differ by one bit."""
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            diff = i ^ j
            if diff & (diff - 1) == 0:
                edges.append((i, j))
    return edges


def create_warped_tesseract(warp: float = 0.3) -> NDArray[np.float64]:
    base = create_tesseract()
    noise = np.random.uniform(-warp, warp, base.shape)
    return base + noise


def rotate_4d(
    points: NDArray[np.float64], angle_xy: float, angle_zw: float, warp: float = 0.0
) -> NDArray[np.float64]:
    """Rotate points in 4D space around XY and ZW planes."""
    cos_xy, sin_xy = math.cos(angle_xy), math.sin(angle_xy)
    cos_zw, sin_zw = math.cos(angle_zw), math.sin(angle_zw)

    rot_xy = np.array(
        [[cos_xy, -sin_xy, 0, 0], [sin_xy, cos_xy, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )

    rot_zw = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, cos_zw, -sin_zw], [0, 0, sin_zw, cos_zw]],
        dtype=np.float64,
    )

    rotated = points @ rot_xy @ rot_zw

    if warp > 0:
        noise = np.random.uniform(-warp, warp, rotated.shape)
        rotated = rotated + noise

    return rotated


def project_4d_to_2d(
    points: NDArray[np.float64],
    width: int,
    height: int,
    distance: float = 2.0,
    fuzz: float = 0.0,
) -> list[tuple[float, float]]:
    """Project 4D points to 2D screen coordinates using perspective projection."""
    projected = []
    center_x, center_y = width / 2, height / 2
    scale = min(width, height) / 4

    for point in points:
        w_factor = distance - point[3]
        if w_factor < 0.1:
            w_factor = 0.1
        w = 1.0 / w_factor

        x = point[0] * w * scale + center_x
        y = point[1] * w * scale + center_y

        if fuzz > 0:
            x += random.uniform(-fuzz, fuzz) * 20
            y += random.uniform(-fuzz, fuzz) * 20

        projected.append((x, y))

    return projected


class Effect(ABC):
    """Abstract base class for visual effects."""

    def __init__(self, config: Config):
        self.config = config
        self.enabled = True

    @abstractmethod
    def update(self, state: State) -> None:
        pass

    @abstractmethod
    def render(self, surface: pygame.Surface, state: State) -> None:
        pass

    def toggle(self) -> None:
        self.enabled = not self.enabled


class TesseractEffect(Effect):
    """Rotating 4D hypercube projection with motion trails."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.vertices = create_warped_tesseract(warp=0.2)
        self.edges = create_tesseract_edges()
        self.trail: list[list[tuple[float, float]]] = []
        self.max_trail = 5

    def update(self, state: State) -> None:
        if random.random() < 0.01 * self.config.chaos:
            self.vertices = create_warped_tesseract(warp=0.3 * self.config.intensity)

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled:
            return

        warp = 0.1 * self.config.chaos * self.config.intensity
        rotated = rotate_4d(
            self.vertices, state.rotation_xy, state.rotation_zw, warp=warp
        )

        fuzz = self.config.chaos * self.config.intensity
        projected = project_4d_to_2d(
            rotated, self.config.width, self.config.height, fuzz=fuzz
        )

        self.trail.append(projected.copy())
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

        for i, trail_points in enumerate(self.trail):
            color = get_psychedelic_color(offset=i * 0.1 + state.hue_offset)
            thickness = max(1, int(3 * (i + 1) / len(self.trail)))

            for edge in self.edges:
                try:
                    p1 = trail_points[edge[0]]
                    p2 = trail_points[edge[1]]

                    if random.random() < 0.2 * self.config.chaos:
                        p2 = (
                            p2[0] + random.randint(-20, 20),
                            p2[1] + random.randint(-20, 20),
                        )

                    pygame.draw.line(surface, color, p1, p2, thickness)
                except (IndexError, TypeError):
                    pass

        for i, point in enumerate(projected):
            color = get_psychedelic_color(offset=i * 0.05 + state.hue_offset)
            radius = int(3 + 3 * self.config.intensity)
            try:
                pygame.draw.circle(
                    surface, color, (int(point[0]), int(point[1])), radius
                )
            except (ValueError, TypeError):
                pass


class FractalEffect(Effect):
    """Recursive fractal pattern generator."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.anchor_points: list[tuple[float, float]] = []
        self.regenerate_anchors()

    def regenerate_anchors(self) -> None:
        count = random.randint(3, 8)
        self.anchor_points = [
            (
                random.randint(100, self.config.width - 100),
                random.randint(100, self.config.height - 100),
            )
            for _ in range(count)
        ]

    def update(self, state: State) -> None:
        if random.random() < 0.02 * self.config.chaos:
            self.regenerate_anchors()

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled:
            return

        max_depth = int(2 + 3 * self.config.intensity)

        for anchor in self.anchor_points:
            self._draw_fractal(
                surface, anchor[0], anchor[1], depth=0, max_depth=max_depth, state=state
            )

    def _draw_fractal(
        self,
        surface: pygame.Surface,
        x: float,
        y: float,
        depth: int,
        max_depth: int,
        state: State,
    ) -> None:
        if depth > max_depth or random.random() > 0.85 - (0.1 * self.config.intensity):
            return

        scale = 40 / (depth + 1) * (0.5 + self.config.intensity)
        if scale < 2:
            return

        margin = scale * 2
        if not (
            margin < x < self.config.width - margin
            and margin < y < self.config.height - margin
        ):
            return

        color = get_psychedelic_color(
            offset=depth * 0.15 + state.hue_offset + x * 0.001
        )

        int_x, int_y = int(x), int(y)
        int_scale = int(scale)
        thickness = max(1, int(4 - depth))

        shape_type = random.choice(["circle", "polygon", "rect"])

        if shape_type == "circle":
            pygame.draw.circle(surface, color, (int_x, int_y), int_scale, thickness)
        elif shape_type == "rect":
            rect = pygame.Rect(
                int_x - int_scale // 2, int_y - int_scale // 2, int_scale, int_scale
            )
            pygame.draw.rect(surface, color, rect, thickness)
        else:
            num_sides = random.randint(3, 6)
            angle_offset = state.rotation_xy + depth
            points = [
                (
                    x + scale * math.cos(angle_offset + i * 2 * math.pi / num_sides),
                    y + scale * math.sin(angle_offset + i * 2 * math.pi / num_sides),
                )
                for i in range(num_sides)
            ]
            pygame.draw.polygon(surface, color, points, thickness)

        num_branches = random.randint(2, 5)
        for _ in range(num_branches):
            angle = random.uniform(0, 2 * math.pi)
            distance = scale * random.uniform(0.8, 2.0)
            new_x = x + distance * math.cos(angle)
            new_y = y + distance * math.sin(angle)
            self._draw_fractal(surface, new_x, new_y, depth + 1, max_depth, state)


class SpiralEffect(Effect):
    """Hypnotic spiral patterns."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.spirals: list[dict] = []
        self.spawn_spiral()

    def spawn_spiral(self) -> None:
        self.spirals.append(
            {
                "x": random.randint(100, self.config.width - 100),
                "y": random.randint(100, self.config.height - 100),
                "rotation": random.uniform(0, 2 * math.pi),
                "speed": random.uniform(0.02, 0.08)
                * (1 if random.random() > 0.5 else -1),
                "arms": random.randint(2, 6),
                "color_offset": random.uniform(0, 1),
                "scale": random.uniform(50, 150),
                "age": 0.0,
                "max_age": random.uniform(3.0, 8.0),
            }
        )

        if len(self.spirals) > 5:
            self.spirals.pop(0)

    def update(self, state: State) -> None:
        for spiral in self.spirals:
            spiral["rotation"] += spiral["speed"] * (1 + self.config.intensity)
            spiral["age"] += state.delta_time

        self.spirals = [s for s in self.spirals if s["age"] < s["max_age"]]

        if random.random() < 0.02 * self.config.chaos and len(self.spirals) < 5:
            self.spawn_spiral()

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled:
            return

        for spiral in self.spirals:
            self._draw_spiral(surface, spiral, state)

    def _draw_spiral(self, surface: pygame.Surface, spiral: dict, state: State) -> None:
        cx, cy = spiral["x"], spiral["y"]
        arms = spiral["arms"]
        rotation = spiral["rotation"]
        scale = spiral["scale"]
        color_offset = spiral["color_offset"]
        age_factor = 1 - (spiral["age"] / spiral["max_age"])

        for arm in range(arms):
            arm_angle = rotation + (arm * 2 * math.pi / arms)

            points = []
            for i in range(50):
                t = i / 50
                r = scale * t * (0.5 + 0.5 * self.config.intensity)
                angle = arm_angle + t * 4 * math.pi

                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                points.append((x, y))

            if len(points) > 1:
                color = get_psychedelic_color(
                    offset=color_offset + arm * 0.1 + state.hue_offset
                )
                color = tuple(int(c * age_factor) for c in color)
                thickness = max(1, int(3 * self.config.intensity * age_factor))
                pygame.draw.lines(surface, color, False, points, thickness)


class WaveEffect(Effect):
    """Oscillating wave distortion patterns."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.phase = 0.0
        self.waves: list[dict] = [
            {"freq": 0.02, "amp": 30, "speed": 1.0, "vertical": False},
            {"freq": 0.015, "amp": 25, "speed": 1.3, "vertical": True},
            {"freq": 0.025, "amp": 20, "speed": 0.8, "vertical": False},
        ]

    def update(self, state: State) -> None:
        self.phase += state.delta_time * 2 * (1 + self.config.intensity)

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled:
            return

        for wave in self.waves:
            self._draw_wave(surface, wave, state)

    def _draw_wave(self, surface: pygame.Surface, wave: dict, state: State) -> None:
        freq = wave["freq"]
        amp = wave["amp"] * self.config.intensity
        speed = wave["speed"]
        vertical = wave["vertical"]

        points = []

        if vertical:
            for y in range(0, self.config.height, 5):
                x = self.config.width // 2 + amp * math.sin(
                    freq * y + self.phase * speed
                )
                points.append((x, y))
        else:
            for x in range(0, self.config.width, 5):
                y = self.config.height // 2 + amp * math.sin(
                    freq * x + self.phase * speed
                )
                points.append((x, y))

        if len(points) > 1:
            color = get_psychedelic_color(offset=speed + state.hue_offset)
            pygame.draw.lines(surface, color, False, points, 2)


class ParticleEffect(Effect):
    """Floating particle system."""

    @dataclass
    class Particle:
        x: float
        y: float
        vx: float
        vy: float
        life: float
        max_life: float
        color_offset: float
        size: float

    def __init__(self, config: Config):
        super().__init__(config)
        self.particles: list[ParticleEffect.Particle] = []
        self.max_particles = 200

    def spawn_particle(self, x: float | None = None, y: float | None = None) -> None:
        if len(self.particles) >= self.max_particles:
            return

        if x is None:
            x = random.randint(0, self.config.width)
        if y is None:
            y = random.randint(0, self.config.height)

        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(20, 100) * self.config.intensity

        self.particles.append(
            ParticleEffect.Particle(
                x=x,
                y=y,
                vx=speed * math.cos(angle),
                vy=speed * math.sin(angle),
                life=0.0,
                max_life=random.uniform(1.0, 4.0),
                color_offset=random.uniform(0, 1),
                size=random.uniform(2, 8),
            )
        )

    def update(self, state: State) -> None:
        dt = state.delta_time

        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life += dt

            if random.random() < 0.1 * self.config.chaos:
                p.vx += random.uniform(-50, 50)
                p.vy += random.uniform(-50, 50)

        self.particles = [p for p in self.particles if p.life < p.max_life]

        spawn_rate = int(5 * self.config.intensity * self.config.chaos)
        for _ in range(spawn_rate):
            if random.random() < 0.3:
                self.spawn_particle()

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled:
            return

        for p in self.particles:
            life_factor = 1 - (p.life / p.max_life)
            color = get_psychedelic_color(offset=p.color_offset + state.hue_offset)
            color = tuple(int(c * life_factor) for c in color)
            size = int(p.size * life_factor)

            if size > 0:
                try:
                    pygame.draw.circle(surface, color, (int(p.x), int(p.y)), size)
                except (ValueError, TypeError):
                    pass


class FlickerEffect(Effect):
    """Background flicker and texture noise."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.flicker_state = 0
        self.colors = [(255, 255, 255), (0, 0, 0)]

    def update(self, state: State) -> None:
        self.flicker_state += 1

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled:
            return

        if self.config.high_contrast_mode:
            bg_color = self.colors[self.flicker_state % 2]
            surface.fill(bg_color)

        noise_count = int(100 * self.config.chaos * self.config.intensity)
        for _ in range(noise_count):
            x = random.randint(0, self.config.width)
            y = random.randint(0, self.config.height)
            size = random.randint(1, int(5 + 10 * self.config.intensity))
            color = get_glitch_color()

            if random.random() < 0.5:
                pygame.draw.rect(surface, color, (x, y, size, size))
            else:
                pygame.draw.circle(surface, color, (x, y), size)


class DistortionEffect(Effect):
    """Screen-level distortion effects."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.jitter_offset = (0, 0)

    def update(self, state: State) -> None:
        if random.random() < 0.3 * self.config.chaos:
            self.jitter_offset = (random.randint(-10, 10), random.randint(-10, 10))
        else:
            self.jitter_offset = (0, 0)

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled:
            return

        if self.jitter_offset != (0, 0) and self.config.intensity > 0.3:
            temp = surface.copy()
            surface.blit(temp, self.jitter_offset)

        if random.random() < 0.1 * self.config.chaos * self.config.intensity:
            self._apply_radial_blur(surface)

    def _apply_radial_blur(self, surface: pygame.Surface) -> None:
        try:
            temp = surface.copy()
            for i in range(1, 3):
                scale = 1.0 + i * 0.02
                scaled_size = (
                    int(self.config.width * scale),
                    int(self.config.height * scale),
                )
                scaled = pygame.transform.scale(temp, scaled_size)

                x_offset = (self.config.width - scaled_size[0]) // 2
                y_offset = (self.config.height - scaled_size[1]) // 2

                scaled.set_alpha(50)
                surface.blit(scaled, (x_offset, y_offset))
        except pygame.error:
            pass


class ChromaticAberrationEffect(Effect):
    """RGB channel separation creating color fringing at edges."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.offset = 3
        self.phase = 0.0

    def update(self, state: State) -> None:
        self.phase += state.delta_time
        base_offset = int(2 + 4 * self.config.intensity)
        jitter = (
            int(random.uniform(-2, 2) * self.config.chaos)
            if self.config.chaos > 0.3
            else 0
        )
        self.offset = base_offset + jitter

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled or self.offset < 1:
            return

        try:
            arr = pygame.surfarray.pixels3d(surface)

            offset = self.offset
            h, w = arr.shape[0], arr.shape[1]

            red_shifted = np.zeros_like(arr[:, :, 0])
            blue_shifted = np.zeros_like(arr[:, :, 2])

            red_shifted[offset:, :] = arr[:-offset, :, 0]
            blue_shifted[:-offset, :] = arr[offset:, :, 2]

            arr[:, :, 0] = (arr[:, :, 0].astype(np.uint16) + red_shifted) // 2
            arr[:, :, 2] = (arr[:, :, 2].astype(np.uint16) + blue_shifted) // 2

            del arr
        except (pygame.error, ValueError):
            pass


class ScanlinesEffect(Effect):
    """CRT-style horizontal scanlines."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.scanline_surface: pygame.Surface | None = None
        self.line_spacing = 3
        self.scroll_offset = 0.0

    def update(self, state: State) -> None:
        self.scroll_offset += state.delta_time * 30 * self.config.intensity
        if self.scroll_offset > self.line_spacing:
            self.scroll_offset -= self.line_spacing

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled:
            return

        if (
            self.scanline_surface is None
            or self.scanline_surface.get_size() != surface.get_size()
        ):
            self._create_scanline_surface(surface.get_size())

        if self.scanline_surface:
            offset_y = int(self.scroll_offset) if self.config.chaos > 0.2 else 0
            surface.blit(
                self.scanline_surface, (0, offset_y), special_flags=pygame.BLEND_MULT
            )

    def _create_scanline_surface(self, size: tuple[int, int]) -> None:
        self.scanline_surface = pygame.Surface(size, pygame.SRCALPHA)
        self.scanline_surface.fill((255, 255, 255, 255))

        darkness = int(180 + 75 * (1 - self.config.intensity))

        for y in range(0, size[1], self.line_spacing):
            pygame.draw.line(
                self.scanline_surface,
                (darkness, darkness, darkness, 255),
                (0, y),
                (size[0], y),
                1,
            )


class CRTEffect(Effect):
    """CRT monitor simulation with vignette, curvature, and phosphor glow."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.vignette_surface: pygame.Surface | None = None
        self.glow_phase = 0.0

    def update(self, state: State) -> None:
        self.glow_phase += state.delta_time * 2

    def render(self, surface: pygame.Surface, state: State) -> None:
        if not self.enabled:
            return

        self._apply_vignette(surface)

        if self.config.intensity > 0.5:
            self._apply_phosphor_glow(surface, state)

    def _apply_vignette(self, surface: pygame.Surface) -> None:
        if (
            self.vignette_surface is None
            or self.vignette_surface.get_size() != surface.get_size()
        ):
            self._create_vignette_surface(surface.get_size())

        if self.vignette_surface:
            surface.blit(self.vignette_surface, (0, 0), special_flags=pygame.BLEND_MULT)

    def _create_vignette_surface(self, size: tuple[int, int]) -> None:
        self.vignette_surface = pygame.Surface(size, pygame.SRCALPHA)
        center_x, center_y = size[0] // 2, size[1] // 2
        max_dist = math.sqrt(center_x**2 + center_y**2)

        for y in range(0, size[1], 4):
            for x in range(0, size[0], 4):
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                factor = 1.0 - (dist / max_dist) ** 1.5
                factor = max(0.3, min(1.0, factor))
                brightness = int(255 * factor)
                pygame.draw.rect(
                    self.vignette_surface,
                    (brightness, brightness, brightness, 255),
                    (x, y, 4, 4),
                )

    def _apply_phosphor_glow(self, surface: pygame.Surface, state: State) -> None:
        glow_intensity = int(
            20 + 10 * math.sin(self.glow_phase) * self.config.intensity
        )

        try:
            arr = pygame.surfarray.pixels3d(surface)
            green_boost = np.clip(
                arr[:, :, 1].astype(np.int16) + glow_intensity, 0, 255
            )
            arr[:, :, 1] = green_boost.astype(np.uint8)
            del arr
        except (pygame.error, ValueError):
            pass


class EffectManager:
    """Manages and orchestrates multiple effects."""

    def __init__(self, config: Config):
        self.config = config
        self.effects: list[Effect] = []
        self._initialize_effects()

    def _initialize_effects(self) -> None:
        if self.config.enable_flicker:
            self.effects.append(FlickerEffect(self.config))
        if self.config.enable_waves:
            self.effects.append(WaveEffect(self.config))
        if self.config.enable_spirals:
            self.effects.append(SpiralEffect(self.config))
        if self.config.enable_tesseract:
            self.effects.append(TesseractEffect(self.config))
        if self.config.enable_fractals:
            self.effects.append(FractalEffect(self.config))
        if self.config.enable_particles:
            self.effects.append(ParticleEffect(self.config))
        if self.config.enable_distortion:
            self.effects.append(DistortionEffect(self.config))
        if self.config.enable_scanlines:
            self.effects.append(ScanlinesEffect(self.config))
        if self.config.enable_chromatic:
            self.effects.append(ChromaticAberrationEffect(self.config))
        if self.config.enable_crt:
            self.effects.append(CRTEffect(self.config))

    def update(self, state: State) -> None:
        for effect in self.effects:
            if effect.enabled:
                effect.update(state)

    def render(self, surface: pygame.Surface, state: State) -> None:
        for effect in self.effects:
            if effect.enabled:
                effect.render(surface, state)

    def toggle_effect(self, effect_type: type) -> bool:
        for effect in self.effects:
            if isinstance(effect, effect_type):
                effect.toggle()
                return effect.enabled
        return False

    def get_effect(self, effect_type: type) -> Effect | None:
        for effect in self.effects:
            if isinstance(effect, effect_type):
                return effect
        return None


class UIOverlay:
    """Heads-up display for status and controls."""

    def __init__(self, config: Config):
        self.config = config
        self.font: pygame.font.Font | None = None
        self.title_font: pygame.font.Font | None = None
        self.show_help = False

    def initialize(self) -> None:
        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 16)
        self.title_font = pygame.font.SysFont("monospace", 24, bold=True)

    def render(self, surface: pygame.Surface, state: State, fps: float) -> None:
        if self.font is None or self.title_font is None:
            return

        fps_text = f"FPS: {fps:.1f} | Frame: {state.frame}"
        fps_surface = self.font.render(fps_text, True, (255, 255, 255))

        padding = 5
        bg_rect = pygame.Rect(
            self.config.width - fps_surface.get_width() - padding * 2 - 10,
            5,
            fps_surface.get_width() + padding * 2,
            fps_surface.get_height() + padding * 2,
        )
        pygame.draw.rect(surface, (0, 0, 0, 128), bg_rect)
        surface.blit(
            fps_surface,
            (self.config.width - fps_surface.get_width() - padding - 10, padding + 5),
        )

        intensity_text = (
            f"Intensity: {self.config.intensity:.0%} | Chaos: {self.config.chaos:.0%}"
        )
        intensity_surface = self.font.render(intensity_text, True, (255, 255, 255))
        bg_rect = pygame.Rect(
            5,
            5,
            intensity_surface.get_width() + 10,
            intensity_surface.get_height() + 10,
        )
        pygame.draw.rect(surface, (0, 0, 0, 128), bg_rect)
        surface.blit(intensity_surface, (10, 10))

        if state.message and state.message_timer > 0:
            msg_surface = self.title_font.render(state.message, True, (255, 255, 0))
            x = (self.config.width - msg_surface.get_width()) // 2
            bg_rect = pygame.Rect(
                x - 10, 40, msg_surface.get_width() + 20, msg_surface.get_height() + 10
            )
            pygame.draw.rect(surface, (0, 0, 0, 180), bg_rect)
            surface.blit(msg_surface, (x, 45))

        if self.show_help:
            self._render_help(surface)

    def _render_help(self, surface: pygame.Surface) -> None:
        if self.font is None:
            return

        help_lines = [
            "=== BRAIN FUZZER CONTROLS ===",
            "",
            "[+/-]     Intensity up/down",
            "[C/V]     Chaos up/down",
            "[UP/DOWN] Rotation speed",
            "[SPACE]   Toggle high contrast",
            "[M]       Cycle color mode",
            "[1-0]     Toggle effects",
            "[P]       Pause",
            "[R]       Randomize",
            "[S]       Save preset",
            "[H]       Toggle this help",
            "[ESC/Q]   Quit",
            "",
            "Effects: 1=Flicker 2=Waves 3=Spirals",
            "         4=Tesseract 5=Fractals 6=Particles",
            "         7=Distortion 8=Scanlines",
            "         9=Chromatic 0=CRT",
        ]

        overlay = pygame.Surface((400, 400), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))

        x = (self.config.width - 400) // 2
        y = (self.config.height - 400) // 2
        surface.blit(overlay, (x, y))

        for i, line in enumerate(help_lines):
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            text_surface = self.font.render(line, True, color)
            surface.blit(text_surface, (x + 20, y + 20 + i * 22))


def run_startup_sequence(screen: pygame.Surface, config: Config) -> None:
    clock = pygame.time.Clock()

    pygame.font.init()
    title_font = pygame.font.SysFont("monospace", 36, bold=True)
    subtitle_font = pygame.font.SysFont("monospace", 18)

    frames = 120

    for frame in range(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN):
                    return

        progress = frame / frames

        bg_value = int(255 * (1 - progress))
        screen.fill((bg_value, bg_value, bg_value))

        num_sides = 6
        center_x, center_y = config.width // 2, config.height // 2
        scale = 150 + 50 * math.sin(frame * 0.1)
        rotation = frame * 0.05

        points = [
            (
                center_x + scale * math.cos(rotation + i * 2 * math.pi / num_sides),
                center_y + scale * math.sin(rotation + i * 2 * math.pi / num_sides),
            )
            for i in range(num_sides)
        ]

        color = hsv_to_rgb(progress, 1.0, 1.0)
        pygame.draw.polygon(screen, color, points, 4)

        inner_scale = scale * 0.6
        inner_rotation = -rotation * 1.5
        inner_points = [
            (
                center_x
                + inner_scale * math.cos(inner_rotation + i * 2 * math.pi / num_sides),
                center_y
                + inner_scale * math.sin(inner_rotation + i * 2 * math.pi / num_sides),
            )
            for i in range(num_sides)
        ]
        pygame.draw.polygon(
            screen, (255 - color[0], 255 - color[1], 255 - color[2]), inner_points, 3
        )

        title_color = (
            255,
            int(50 + 200 * (1 - progress)),
            int(50 + 200 * (1 - progress)),
        )
        title = title_font.render("BRAIN FUZZER", True, title_color)
        title_x = (config.width - title.get_width()) // 2
        screen.blit(title, (title_x, center_y - 180))

        subtitle = subtitle_font.render(
            "Visual Perception Research Tool", True, (200, 200, 200)
        )
        subtitle_x = (config.width - subtitle.get_width()) // 2
        screen.blit(subtitle, (subtitle_x, center_y - 140))

        if progress > 0.5:
            warning_alpha = int(255 * (progress - 0.5) * 2)
            warning = subtitle_font.render(
                "WARNING: May cause disorientation",
                True,
                (255, warning_alpha, warning_alpha),
            )
            warning_x = (config.width - warning.get_width()) // 2
            screen.blit(warning, (warning_x, center_y + 160))

            instruction = subtitle_font.render(
                "Press H for help | ESC to quit", True, (150, 150, 150)
            )
            instruction_x = (config.width - instruction.get_width()) // 2
            screen.blit(instruction, (instruction_x, center_y + 190))

        pygame.display.flip()
        clock.tick(60)


def handle_input(
    event: pygame.event.Event,
    config: Config,
    state: State,
    effect_manager: EffectManager,
    ui: UIOverlay,
) -> None:
    if event.type != pygame.KEYDOWN:
        return

    key = event.key

    if key in (pygame.K_ESCAPE, pygame.K_q):
        state.running = False
        return

    if key == pygame.K_p:
        state.paused = not state.paused
        state.message = "PAUSED" if state.paused else "RESUMED"
        state.message_timer = 1.5
        return

    if key == pygame.K_h:
        ui.show_help = not ui.show_help
        return

    if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        config.intensity = min(1.0, config.intensity + 0.1)
        state.message = f"Intensity: {config.intensity:.0%}"
        state.message_timer = 1.0
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        config.intensity = max(0.1, config.intensity - 0.1)
        state.message = f"Intensity: {config.intensity:.0%}"
        state.message_timer = 1.0
    elif key == pygame.K_c:
        config.chaos = min(1.0, config.chaos + 0.1)
        state.message = f"Chaos: {config.chaos:.0%}"
        state.message_timer = 1.0
    elif key == pygame.K_v:
        config.chaos = max(0.0, config.chaos - 0.1)
        state.message = f"Chaos: {config.chaos:.0%}"
        state.message_timer = 1.0
    elif key == pygame.K_UP:
        state.rotation_speed_xy *= 1.2
        state.rotation_speed_zw *= 1.2
        state.message = "Rotation +20%"
        state.message_timer = 1.0
    elif key == pygame.K_DOWN:
        state.rotation_speed_xy *= 0.8
        state.rotation_speed_zw *= 0.8
        state.message = "Rotation -20%"
        state.message_timer = 1.0
    elif key == pygame.K_SPACE:
        config.high_contrast_mode = not config.high_contrast_mode
        state.message = f"High Contrast: {'ON' if config.high_contrast_mode else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_r:
        config.intensity = random.uniform(0.3, 1.0)
        config.chaos = random.uniform(0.2, 0.8)
        state.rotation_speed_xy = random.uniform(0.02, 0.1)
        state.rotation_speed_zw = random.uniform(0.02, 0.1)
        state.message = "RANDOMIZED"
        state.message_timer = 1.5
    elif key == pygame.K_1:
        enabled = effect_manager.toggle_effect(FlickerEffect)
        state.message = f"Flicker: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_2:
        enabled = effect_manager.toggle_effect(WaveEffect)
        state.message = f"Waves: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_3:
        enabled = effect_manager.toggle_effect(SpiralEffect)
        state.message = f"Spirals: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_4:
        enabled = effect_manager.toggle_effect(TesseractEffect)
        state.message = f"Tesseract: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_5:
        enabled = effect_manager.toggle_effect(FractalEffect)
        state.message = f"Fractals: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_6:
        enabled = effect_manager.toggle_effect(ParticleEffect)
        state.message = f"Particles: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_7:
        enabled = effect_manager.toggle_effect(DistortionEffect)
        state.message = f"Distortion: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_8:
        enabled = effect_manager.toggle_effect(ScanlinesEffect)
        state.message = f"Scanlines: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_9:
        enabled = effect_manager.toggle_effect(ChromaticAberrationEffect)
        state.message = f"Chromatic: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_0:
        enabled = effect_manager.toggle_effect(CRTEffect)
        state.message = f"CRT: {'ON' if enabled else 'OFF'}"
        state.message_timer = 1.0
    elif key == pygame.K_m:
        modes = ["psychedelic", "fire", "ice", "matrix", "monochrome"]
        current_idx = (
            modes.index(config.color_mode) if config.color_mode in modes else 0
        )
        config.color_mode = modes[(current_idx + 1) % len(modes)]
        set_color_mode(config.color_mode)
        state.message = f"Color: {config.color_mode.upper()}"
        state.message_timer = 1.0
    elif key == pygame.K_s:
        preset_path = f"preset_{int(time.time())}.json"
        config.save_preset(preset_path)
        state.message = f"Saved: {preset_path}"
        state.message_timer = 2.0


def main() -> None:
    args = parse_args()

    if args.list_presets:
        print("Built-in presets:")
        for name, settings in BUILTIN_PRESETS.items():
            print(f"  {name}: {settings}")
        return

    config = config_from_args(args)
    state = State()

    if args.save_preset:
        config.save_preset(args.save_preset)
        print(f"Preset saved to: {args.save_preset}")
        return

    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("fuzzer_errors.log")],
    )
    logger = logging.getLogger(__name__)

    def signal_handler(sig, frame):
        logger.info("Interrupt received, shutting down...")
        state.running = False

    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Initializing Brain Fuzzer...")
    logger.info(
        f"Config: intensity={config.intensity}, chaos={config.chaos}, color_mode={config.color_mode}"
    )

    set_color_mode(config.color_mode)
    pygame.init()

    flags = pygame.DOUBLEBUF
    if config.fullscreen:
        flags |= pygame.FULLSCREEN

    screen = pygame.display.set_mode((config.width, config.height), flags)
    pygame.display.set_caption("Brain Fuzzer: Visual Perception Research Tool")

    clock = pygame.time.Clock()

    ui = UIOverlay(config)
    ui.initialize()

    if not args.skip_intro:
        run_startup_sequence(screen, config)

    effect_manager = EffectManager(config)

    start_time = time.time()
    last_time = start_time

    logger.info("Entering main loop")

    try:
        while state.running:
            current_time = time.time()
            state.delta_time = current_time - last_time
            last_time = current_time
            state.time_elapsed = current_time - start_time

            if (
                config.session_duration > 0
                and state.time_elapsed >= config.session_duration
            ):
                logger.info("Session duration reached")
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    state.running = False
                else:
                    handle_input(event, config, state, effect_manager, ui)

            if not state.running:
                break

            if state.message_timer > 0:
                state.message_timer -= state.delta_time

            if state.paused:
                ui.render(screen, state, clock.get_fps())
                pygame.display.flip()
                clock.tick(30)
                continue

            state.frame += 1
            state.rotation_xy += state.rotation_speed_xy * (
                1 + random.uniform(-0.1, 0.1) * config.chaos
            )
            state.rotation_zw += state.rotation_speed_zw * (
                1 + random.uniform(-0.1, 0.1) * config.chaos
            )
            state.hue_offset += config.color_cycle_speed

            if not config.high_contrast_mode:
                bg_hue = (state.hue_offset * 0.1) % 1.0
                bg_color = hsv_to_rgb(bg_hue, 0.3, 0.1)
                screen.fill(bg_color)

            effect_manager.update(state)
            effect_manager.render(screen, state)
            ui.render(screen, state, clock.get_fps())

            pygame.display.flip()
            clock.tick(config.target_fps)

            if state.frame % 500 == 0:
                logger.debug(
                    f"Frame: {state.frame}, FPS: {clock.get_fps():.1f}, Elapsed: {state.time_elapsed:.1f}s"
                )

    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        raise

    finally:
        pygame.quit()
        logger.info(
            f"Session ended after {state.time_elapsed:.1f}s, {state.frame} frames"
        )
        print(f"\nSession lasted {state.time_elapsed:.1f} seconds")
        print(
            "Fuzzer terminated. Observe aftereffects: disorientation, afterimages, visual persistence."
        )


if __name__ == "__main__":
    main()
