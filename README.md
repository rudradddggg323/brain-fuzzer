# Brain Fuzzer

A psychedelic visual effects generator designed to explore the boundaries of human visual perception through rapidly changing geometric patterns, color cycling, and perceptual illusions.

**WARNING: This program produces intense visual stimulation that may cause disorientation, afterimages, or discomfort. Not recommended for individuals with photosensitive epilepsy or similar conditions.**

<img width="811" height="836" alt="image" src="https://github.com/user-attachments/assets/6fd85252-2ec9-4e17-83f8-ea6d1ede2279" />

## Features

- **10 Layered Visual Effects**: Tesseract (4D hypercube), fractals, spirals, waves, particles, flicker, distortion, scanlines, chromatic aberration, and CRT simulation
- **5 Color Modes**: Psychedelic, fire, ice, matrix, and monochrome palettes
- **Real-time Controls**: Adjust intensity, chaos, rotation speed, and effect toggles on the fly
- **Preset System**: Save and load custom configurations
- **Built-in Presets**: Quick-start profiles (mellow, intense, matrix, minimal, chaos)
- **Session Management**: Set duration limits or run indefinitely

## Requirements

- Python 3.10 or higher
- pygame 2.x
- numpy 2.x

## Installation

```bash
# Clone the repository
git clone https://github.com/geeknik/brain-fuzzer.git
cd brain-fuzzer

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pygame numpy

# Run the fuzzer
python main.py
```

## Usage

### Basic Usage

```bash
# Run with defaults (800x800 window, 2 minutes)
python main.py

# Fullscreen mode with high intensity
python main.py --fullscreen --intensity 0.9

# 60-second session with maximum chaos
python main.py --duration 60 --chaos 0.8

# Custom resolution
python main.py --width 1920 --height 1080
```

### Preset System

```bash
# List built-in presets
python main.py --list-presets

# Load a preset
python main.py --preset mellow

# Save current config and exit
python main.py --width 1920 --height 1080 --save-preset my_config.json

# Load custom preset
python main.py --preset my_config.json
```

### Effect Control

```bash
# Disable specific effects
python main.py --no-flicker --no-distortion

# Matrix-style visuals only
python main.py --color-mode matrix --no-flicker

# Minimal configuration
python main.py --intensity 0.5 --chaos 0.1 --no-fractals --no-particles
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `+` / `-` | Increase/decrease intensity |
| `C` / `V` | Increase/decrease chaos level |
| `↑` / `↓` | Increase/decrease rotation speed |
| `SPACE` | Toggle high contrast flicker mode |
| `M` | Cycle through color modes |
| `1` | Toggle flicker effect |
| `2` | Toggle wave effect |
| `3` | Toggle spiral effect |
| `4` | Toggle tesseract effect |
| `5` | Toggle fractal effect |
| `6` | Toggle particle effect |
| `7` | Toggle distortion effect |
| `8` | Toggle scanline effect |
| `9` | Toggle chromatic aberration |
| `0` | Toggle CRT effect |
| `P` | Pause/resume |
| `R` | Randomize all parameters |
| `S` | Save current config to timestamped file |
| `H` | Show/hide help overlay |
| `ESC` / `Q` | Quit |

## Visual Effects

### Geometric Effects

- **Tesseract**: Rotating 4D hypercube projection with motion trails
- **Fractals**: Recursive geometric patterns with random branching
- **Spirals**: Hypnotic multi-armed spiral patterns
- **Waves**: Oscillating sine wave distortions

### Particle Systems

- **Particles**: Dynamic particle system with physics simulation

### Post-Processing

- **Flicker**: Background noise and high-contrast mode
- **Distortion**: Screen jitter and radial blur
- **Chromatic Aberration**: RGB channel separation for color fringing
- **Scanlines**: CRT-style horizontal scanlines with scrolling
- **CRT**: Vignette, phosphor glow, and monitor simulation

## Color Modes

- **Psychedelic**: Full spectrum HSV color cycling
- **Fire**: Red-orange-yellow gradient with heat effects
- **Ice**: Blue-cyan palette with cool tones
- **Matrix**: Green phosphor terminal aesthetic
- **Monochrome**: Grayscale intensity variations

## Built-in Presets

- **mellow**: Low intensity (0.4), minimal chaos (0.2), ice colors
- **intense**: Maximum intensity (1.0), high chaos (0.8), fire colors
- **matrix**: Green terminal aesthetic, moderate intensity (0.6), flicker disabled
- **minimal**: Simple effects only, no fractals or particles
- **chaos**: Maximum everything with high contrast flicker

## CLI Arguments

### Display Options

- `-W`, `--width`: Window width (default: 800)
- `-H`, `--height`: Window height (default: 800)
- `-f`, `--fullscreen`: Fullscreen mode
- `--fps`: Target frame rate (default: 60)

### Session Options

- `-d`, `--duration`: Session duration in seconds (0 = infinite, default: 120)
- `-i`, `--intensity`: Effect intensity 0.1-1.0 (default: 0.7)
- `-c`, `--chaos`: Chaos level 0.0-1.0 (default: 0.5)
- `--color-mode`: Color palette (psychedelic|fire|ice|matrix|monochrome)
- `--high-contrast`: Enable high contrast flicker mode

### Effect Toggles

- `--no-tesseract`: Disable tesseract effect
- `--no-fractals`: Disable fractal effect
- `--no-spirals`: Disable spiral effect
- `--no-waves`: Disable wave effect
- `--no-particles`: Disable particle effect
- `--no-flicker`: Disable flicker effect
- `--no-distortion`: Disable distortion effect
- `--no-chromatic`: Disable chromatic aberration
- `--no-scanlines`: Disable scanline effect
- `--no-crt`: Disable CRT effect

### Preset Management

- `-p`, `--preset`: Load preset from JSON file
- `--save-preset`: Save current config to JSON and exit
- `--list-presets`: List built-in presets and exit

### Debug Options

- `-v`, `--verbose`: Enable verbose logging
- `-q`, `--quiet`: Quiet mode (errors only)
- `--skip-intro`: Skip startup animation

## Examples

```bash
# Gentle introduction
python main.py --preset mellow --duration 30

# Full sensory overload
python main.py --fullscreen --preset chaos

# Custom matrix experience
python main.py --color-mode matrix --intensity 0.8 --no-flicker --fps 120

# Performance testing
python main.py --fps 144 --verbose --duration 60

# Create and use custom preset
python main.py --intensity 0.6 --chaos 0.4 --color-mode ice --save-preset ice_calm.json
python main.py --preset ice_calm.json --fullscreen
```

## Technical Details

- Built with pygame for cross-platform graphics
- NumPy for efficient array operations (chromatic aberration, CRT effects)
- 4D rotation matrices for tesseract projection
- Recursive fractal generation with depth limiting
- Real-time particle physics simulation
- Multi-layered rendering pipeline

## License

[MIT License](LICENSE)

## Safety Notice

This software is intended for visual perception research and artistic exploration. Users should:

- Take breaks every 15-20 minutes
- Avoid use if you have photosensitive epilepsy
- Stop immediately if you experience discomfort, nausea, or disorientation
- Observe aftereffects (afterimages, visual persistence) as part of the experience
- Use responsibly and at your own risk
