# fc-sat

Offline renderer for a vertical "oddly satisfying" YouTube Short: balls bounce inside a ring, every wall hit can spawn another ball, the count and the pitch rise together, then the shot resets so the last frame matches the first.

The picture is rendered frame by frame and piped to ffmpeg. Nothing is screen-recorded. The same config and seed produce the same simulation and the same audio.

## Setup

Python 3.11 or newer. This project is tested with the Python 3.12 venv in `.venv`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ffmpeg must be on `PATH` and must include `libx264` and the native `aac` encoder. If ffmpeg is missing, the tool tries the binary shipped with `imageio-ffmpeg`. On macOS, Homebrew ffmpeg is enough:

```bash
brew install ffmpeg
ffmpeg -encoders | grep -E 'libx264|aac'
```

The hook font is the OFL Montserrat ExtraBold file in `assets/`. If that file is absent, the renderer looks for Arial Bold (Windows and macOS) or DejaVu Sans Bold (Linux) and exits with an install message if none of them exist.

## Render

```bash
python make.py --config configs/default.yaml --out out/a.mp4
python make.py --config configs/default.yaml --out out/preview.mp4 --preview
python make.py --config configs/default.yaml --out out/batch.mp4 --batch 4 --vary seed,palette,hook
python -m fc_sat.verify out/a.mp4 --config configs/default.yaml
```

`--preview` writes 540x960 at 30 fps with no audio, using the same simulation. Verification runs after a full render and stays off for preview unless you pass `--verify`.

Other flags:

- `--seed N` overrides the config seed
- `--workers N` sets render processes (default is CPU count minus one; `0` in the config means the same)
- `--batch N` writes N files named `out/{stem}_{seed}_{palette}_{hookidx}.mp4`
- `--vary seed,palette,hook` chooses which of those three change. A combination is never reused. One failed variant is logged and the batch continues. The process exits non-zero if any variant failed.
- `--no-cache` ignores `.cache/sim_*.npz`
- `--keep-temp` keeps the float WAV next to the mp4

Each output gets a sidecar JSON (`out/a.json`) with the resolved config and measured stats, and `out/a.post.txt` with the hook as a title plus `#shorts #oddlysatisfying #satisfying #asmr`.

The mp4 is written to a temporary `*.partial.mp4` and renamed into place only after ffmpeg exits 0.

## Pipeline

1. **Sim.** Fixed step of 1/240 s, four substeps per 60 fps frame. Cached by a hash of the sim fields, so a palette or hook change does not re-simulate.
2. **Audio.** Built only from the event log and the timeline, at 48 kHz stereo.
3. **Render.** Each frame is a function of the sim state, the frame index, and the config. Workers render chunks in order and the main process pipes BGR frames to ffmpeg.

Image y grows downward, so gravity is +y.

## Timeline

`growth_seconds` (Tg, default 19) plus a fixed tail:

| phase | duration | what happens |
| --- | --- | --- |
| growth | Tg | count follows the throttle from 1 ball to the cap |
| hold | 0.5 s | balls freeze, one low thud, no mallet notes |
| implode | 1.2 s | balls ease to the center and shrink, descending sweep |
| beat | 0.3 s | empty ring, one gentle pulse |
| reset | 0.5 s | the opening ball and the hook fade back in |

Default length is 21.5 s (1290 frames at 60 fps). The total is not allowed to exceed 30 s.

The last frame uses the same opening pose as frame 0: hook at full opacity, counter `BALLS: 1`, ball at its start position, ring glow at idle.

## Config

`configs/default.yaml` is the source of truth. `configs/palettes.yaml` and `configs/hooks.yaml` sit beside it.

| key | default | role |
| --- | --- | --- |
| `seed` | 7 | sim spawn jitter and audio wobble |
| `canvas.width`, `canvas.height` | 1080, 1920 | full-render size |
| `fps` | 60 | full-render frame rate |
| `growth_seconds` | 19 | Tg |
| `hold_seconds`, `implode_seconds`, `beat_seconds`, `reset_seconds` | 0.5, 1.2, 0.3, 0.5 | tail |
| `ring.center` | [540, 880] | ring center in pixels |
| `ring.radius` | 380 | inner collision radius |
| `ring.thickness` | 6 | stroke is centered on the radius, so the outer edge is 383 |
| `gravity` | 900 | px/s^2, downward. 0 is allowed |
| `restitution` | 1 | wall bounce |
| `radius_base`, `radius_exp` | 28, -0.28 | `radius = clamp(base * count^exp, radius_min, radius_max)` |
| `radius_min` | 8 | radius floor. Raise this if the cap frame looks too sparse |
| `radius_max` | 28 | radius ceiling |
| `cap` | 1000 | maximum balls |
| `spawn_cooldown` | 0.15 | seconds before a ball can spawn again |
| `spawn_jitter_deg` | 25 | child velocity is the parent's reflected velocity rotated by a uniform angle in this range |
| `throttle` | see yaml | log-linear keyframes as fractions of Tg. The last keyframe is `(0.985, cap)`; after that the ceiling stays at the cap until Tg |
| `min_bounce_speed` | 450 | inward speed floor after a wall hit, px/s |
| `max_speed` | 1600 | speed clamp, px/s |
| `palette` | sunset | name in `palettes.yaml` |
| `hook` | Every bounce adds a ball | top line, also the suggested title |
| `show_wait_text`, `wait_text` | false, "Wait for it..." | optional mid line at 0.55 Tg for 1.5 s |
| `show_cta`, `cta_text` | false, Subscribe | optional line during the beat only, so the loop seam stays clean |
| `collisions` | false | soft ball-ball push. Turns itself off if a substep exceeds 8 ms |
| `sound_preset` | mallet | only preset |
| `audio_offset_ms` | 0 | shifts hit sounds relative to the picture. Positive delays the sound |
| `bloom_strength` | 0.6 | added bloom |
| `workers` | 0 | 0 means CPU count minus one |
| `tune.max_attempts` | 60 | pacing retries |
| `tune.speed_range` | [700, 1500] | initial speed search |
| `tune.gravity_range` | [200, 1400] | gravity search |
| `tune.seed_offset_range` | [0, 59] | added to `seed` on later attempts |

## Pacing

After each attempt the sim checks:

- first wall hit in [0.30, 0.60] s
- count at `0.14 * Tg` is 8 +/- 2
- the cap is first reached inside `[0.97 * Tg, Tg]`
- no gap longer than 0.5 s between consecutive hits after the first hit

The open time before the first hit is not a gap. If all attempts fail, the process exits and names the check that failed.

Attempt 0 uses the configured gravity (clamped into range) and solves the starting inset so the opening flight is near 0.45 s. Later attempts walk speed, gravity, and seed offset in a fixed order. To make the early bounces easier to count, keep `spawn_cooldown` near 0.15 s and don't raise the early throttle keyframes. If the cap is missed, the usual cause is a ceiling that only reaches the cap at the very end of Tg; the default last keyframe at 0.985 leaves a short window where the cap is actually allowed.

## Palettes and hooks

Add a palette by appending 3 to 5 hex stops to `configs/palettes.yaml`, then set `palette:` to that name. Stops are interpolated in OKLab.

Add a hook by appending a string to `configs/hooks.yaml`. `--vary hook` cycles those lines. The `hook:` field in the main config is the line used for a single render.

## Audio sync

`audio_offset_ms` moves every hit, chime, and one-shot together. Positive values delay the sound relative to the picture. Render a preview to judge the picture, then a short full render (or the real one) and nudge the offset by 20–40 ms. The master chain fades 5 ms at each end and zeros the last 300 ms before it measures loudness, so the loop point stays quiet.

Loudness is a loop of at most five passes: measure integrated LUFS, gain toward -14, soft-limit, enforce a -1 dBTP true-peak ceiling on a 4x oversampled peak, measure again. It accepts when the result is within 0.5 LU of -14 and the true peak is at or under -1 dBTP.

## Safe zones

Text and the ring stroke stay inside x [130, 950] and y [200, 1536]. With the default ring (center x 540, radius 380, thickness 6) the stroke box is x 157 to 923 and y 497 to 1263. That leaves 27 px before the right-hand 12% band so bloom does not land in it. The renderer also never writes pixels in the bottom 20% or the right 12% after the vignette is in place.

`verify` checks those boxes with the same layout math, and compares the decoded bottom 20% and right 12% to the vignette at the same pixels. The tolerances are a mean absolute RGB error of 8/255 and a 99th-percentile luma delta of 0.06. Those numbers are not there to be loosened; if a render fails, the measured values and `*.safezone.png` are the thing to inspect.

## Tests

```bash
pytest
```

Covers determinism, balls staying inside the ring, the throttle and cooldown, event order, audio length and the loudness loop, the preview loop seam, config errors, unique batch names, pacing failures (including a gap that only exists before the first hit), the cap hold at 0.985 Tg, and a solid-color encode/decode through the BT.709 filter.
