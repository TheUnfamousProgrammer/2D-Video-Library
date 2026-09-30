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

## Melody Hop

A second generator plays a public-domain melody as a ball hopping between pads. The bounce pipeline is unchanged. Hop has its own config and CLI.

```bash
python make_hop.py --config configs/hop_default.yaml --song songs/ode_to_joy.yaml --out out/hop_ode.mp4
python make_hop.py --config configs/hop_default.yaml --song songs/ode_to_joy.yaml --out out/hop_preview.mp4 --preview
python make_hop.py --config configs/hop_default.yaml --song songs/ode_to_joy.yaml --out out/hop_batch.mp4 --batch 3 --vary palette,hook
python -m fc_sat.verify out/hop_ode.mp4 --config configs/hop_default.yaml
```

`--preview` is 540x960 at 30 fps with no audio. A full render verifies unless it is a preview. `--verify` forces the check. Batch names are `out/{stem}_{song}_{palette}_{hookidx}.mp4`, plus `_octN` and `_bpmN` when those fields vary. A variant tuple is never repeated. One failure is logged and the batch continues; the process exits 1 if any variant failed.

Each output gets `{out}.json` and `{out}.post.txt`. The post file is the hook plus `#shorts`, the tags `#shorts #guessthesong #satisfying #music #asmr`, and a pinned comment `Answer: {title} ({composer})`.

### Hop config

`configs/hop_default.yaml`, with palettes from `configs/palettes.yaml` and lines from `configs/hop_hooks.yaml`.

| key | default | role |
| --- | --- | --- |
| `seed` | 1 | reverb noise and spark directions |
| `song` | `songs/ode_to_joy.yaml` | melody file |
| `octave_shift` | 1 | synthesis only. Pad layout stays on the written pitches |
| `bpm` | null | overrides the song tempo. Null keeps the file's bpm |
| `repeats` | 1 | 1 to 3 copies of the loop, tiled |
| `palette` | sunset | pad colors, sampled across the palette |
| `hook` | Guess the song | top line. Alpha is 1 until 2.6 s, out by 3.0 s, and back in over the last 0.6 s of every loop |
| `show_note_names` | false | pitch names under the pads |
| `workers` | 1 | 1 renders in-process. More than 1 uses a spawn pool |
| `bloom_strength` | 0.6 | same bloom as the bounce film |
| `audio_offset_ms` | 0 | circular shift of the finished loop. Positive delays the sound |
| `layout.*` | span 700, pad width 112, top y 1180 | pad row. Pitches sit low to high, left to right |
| `hop.h_ref`, `h_min`, `h_max`, `exponent` | 220, 60, 600, 1.2 | arc height from the flight duration |
| `effects.trail_seconds` | 0.30 | analytic trail |
| `effects.sparks` | 10 | sparks per landing |
| `effects.ring` | true | shockwave |
| `audio.reverb_wet`, `audio.rt60` | 0.22, 1.4 | circular reverb |

### Adding a song

Create `songs/your_song.yaml` with `title`, `composer`, `bpm`, `beats_per_bar`, and `notes`. Notes are `NAME:BEATS` tokens. `|` is spacing. `R:BEATS` is a rest. A leading rest is dropped. Any other rest lengthens the previous flight, and a rest at the end lengthens the flight that wraps onto the first note.

```yaml
title: Ode to Joy
composer: Beethoven (public domain)
bpm: 120
beats_per_bar: 4
notes: "E4:1 E4:1 F4:1 G4:1 | ..."
```

Only public-domain melodies belong here. Play the render and check it by ear before you publish; the parser will not catch a wrong rhythm. The file needs 2 to 10 distinct pitches, positive beat values, and a loop of 60 seconds or less. Errors name the token index.

### Tempo snapping

The musical length is `T0 = beats * 60 / bpm`. The frame count is `N = round(T0 * 60)`, and the video length is `T = N / 60`. Onsets use `spb = T / beats`, so every landing is an exact frame and an exact audio sample (`frame * 800` at 48 kHz). Ode to Joy at 120 bpm is already 16.0 seconds, so the snap does not move it. A song that is not an exact number of frames is sped or slowed by a fraction of a beat to land on the grid.

### Circular audio

The kalimba notes and the reverb wrap inside one period of `N * 800` samples. There is no fade and no silent tail. Convolution is an FFT around the circle, then the mean is removed. True peak uses a periodic 4x FFT resample. The same five-pass loudness loop then aims at -14 LUFS and -1 dBTP. `repeats` tiles that finished period.

### Sync check

`python -m fc_sat.verify` stays on the bounce film unless you pass `--mode hop` or the config contains a `song` field. Hop checks the container (1080x1920, 60 fps, H.264, yuv420p, AAC 48 kHz stereo, duration `N/60 * repeats`, under 100 MB), loudness within 1 LU of -14, FFT true peak at or under -1 dBTP, and the safe-zone pixels against the vignette.

The frame before a landing reuses that landing's glow and squash, so the loop point is not a cut through the brightest moment of the hit. The pad flash and the sparks still begin on the onset frame.

The audio seam compares the first and last sample of each channel with the 99th percentile of consecutive sample steps. The picture seam compares the first and last frame with the 99th percentile of consecutive-frame changes. Note sync searches +/-40 ms around each onset. The audio onset is the peak of the positive first difference of a smoothed dB envelope (2 ms hop, about 10 ms of smoothing). The picture onset is the frame with the largest luma increase inside that note's pad. It passes when the absolute median offset is within 20 ms and the 90th percentile of the absolute offsets is within 34 ms. A failure prints every note's offset so the detector can be fixed before either threshold moves.
