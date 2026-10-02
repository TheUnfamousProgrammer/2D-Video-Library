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

## Pixel Morph

A third generator shows image A as a grid of square particles. They fly on arcs to form image B, hold, then fly back so the last frame is A again. Bounce and Melody Hop are unchanged.

```bash
python tools/fetch_starter_images.py
python make_morph.py --config configs/morph_default.yaml --a assets/images/mona_lisa.jpg --b assets/images/starry_night.jpg --out out/morph_preview.mp4 --preview --contact-sheet
python make_morph.py --config configs/morph_default.yaml --a assets/images/mona_lisa.jpg --b assets/images/starry_night.jpg --out out/morph_mona.mp4
python make_morph.py --config configs/morph_default.yaml --pairs configs/pairs.yaml --out out/morph_batch.mp4
python -m fc_sat.verify out/morph_mona.mp4 --config configs/morph_default.yaml
```

`--preview` is 540x960 at 30 fps with no audio. It samples the same phase function at `t = i/30`, and the last preview frame is time T so it matches frame 0. A full render verifies unless it is a preview. `--contact-sheet` writes `{out}.contact.png`: two rows (A to B, then B to A) and five columns at 0, 25, 50, 75, and 100 percent of that leg, each cell 270x480.

`--pairs` reads a YAML list. Each item is `a`, `b`, and optional `hook`, `focus_a`, `focus_b` (`[fx, fy]` in 0..1). Files are named `{stem_a}_{stem_b}_{hook_index}.mp4` in the directory of `--out`. A tuple is never repeated. One failure is logged and the batch continues; the process exits 1 if any pair failed.

Each output gets `{out}.json` (config, image sha256, assignment error, render seconds) and `{out}.post.txt` (hook plus `#shorts`, tags `#shorts #satisfying #pixels #art #morph`, the pinned comment "What did the picture turn into?", and credits when `assets/images/credits.json` knows the file).

### Morph config

`configs/morph_default.yaml`. Lines live in `configs/morph_hooks.yaml`.

| key | default | role |
| --- | --- | --- |
| `seed` | 7 | arc sign and tick choice |
| `cols`, `rows`, `cell` | 68, 90, 12 | grid. The box is 816x1080 at (132, 400) |
| `origin_x`, `origin_y` | 132, 400 | top-left of the image box. It must sit in x [130, 950], y [200, 1536] |
| `timeline.*` | 0.4, 6.0, 2.2, 5.5, 0.4 | hold A, A to B, hold B, B to A, hold A. 14.5 s, 870 frames |
| `delay_frac` | 0.30 | ripple. 0 is the center, 1 is the farthest cell |
| `arc_amp` | 0.15 | sideways arc as a fraction of the flight length |
| `lift` | 0.5 | in-flight squares grow by this times sin(pi E) |
| `spatial_weight` | 0.3 | travel penalty in the assignment. Lower matches color better and flies farther |
| `recolor_strength` | 1.0 | 0 keeps each particle's original color. 1 lands on the destination color |
| `hook` | Watch the pixels move | starts at 96 px and shrinks until it fits above the box |
| `max_particles` | 9000 | `cols * rows` above this asks for a smaller grid |
| `workers` | 0 | 0 means CPU count minus one. 1 stays in-process |
| `audio.whoosh_db` | -18 | whoosh peak under a nominal 0.5 tick, when every particle is flying |
| `audio.tick_density` | 12 | most arrivals kept in any 100 ms window |
| `audio_offset_ms` | 0 | shifts events before the master, from -500 to 500 |
| `max_delta_e` | 0.06 | OKLab fidelity limit, enforced when recolor_strength is at least 0.99 |
| `focus_a`, `focus_b` | [0.5, 0.5] | crop anchor. 0 pins to the left or top, 1 to the right or bottom |

### Adding images

You can point `--a` and `--b` at your own pictures. You are responsible for having the rights to them. The fetch script only keeps Wikimedia files whose `LicenseShortName` is public domain or CC0, writes `assets/images/credits.json`, and leaves the binaries untracked. Do not commit an image whose license you have not checked.

The fetch script asks Wikimedia for each file's `LicenseShortName` (via the Wikipedia API when `commons.wikimedia.org` does not answer) and downloads it through `Special:FilePath`, falling back to the canonical upload URL. It requests a 2000 px rendition, which is plenty for a 68-cell grid. The license check is still the original file's metadata.

Loading applies EXIF orientation. RGBA is composited onto `#07070B`. CMYK, palette, and grayscale become RGB. A picture smaller than the grid, an unreadable file, or A and B with the same sha256 is rejected. The same pixels after the crop are rejected too. The crop matches `cols:rows`, then resizes with area interpolation.

### Assignment and motion

Both grids are converted to OKLab. The cost is the squared OKLab distance plus `spatial_weight * (d / diag)^2`, where `d` is the distance between cell centers and `diag` is `hypot(cols, rows)`. The solver is `scipy.optimize.linear_sum_assignment` on a float64 matrix. If that matrix would pass about 1 GB (`n^2 * 8` bytes) the run stops before allocating it. The permutation is cached at `.cache/morph_{hash}.npz`. The hash covers the prepared pixels, the grid, and `spatial_weight`. Seed, hook, timing, and recolor do not bust it. The log prints the mean and 95th percentile OKLab error of the matched colors before recoloring, and the solve time.

Delay is `delay_frac` times the rank of each start cell's distance from the center (`rank / (n - 1)`, mergesort for ties), so motion leaves the center first. Ease is the smootherstep `6u^5 - 15u^4 + 10u^3`. The arc is perpendicular to the flight, `(-dy, dx)`, scaled by `arc_amp * length * s_i`. `s_i` is in `[0.5, 1]` with a random sign from `numpy.random.default_rng([seed, i])`. A zero-length flight has no arc. Rest centers stay on the cell grid so a settled frame matches a nearest-neighbor upscale. In flight the center is clamped inside the box by `0.75 * cell`, and the square is clipped to the box. The flying square is `(cell + 1) * (1 + lift * sin(pi * E))`, drawn with antialiasing, furthest along on top.

The outbound color blend is `recolor_strength * E`. The return blend goes all the way back to the original color so frame 0 and the last frame match. At `recolor_strength` 1 that is the same formula in both directions. Hooks and the pinned comment do not claim the pixels are unchanged while recoloring is on.

### Morph audio and verification

Ticks are the bounce mallet (middle velocity layer) on the 15-note C major pentatonic from C4. Pitch rises on the way to B and falls on the way back, plus a seeded step of -1, 0, or +1. Gain is 0.35 to 0.6, panned with equal power from the particle's x. The whoosh is band-passed noise (about 1.4x around a sweep from 300 Hz to 1800 Hz, and back) following the smoothed fraction of particles in flight. A C-E-G chime (decay 1.2 s) marks the B hold. A quieter 0.6 s chime marks the return home; the shared bounce master then zeros the last 300 ms, so most of that final chime is silence. The master is DC removal, a 12 kHz low-pass, 5 ms fades, the zero tail, then the loudness loop to -14 LUFS and -1 dBTP with the resample_poly true-peak meter. Audio length is `N * 800` samples.

`python -m fc_sat.verify` stays on bounce unless you pass `--mode morph` or the config contains `recolor_strength` (a `song` field still selects hop). Morph checks the container (1080x1920, 60 fps, H.264 high, yuv420p, BT.709 TV tags, AAC 48 kHz stereo, duration `N/60`, A/V within 20 ms, under 100 MB), loudness within 1 LU of -14, true peak at or under -1 dBTP, no sample clipping, and the last 300 ms under -50 dBFS. The loop SSIM of the first and last frame must be at least 0.995. The middle frame of the B hold, and frame 0, are averaged per cell and compared in OKLab to the prepared grids. That comparison fails the file only when `recolor_strength >= 0.99` and the mean error is above `max_delta_e`. There must be no black frame, and no 1 second window may contain more than 3 frames whose mean luma jumps by more than 0.10. Outside the image box and the hook, pixels must stay within 8/255 mean absolute RGB and 0.06 p99 luma of the vignette. A failure prints the measured values and writes the worst frame; the thresholds are not loosened to force a pass. A sample of frames is also drawn before compositing to confirm no particle pixel leaves the image box.

## Flag Arena

Top-down battle royale. Thirty-two country balls, same radius, same mass, same combat rules. There is no gravity. A `gravity` key anywhere in the config is rejected. The only impulses are ball contact, a ball's own dash, the boss, and the opening clash. Damping and the edge brake are accelerations and are not logged as impulses.

The solver is numpy. Fixed step 1/240. Hashes match inside that solver. pymunk is not used for the arena.

```bash
python tools/fetch_flags.py
python make_arena.py --cast configs/casts/world.yaml --out out/arena.mp4
python make_arena.py --cast configs/casts/world.yaml --out out/arena.mp4 --sim-only
python make_arena.py --cast configs/casts/world.yaml --out out/arena.mp4 --preview --seed 5 --contact-sheet
python make_arena.py --cast configs/casts/world.yaml --out out/arena.mp4 --full --seed 5
```

With no mode flag the command runs the 40-seed search and a debug reel. If fewer than 30% of seeds pass, it prints each gate's rate and exits non-zero. It still writes the reel of the seed that cleared the most gates. `--seed` renders that seed even when it would lose the search. `--full` is the only mode that encodes the 1080x1920 master. Do not start that until a seed passes.

Output names include the cast stem and the seed (`arena_world_s5.mp4`). Preview is 540x960 at 30 fps with no audio. The reel is 360x640 at 15 fps: circles, ISO codes, velocity vectors, windup lines.

### Tuned knobs

These are the only numbers that were moved off the spec table. Gates were not relaxed.

- `physics.damping` stays 2.2. Lower moves the opening clash earlier than 0.4 s. Higher drops mean speed under 140 px/s.
- `ai.dash_speed` is 1200, not 1500, so one hit does not cross the floor.
- `ai.aggression_scale` is 1.2.
- `platform.keyframes` start at half-size 1020 x 1150 and shrink at about 12 px/s, under the 28 px/s cap. The spec 410 x 460 floor wiped the cast before 12 s. Camera zoom is `min(410 / hw, 460 / hh)`, so the wider floor still fills the same on-screen box and the balls grow as it shrinks. The opening punch is clamped by that limit and shows up only after the floor has shrunk.

A 40-seed search on this tuning passed 0/40. The gates that still miss are the final duel, the winner window, and dead time. Causes, the opening impact, and neighbor spacing passed. See `out/sim_report.md`.

### Budgets

One seed is meant to stay under 3 s. Long matches that run to the horizon are slower than that on this machine. The debug reel for seed 5 encoded in about 21 s. One full-resolution frame measured 45 ms; 2881 frames at that rate is about 2.1 minutes in-process. That is under 25 minutes, so half-resolution bloom and fewer particles were not applied.

### Guards and casts

`configs/guards.yaml` is the block list. Casts cannot override it. `--allow-sensitive` prints a warning and continues.

- Religious inscriptions: SA, AF, IQ, IR, BN
- Contested: TW, PS, XK, EH
- Active conflict: RU, UA, IL, SD, SS, MM

The cast is locked at 32. Quote ISO codes in YAML (`NO` would otherwise become false). The world cast starts BR, AR, FR, DE and ends with iso3 VNM. Africa uses GM and SL. Asia uses BT.

Flags are flag-icons v7.5.0 (MIT). `tools/fetch_flags.py` confirms that tag and does not fall through. The Ohio cameo is a stylized burgee when Commons is not public domain. That is recorded in `assets/cameos/CREDITS.md`.

### Memes and hooks

`configs/memes.yaml` placeholders are `{killer}`, `{victim}`, `{cameo}`, and `{name}`. Phrases do not mention nationality. The validator rejects a denylist token and warns above 34 characters. One hook line in `configs/arena_hooks.yaml` is 25 characters (`32 countries. 1 survives.`), so `hook_max_chars` is 25. The sub-caption is "Same size. Same weight. Same moves." It does not say "pure physics."

The two clash balls draw their first cooldown from `[0.7, 1.2]` so a windup cannot cancel the opening impact. Other balls use `[0.2, 1.2]`.

### Audio and voice

`finish_broadcast` removes DC, fades 5 ms, and runs the shared loudness loop. It does not call `master()`. The AAC delivery path still clips at about -9.2 dB and low-passes at 12 kHz so the native encoder stays at or under -1 dBTP. `{out}.sfx_only.mp4` is the picture with effects and no bed.

`{out}.announcer.txt` is always written. The header starts with `target model: Eleven v4 (verify the model id in the ElevenLabs docs)`. The request uses `voice.model_id`. No API call is made without `ELEVENLABS_API_KEY`. A missing key logs one line and continues.

### Phone checklist

Watch a passing master on a phone speaker before you post it.

- Flags still read at arm's length, including the kill-feed marks and the podium.
- You can point at the hit that knocked someone out. The windup line is visible before the dash.
- Impacts and the bed balance on the phone speaker. The drop before the last knockout is obvious, and the celebration is still audible.
- Hook, counter, kill feed, and the end card sit clear of the platform box.

## Odd One Out

A static grid, one item different. Three levels: hue, tilt, then a dot shifted off center. Each has a countdown and a reveal. Frame 0 is already level 1 with a full timer. Captions never claim a percentage. Fills are flat: no bloom, glow, gradient, shadow, or specular.

```bash
python make_odd.py --config configs/odd_default.yaml --out out/odd.mp4
python make_odd.py --config configs/odd_default.yaml --out out/odd.mp4 --ladder
python make_odd.py --config configs/odd_default.yaml --out out/odd_preview.mp4 --preview
python make_odd.py --config configs/odd_default.yaml --out out/odd.mp4 --audio-only
python -m fc_sat.verify out/odd.mp4 --config configs/odd_default.yaml
python make_odd.py --config configs/odd_default.yaml --out out/odd.mp4 --full --approved
```

The default command writes the sim, a clean puzzle PNG and a ringed answer PNG per level, the contact sheet, the report, the answers, and the post text. It does not encode video. `--ladder` writes `ladder/{type}_rung{n}_{count}items.png` (15 stills), `ladder/ANSWERS.md`, and `ladder/index.html`. It does not encode video and finishes in under 30 s. Open the html file in a browser to time yourself. `--preview` is 540x960 at 30 fps, no audio, coordinates halved, preset `veryfast`. `--full` is refused unless `--approved` is also passed. That command is the 1080x1920 60 fps master (H.264 high, yuv420p, crf 16, preset slow, AAC 192k, BT.709 tv, +faststart).

### Defaults

Items fade from `pop_floor` 0.5 to full over 0.2 s, so frame 0 already shows the discs and the timer is already full. Positions never move. The odd cell is one uniform draw and is not repeated on the next level. `detail_mode` is `moved`: the odd dot is shifted by a fraction of the item radius in one of eight seeded directions, and the others stay centered. The dot radius is 14% of the item radius. The seconds number is centered on the timer row, to the right of the bar, so it does not meet the caption. The outro question is 52 px at y 420, above the field. The CTA is 40 px at y 1450, below it. There is no bloom key.

Difficulty is a rung from 1 to 5, 1 easiest. Normal uses hue rung 2 (OKLab 0.15, lightness at least 0.05), tilt rung 3 (9 degrees), and detail rung 4 (offset 0.22 of the radius). The CVD floor is half the hue rung's distance, so rung 2 requires 0.075, which is above 0.07. Easy uses rung 1 on all three (0.20, 18 degrees, offset 0.55) and does not lower that CVD rule.

### Tiers

`normal` is the upload film: 5x5 at 90 px, 6x6 at 84 px, 7x7 at 76 px, timers 6, 7, 8 s, reveal 1.2, dissolve 0.2, outro 2.0, 27.0 s (1620 frames). `easy` uses the same grids and timers with rung 1. `hard` adds level 4, a 7x7 hue at rung 5 with a 9 s timer, and runs 37.4 s. The verifier's `[22, 32]` s window is for the three-level film. A hard file fails that window on purpose. Rung 5 is allowed, but its pixel measurement checks print `SKIPPED (not reliably measurable after compression)` and the report flags them. Rungs 1 to 4 are measured.

### Config keys

`configs/odd_default.yaml` is the schema. Unknown keys and missing keys name the field. `hook` must be a line in `configs/odd_hooks.yaml`. Reveal labels are "It was a different color", "It was tilted", and "The dot was off center". The caption validator rejects a digit run followed by `%`.

Top level: `generator`, `seed`, `tier`, `width`, `height`, `fps` (locked to 1080, 1920, 60), `workers`, `reveal_seconds`, `dissolve_seconds`, `outro_seconds`, `pop_seconds`, `pop_floor`, `hook`, `cta`, `background`, and `rung` (`hue`, `tilt`, `detail`, each an integer 1 to 5). Then `levels` (`id`, `difference`, `grid`, `size`, `timer`, optional `rung`), `field`, `item` (including `detail_mode` and `dot_fraction`), `layout`, `constraints`, `audio`, and `tiers`. A tier carries `level_ids`, `base_l`, `base_c`, and an optional `rung` block. A missing tier rung uses the top-level rung. A level `rung` overrides that type's rung for that level only. Ladder image seeds are `seed + 100 * rung + level id`.

The distance tables are in code:

| rung | hue OKLab | tilt degrees | detail offset |
| --- | --- | --- | --- |
| 1 | 0.20 | 18 | 0.55 |
| 2 | 0.15 | 13 | 0.42 |
| 3 | 0.12 | 9 | 0.30 |
| 4 | 0.09 | 6 | 0.22 |
| 5 | 0.06 | 4 | 0.15 |

### How detectability is checked

`python -m fc_sat.verify` with mode `odd` (automatic when `generator: odd`) checks the container, the `[22, 32]` s window, size under 100 MB, loudness within 1 LU of -14, true peak at or under -1 dBTP, no clipped samples, hook pixels on frame 0, the photosensitivity jump count, layout math, rendered text boxes (at most four, none on the field), reveal frames sitting on the timer boundary, one odd item, no repeated cell, and CVD.

Pixel checks use the sim positions. They run on the master and again on `{stem}.harsh.mp4` (crf 30, preset veryfast, 720x1280). The default command also prints those measurements for one settled frame per level. Rung 5 skips the measurement and prints `SKIPPED (not reliably measurable after compression)`.

- hue: OKLab distance of a 5x5 center patch versus the median of up to 12 other items, at least 70% of the rung's nominal distance
- tilt: `minAreaRect` angle folded into 0..45 degrees; the odd item differs from the others' median by at least 60% of the nominal angle
- detail: white-pixel centroid offset from the item center, divided by the item radius. The odd item is at least 60% of the nominal offset. Every other item stays under 25% of that nominal offset

A failure prints the measured value and writes `{stem}.L{n}.worst.png`. Thresholds are not loosened to force a pass.

### Phone checklist

- L1's color gap reads on a phone. L2's 9 degree tilt is visible in a still. L3 is the disc whose dot sits off center, and nothing else points at it before the reveal.
- Open `ladder/index.html` and time each still before trusting a rung.
- The blip, the ticks, and the reveal ding balance on a phone speaker.
- Level label, caption, timer, and the outro lines stay clear of each other and of the field. At most four text elements are on screen.

## Collatz

A vertical Short about the Collatz conjecture. The picture is drawn with skia (Pillow 2x supersampling if skia is missing) and piped through the same BT.709 ffmpeg command as the other films. Other generators are not part of this pipeline.

```bash
python make_collatz.py doctor
python make_collatz.py facts
python make_collatz.py vo
python make_collatz.py animatic --out out/collatz_animatic.mp4
python make_collatz.py hooks
python make_collatz.py audio --out out/collatz.mp4
python make_collatz.py preview --hook A --out out/collatz_preview.mp4
python make_collatz.py full --approved --out out/collatz.mp4
```

`full` is refused without `--approved`. The partial file is `{stem}.partial.mp4` and is renamed into place only after ffmpeg exits 0. Ctrl+C deletes it.

### Defaults

Hook A ("Always"). `reply_commitment` is true, so the description asks for a number. Voice stability is 0.5 and similarity is 0.75, except L8 and L10 which use stability 0.35. Seeds are `1000+0..2`. Pronunciation stays plain (`Collatz`, `sextillion`) until `configs/collatz_pronunciation.yaml` says otherwise. Take picks live in `configs/vo_picks.yaml`. There is no ElevenLabs key in the repo; `vo` writes `out/script_for_manual_tts.md` and accepts `L1.wav` ... `L14.wav` via `--vo-dir`. Style and speed are not sent. If a setting is rejected, the client retries without it.

The verification bound is whatever https://pcbarina.fit.vutbr.cz/ states as already verified. On 2026-10-02 that was 2075×2^60 (2.39 sextillion), ahead of the paper's 2^71 (2.36 sextillion). "Over two sextillion" stays true while the bound is at least 2^71. `facts` prints the diff when the page moves. c_1937 is included because Wikipedia and the MacTutor biography both say Collatz proposed it in 1937.

Step 36 of 27, at 5 frames per step from 13.30s, is 16.30s. The outline's 17.30s does not match that clock, so the retention map uses 16.30s.

Text is capped at 4 elements. Width is fitted down to 60% of the asked size so the ink stays inside x [130, 950] and y [200, 1536]. On the 111-step chart the node radius shrinks with the step spacing so the line stays readable. The counter draws the integer short-scale prefix and its name (`2.39 sextillion`), never a binary float. The wav is held to -2 dBTP so the AAC file stays at or under -1 dBTP.

### Claims

`configs/claims.yaml` is the only place a number is allowed to come from. Computed claims are re-run on every `facts` build. A digit or a spelled number in the voiceover, captions, titles, or description must name a claim or sit on the whitelist (`1`, `30`). The report is `out/claims_report.md`.

### Voice

Confirm the ElevenLabs plan allows commercial use. The voice is synthetic. YouTube's altered-content toggle is for realistic altered content; a plainly animated explainer with an AI narrator is probably outside it, but check the current Studio question before publishing. No SSML breaks. One bracket tag per line, at the start, and not on the hook. Numbers in the spoken line are spelled out.

### Hook lab

Mute the first 2 seconds and watch it at thumb distance. `hooks` writes a 540x960 clip and a 360px-wide strip for A, B, and C. Pick the variant whose promise is obvious with the sound off. The final render takes `--hook A|B|C`.

### Posting

`postkit` writes `out/postkit.md` from the claims. Pin comment A. Reply to a number with `python tools/collatz_reply.py N`. Read Viewed vs Swiped away against `out/retention_map.md`. If the first 2 seconds lose about 70% to swipes, change the hook and upload the next variant at least a day later.

### Phone checklist

- The hook line is readable on a phone with the sound off, at arm's length.
- The promise is clear in the first 2 seconds.
- The voiceover is intelligible on a phone speaker.
- The ending stays open: the chord has no third, and the last line is "Pick another number."
