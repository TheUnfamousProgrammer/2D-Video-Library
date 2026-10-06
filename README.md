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

Output names include the cast stem and the seed (`arena_world_s5.mp4`). Preview is 540x960 at 30 fps and includes the mix. The reel is 360x640 at 15 fps: circles, country names, velocity vectors, windup lines. On-screen labels use the full country name with its flag. While four balls are alive the banner names the semifinalists, and while two are alive it names the finalists.

### Tuned knobs

These are the only numbers that were moved off the spec table. Gates were not relaxed.

- `physics.damping` stays 2.2. Lower moves the opening clash earlier than 0.4 s. Higher drops mean speed under 140 px/s.
- `ai.dash_speed` is 1200, not 1500, so one hit does not cross the floor.
- `ai.aggression_scale` is 1.2.
- `platform.keyframes` start at half-size 1020 x 1150 and shrink at about 12 px/s, under the 28 px/s cap. The spec 410 x 460 floor wiped the cast before 12 s. Camera zoom is `min(520 / hw, 640 / hh)`, so the wider floor fills a larger on-screen box. Flags are drawn at twice the collision radius, because a physics-sized ball on that floor is too small to read. The balls still grow as the floor shrinks. The opening punch is clamped by that limit and shows up only after the floor has shrunk.

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

Difficulty is a rung from 1 to 5, 1 easiest. Normal uses rung 5 on all three: hue OKLab 0.03 (lightness at least 0.02), tilt 4 degrees, detail offset 0.15 of the radius. The CVD floor is half the hue distance, so rung 5 requires 0.015. Easy uses rung 1 on all three (0.20, 18 degrees, offset 0.55) and does not lower that CVD rule. Rung 5 is played, and its pixel check prints `SKIPPED (not reliably measurable after compression)`. The lightness floor is 0.02 so the hue gap is not also a brighter disc. Rung 5 hue is 0.03 because 0.06 still read as a different color.

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
| 5 | 0.03 | 4 | 0.15 |

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

## Polycircle

A 30.4 s vertical Short. A square gains one side on each beat, then faster, until it looks like a circle. A zoom shows it is still a polygon. The sides then double until the gap is gone, and the last beat snaps back to the square so the video loops. The picture is drawn with skia (pycairo if skia is missing, otherwise Pillow at 2x) and piped through the shared BT.709 ffmpeg command. Other generators are not part of this pipeline.

```bash
python make_polycircle.py doctor
python make_polycircle.py facts
python make_polycircle.py timeline
python make_polycircle.py animatic
python make_polycircle.py hooks
python make_polycircle.py audio
python make_polycircle.py preview --hook A --out out/polycircle_preview.mp4
python make_polycircle.py postkit
python make_polycircle.py verify --out out/polycircle_preview.mp4
python make_polycircle.py full --approved --hook A --out out/polycircle.mp4
```

`full` is refused without `--approved`. Draft encodes (animatic, hooks, preview) use ffmpeg preset `veryfast`. The full file uses `slow`. The shared pipe writes `.{stem}.partial.mp4` and renames it into place only after ffmpeg exits 0. Ctrl+C deletes the partial file.

### Defaults

Claims live in `configs/polycircle_claims.yaml`. `configs/claims.yaml` belongs to Collatz, so this film does not use it. On-screen text is `configs/polycircle_text.yaml`. The default hook is A. Tempo is 150 bpm, which lands on exact frames (24 per beat, 96 per bar). Other tempos snap to the nearest frame and `facts` prints the maximum snap error, which must stay at or under half a frame.

Bar colors are the eight hues in the config, shifted in OKLab so their lightness stays within 0.03. The camera's "screen center" is the anchor at (540, 910), the polygon center, not the canvas midline at y=960. At 36x zoom the top of the circle sits on that anchor. The gold dimension line is drawn 36 px to the right of the center column so the gap measurement stays clean. Gap labels always use the 1080 px frame numbers, including in the half-resolution preview. The line "THE GAP KEEPS SHRINKING" is wrapped so each line stays at or under 22 characters. The SIDES label becomes the polygon's name (square, pentagon, hexagon, and on through the systematic names) while that count holds for at least 12 frames. The 96-gon is an enneacontakaihexagon, the 61-gon is a hexacontakaihenagon, and the infinity counter is an apeirogon. Faster than that, the label goes back to SIDES.

Rotation speed rises from about 30 deg/s to about 150 deg/s and then eases to a stop at frame 864. A single scale (printed by `facts`) makes the freeze angle exact and makes frame 0 match that angle modulo a quarter turn, which is the square's symmetry. The opening shape is a diamond, a vertex near the top, and frame 1823 matches frame 0.

"NEVER" is at frame 1632, which is 27.2 s, because bar 18 starts there. The retention map uses that computed time. The infinity sign is the font glyph when JetBrains Mono has it, and a vector lemniscate otherwise. The music-off file is the sound-effects bus only (impacts, whoosh, risers, bells, ticks, the chime, the tape stop, and the reverse cymbal), mastered the same way as the full mix. A 10 metre circle needs 158 sides before the gap is under 1 millimetre; that reply is in the post kit.

### Claims

Every number on screen, in a title, or in the description comes from the claims file and is recomputed on each `facts` run. A digit sequence that does not map to a claim fails the lint. The report is `out/claims_report.md`.

### Hook lab

Mute the video, watch the first 2 seconds at thumb distance, and pick the clearest promise. `hooks` writes the first 3.5 s of variants A, B, and C at 540x960 and 30 fps, with the kick on the first frame, plus a muted 360 px strip of frames at 0.0, 0.5, 1.0, and 2.0 s. The first new side arrives at 0.4 s. The final render takes `--hook A|B|C`.

### Posting

`postkit` writes `out/postkit.md` from the claims, and stills of the drop, the zoom hold, and the 61 frame in `out/thumbs/`. The upload title is the phrase YouTube autocomplete completes, and it stays inside 40 characters. Hashtags stay in the description, five of them, `#shorts` first. Tags are the real search phrases plus the misspelling. Set the audience to not made for kids so the pinned comment can collect replies. On the altered-content question, the music is original and there is no third-party footage. Pin comment A. In YouTube Studio, read Viewed vs Swiped away and the retention graph against `out/retention_map.md`. If swipe-away is high, post the next title as its own upload at least a day later.

### Phone checklist

- The hook is readable on a phone with the sound off.
- The fast add phases, including the 3-frame steps, stay readable.
- The sub-bass is audible on a phone speaker.
- The loop feels seamless.
- The ending lands: the square returns, and the next kick would fall on the seam.

## Circlesquare

A 30.4 s vertical Short. A chain of rotating circles traces a closed curve. Each beat adds one circle, and the curve turns from a circle into a square. A zoom into a corner shows the square is still rounded. The circles then double until the corner is sharp on a 1080 px frame, and the last beat snaps back to one circle so the video loops. There is no voiceover. The picture is drawn with skia (pycairo if skia is missing, otherwise Pillow) and piped through the shared BT.709 ffmpeg command. The music engine is the polycircle score, with this film's timeline and seed 11. The shared master stops the wav at -2.3 dBTP; this mix trims another 0.55 dB so the AAC file stays at or under -1 dBTP. Polycircle outputs are not touched.

```bash
python make_circlesquare.py doctor
python make_circlesquare.py facts
python make_circlesquare.py timeline
python make_circlesquare.py stills
python make_circlesquare.py hooks
python make_circlesquare.py audio
python make_circlesquare.py preview --hook A --out out/circlesquare_preview.mp4
python make_circlesquare.py postkit
python make_circlesquare.py verify --out out/circlesquare.mp4
python make_circlesquare.py full --approved --hook A --out out/circlesquare.mp4
```

`full` is refused without `--approved`. Draft encodes use ffmpeg preset `veryfast`. The full file uses `slow`. The shared pipe writes a partial mp4 and renames it into place only after ffmpeg exits 0. Ctrl+C deletes the partial file. Preview and hook clips are 540x960 at 30 fps. The posted file is 1080x1920 at 60 fps, 1824 frames.

### Math

The square is centered at (540, 905), half-side 340 px. The curve is the Fourier series of that square: frequencies 1, -3, 5, -7, ... with coefficients `r1 / k^2 * exp(i pi/4)`, and `r1 = 8 sqrt(2) a / pi^2` (389.7 px). One revolution takes one bar, so the tip meets the corner ray on every bar line and frame 1824 has the same phase as frame 0. The gap is the maximum distance from the curve to the square's perimeter. It is computed on a dense sample through K = 2976 and by `2a / (pi^2 K)` after that. The two agree at the overlap. The smallest K with a gap at or under half a pixel is 138.

At K = 2976 the world gap is 0.023 px, not the 0.01 px a first estimate suggested. The exact-square shortcut still draws the four corners when K is above 2976 and the zoom is at most 2, because 0.023 px is invisible at that zoom. The gap drops under 0.01 px around K = 11904. When the zoom is above 2 the true curve is drawn. K above 23808 reuses the 23808-circle curve (or the exact square at low zoom). The counter still shows the true K. The blend into 47616 happens while the zoom is still 36, so that one doubling is the capped curve at 36x; the leftover error is about a tenth of a pixel.

Frame 954 is the published corner measurement (zoom 36, 93 circles, gap about 27 px on screen). The first doubling therefore eases across frames 955-960 instead of the usual eight frames, so that measurement is still the settled 93-circle curve. Later doublings ease across the eight frames before they land.

Frame 1823 is phase-locked to frame 0 (a one-frame hold) so the loop seam matches. A pure 96-frame rotation would move the arm by about 25.5 px on that last frame, and the global SSIM of that motion sits just under 0.99.

### Claims

Every number on screen, in a title, or in the description comes from `configs/circlesquare_claims.yaml` and is recomputed on each `facts` run. A digit sequence that does not map to a claim fails the lint. The report is `out/claims_report.md`. Gap labels use two significant digits in fixed decimals, never scientific notation. From frame 948 to 1343 the sub-line is the gap at 36x, including while the camera zooms back out, because the label says "AT 36x".

### Hook lab

`hooks` writes the first 3.5 s of variants A, B, and D at 540x960 and 30 fps, with audio, plus a muted strip of frames at 0.0, 0.2, 0.5, 1.0, and 2.0 s. It prints the curve's bounding-box fraction, the design stroke, the design text size, and the motion in the first 24 frames. Those stroke and text numbers are design pixels on the 1080 px frame. A 16-character line at 72 px does not fit a 540-wide picture at 72 output pixels, so the half-resolution files scale the type and keep strokes at least 10 output pixels. Variant D shows the finished-looking square for the first beat, then cuts to one circle. The final render uses A.

### Posting

`postkit` writes `out/postkit.md` from the claims. Upload with #Shorts. Set the audience to not made for kids. On the copyright question, the music is original and there is no third-party footage. Pin comment A (a 10 metre square needs about 1,013 circles to land within 1 millimetre). Then read Viewed vs Swiped away and the retention graph against `out/retention_map.md`. If swipe-away is high, post the next hook variant as its own upload at least a day later.

The top slot can show two lines, so a frame can carry five text strings (two lines, the CIRCLES label, the counter, and the gap). The safe zone still holds, and the boxes do not overlap. The dashed reference square is a 4 px stroke at 55% opacity with a 6 px dash. During the zoom the two edges at the corner are solid 4 px lines at 80% opacity.

### Phone checklist

- The hook is readable on a phone with the sound off.
- Frame 0 is a filled circle, already turning, with a thick stroke.
- The corner zoom is the moment the square stops looking finished.
- The sub-bass is audible on a phone speaker.
- The loop feels seamless.

## Paperfold

A 30.4 s vertical Short. A sheet of paper doubles on every fold while the camera pulls back through seven paper-cut dioramas, from a desk to deep space. At 30 folds the stack reaches space. The paper needed for that fold is about 400 times the Earth to Sun distance. In theory fold 42 passes the Moon. The real record is 12 folds. The last frames collapse back to the opening card so the video loops. There is no voiceover. The stack's width is not to scale.

```bash
python make_paperfold.py doctor
python make_paperfold.py ingest
python make_paperfold.py stills
python make_paperfold.py facts
python make_paperfold.py timeline
python make_paperfold.py audio
python make_paperfold.py hooks
python make_paperfold.py preview --out out/paperfold_preview.mp4
python make_paperfold.py postkit
python make_paperfold.py verify --out out/paperfold.mp4
python make_paperfold.py full --approved --out out/paperfold.mp4
```

`full` is refused without `--approved`. The master is 1080x1920 at 60 fps, 1824 frames. Preview and hook clips are 540x960 at 30 fps. The pipe sets `-framerate`, `-fps_mode cfr`, and an output `-r`, and tags the picture BT.709. The file is written to a `.partial.mp4` and renamed into place only after ffmpeg exits 0.

### Art

Plates and cutouts live in `assets/art/plates` and `assets/art/objects`. `ingest` writes lossless PNG in `assets/art/normalized` and keyed PNG in `assets/art/keyed`. A plate whose aspect is within 1.5% of 9:16 is scaled to fill and center-cropped. Plates smaller than 1080x1920 are upscaled with Lanczos and a mild unsharp, and the step is logged. Each plate is shifted so its ground line lands on the frame's ground line. Rows the shift adds below the picture are rebuilt from the plate's own flat bottom band: a per-column median, a low-pass across x (sigma 40 px), then grain of sigma 1.5. They are never a copy of a content row.

The deep-space plate does not get speckle removal, denoise, or a sky mask. Its teal top band becomes a vertical gradient over the top 250 rows with a 48 px feather, and the Moon is left untouched. Street and city plates get a chroma-only denoise only when isolated cyan or red specks are measured. If the city plate's tallest tower is taller than the Burj Khalifa at that scene's scale, ingest replaces it with a procedural low-rise skyline and records the choice in `out/qa_log.md`.

### Claims

Constants and formulas live in `configs/paperfold_claims.yaml`. `facts` recomputes every claim and fails a digit on screen, in a title, or in the description that is not mapped to one. Stack height is `h(n) = t * 2^n` with `t = 0.1 mm`. The paper length is Gallivan's single-direction formula. Display strings such as "PAPER NEEDED 400x EARTH TO SUN" are formatted from those values. The description and the pinned comment say "about 400x" and list the assumptions: 0.1 mm paper, single-direction folds, Gallivan's formula.

### Hook lab

`hooks` writes four silent 540x960 clips and prints, for each variant, the main-shape area, the minimum stroke, the minimum text size, and the motion energy over the first 24 frames. A is the default and the one `full` uses: "42 FOLDS." / "THE MOON?". D opens on the Moon still for 24 frames, then cuts to the card.

### Posting

`postkit` writes `out/postkit.md`. Upload with #Shorts, set the audience, answer the copyright and altered-content questions (the music is original, the art is AI-generated), and pin comment A. Then read Viewed vs Swiped away and the retention graph against `out/retention_map.md`. Reply to a place or a distance with `python tools/fold_reply.py moon` or `python tools/fold_reply.py 100 km`. Units are km, m, ly, and AU. Unknown names, zero, and negative distances are rejected. A distance past fold 120 is reported as beyond that fold.

## Linedraw

A 30.4 s vertical Short. Random straight lines are thrown onto a sheet of sketch paper. The ones that move the drawing closer to a picture are kept. At first it is chaos. The music builds, the lines stop for one silent beat, and on the drop the picture snaps into view. A zoom shows that the mouth is only crossing straight lines. The last beat throws the lines off the page so frame 1823 matches frame 0. There is no voiceover. The meaning is in the type.

```bash
python make_linedraw.py doctor
python make_linedraw.py optimize --target assets/target/target.png
python make_linedraw.py stills --target assets/target/target.png
python make_linedraw.py audio --target assets/target/target.png
python make_linedraw.py hooks --target assets/target/target.png
python make_linedraw.py preview --target assets/target/target.png
python make_linedraw.py full --approved --target assets/target/target.png
```

`full` is refused without `--approved`. The master is 1080×1920 at 60 fps, exactly 1824 frames, H.264 high, CRF 16, preset slow, BT.709 television range, AAC 256k at 48 kHz, `+faststart`. The pipe sets `-framerate 60`, `-fps_mode cfr`, and an output `-r 60`. The file is written to `linedraw.partial.mp4` and renamed into place only after ffmpeg exits 0. A second file, `out/linedraw.music_off.mp4`, is the same picture with scratches, whooshes, the impact, and the tape stop only. Preview and hook clips are 540×960 at 30 fps.

### Setup

Use the project venv from the top of this file, then install the pinned requirements. Linedraw adds `scikit-image`, `soundfile`, and `numba` on top of numpy, scipy, pillow, opencv-python, skia-python, and pyloudnorm. Numba is optional: if it does not install, the fly-off painter falls back to OpenCV. ffmpeg must be on `PATH` (Homebrew is enough). If it is missing, the encoder uses the `imageio-ffmpeg` binary.

Fonts are the OFL files `assets/fonts/Montserrat-ExtraBold.ttf` and `assets/fonts/JetBrainsMono-ExtraBold.ttf`, with their license texts beside them. If a download fails, the renderer uses the system bold fallbacks in `fc_sat/fonts.py`.

### Run it on any picture

```bash
python make_linedraw.py full --approved \
  --target path/to/picture.png \
  --subject "A NAME" \
  --credit "Photo by Ada" \
  --rights own
```

- `--target` defaults to `assets/target/target.png`.
- `--subject` is optional. It is used once, on the drop caption, at most 18 characters per line. With no subject the caption is "THERE IT IS".
- `--credit` is optional. It is printed in the description before "Music: original, made with code."
- `--rights` is `public-domain`, `own`, `licensed`, or `unknown` (the default).
- `--show-original` is `auto`, `yes`, or `no`. Auto shows the color crop when rights are public-domain, own, or licensed, and compares an earlier drawing with the finished one when rights are unknown.
- `--crop x0,y0,x1,y1` overrides the automatic crop, in source-image pixels.
- `--weight` is 0 to 4 (default 2.5) and pulls lines toward the face.
- `--seed` defaults to 7. The same seed repeats the same lines.

Rights warning: when `--rights` is `unknown`, the image may be copyrighted and it may show a real person. The post kit and this section say so. The render is not blocked. Do not post that video until you have permission, and answer YouTube's copyright and altered-content questions honestly.

### Algorithm

The picture is luminance. If the shorter side is under 600 px it is upscaled with Lanczos first. The crop is aspect 0.69. OpenCV's frontal-face cascades (default and alt2, plus a slightly blurred copy) pick the largest face and the crop tries to put it in the upper third. If that would cut the head off, the crop keeps the head and the face sits a little lower. With no face, the crop uses the spectral-residual saliency peak, then a center crop. A 2,000-line trial picks the CLAHE clip (0.012 or 0.02) with the higher likeness.

The search grid is 254×368. One cell is 3 output pixels, so the paper on screen is 762×1104. The paper starts at mid grey. Each step throws 150 candidate lines. Dark ink subtracts and light ink adds, with strength 0.06 per unit of anti-aliased coverage. A candidate is kept only when its weighted gain is positive. The longest lines shrink from 300 cells to 40. The search stops at 36,000 kept lines or after 2,000 empty steps. Likeness is the SSIM of the canvas and the target after a 1.5-cell blur, logged every 100 kept lines, in `out/lines.npz` and `out/lines_report.md`.

L_aha is the first logged count at 60% of the final likeness. The film is supposed to hold at 60% of L_aha before the drop. If that count is already past 30% likeness, the face is readable too early, so the silent beat holds at the last logged count still under 30% and the drop still lands on L_aha. The choice is in `out/lines_report.md`.

### Sound

Original, synthesized at 48 kHz. 150 bpm, A minor, chords Am F C G. An 808 with a pitch drop and tanh drive, kick, clap, hats, a cowbell motif, a soft pad, vinyl crackle, a pencil scratch per landed line, hook plucks, whooshes, a drop impact, a riser, a snare roll, and a tape stop. Buses are sidechained to the kick. A small room sits on the cowbell, claps, and scratches. The master is high-passed at 30 Hz, gently compressed, limited, and normalized to about −14 LUFS with true peak at or under −1 dBTP. The wav is held a little lower so AAC stays under that ceiling. Frames 744–767 are silent. Frames 1806–1823 are silent so the loop restarts clean. Stems are in `out/stems/`. The measured gates are in `out/audio_report.md`.

### Posting checklist

`postkit` writes `out/postkit.md`. Titles do not say what the picture is. The description starts with the hook, then one sentence on the rule (random lines are thrown and only the ones that bring the drawing closer are kept; likeness is a similarity score), then the credit and "Music: original, made with code.", then `#shorts #art #satisfying #generativeart #asmr`.

Before you upload:

- Set the audience. If rights are unknown the picture may show a real person, so do not mark it made for kids.
- Answer the copyright question and the altered-content question. The music is original. The picture was redrawn with code.
- Pin "{kept} lines kept out of {thrown} thrown."
- After it posts, compare Viewed vs Swiped away and the retention graph with `out/retention_map.md`. The silent beat is at 12.4 s and the drop is at 12.8 s.


## Coinspin

A 30.4 s vertical Short. A gold coin rolls once around a coin of the same size and the counter counts its spins: 2, not 1. A coin twice as wide gives 3. Then the 1982 SAT question: a coin one third the size of the other, with the printed choices 3/2, 3, 6, 9/2 and 9. The counter reaches 3 (the test's answer) with a quarter lap still to go, the beat drops out, and the drop lands on 4. Rim ticks split it into 3 from rolling and 1 from the trip around. The big coin doubles on every beat and the +1 stays. The coins become the Sun and Earth: 365 days, 366 spins against the stars, one every 23 h 56 m 4 s. The last beat snaps back to two coins so frame 1823 matches frame 0. There is no voiceover.

```bash
python make_coinspin.py doctor
python make_coinspin.py facts
python make_coinspin.py timeline
python make_coinspin.py stills
python make_coinspin.py hooks
python make_coinspin.py audio
python make_coinspin.py preview --hook A --out out/coinspin_preview.mp4
python make_coinspin.py postkit
python make_coinspin.py verify --out out/coinspin.mp4
python make_coinspin.py full --approved --hook A --out out/coinspin.mp4
```

`full` is refused without `--approved`. The master is 1080x1920 at 60 fps, 1824 frames; preview and hook clips are 540x960 at 30 fps. Outputs use a `coinspin_` prefix so other films' `out/` files are not overwritten. The music engine is the polycircle score with this film's events and seed 13: every quarter turn of the arrow is a pluck and every finished spin is a bell. The mix keeps the circlesquare AAC trim of 0.55 dB.

### Rolling

Angles run clockwise from straight up. In one lap the rolling coin's center goes once around the fixed coin, and with no slipping its arrow turns `(R + r) / r` times as far, so a coin `k` times wider gives `k + 1` spins. The stage is radius 400 px at (540, 915), and the pair is scaled by `400 / (k + 2)` so it always fits. Laps: frames 0-192 at 1x (spins on 96 and 192), 216-360 at 2x, 384-768 at 3x (spins on 480, 576, 672, 768), then an uncounted 3x lap to 960. Between laps the coin sits on top for one beat while the big coin grows. Every spin lands on the sixteenth-note grid. From 960 the big coin doubles on each beat, 6x to 196,608x, easing over 8 frames. A rolling coin under 20 px gets a gold ring and is never drawn smaller than 7 px. Earth's year runs 1392-1488 with a sine ease so it never moves more than about 34 px a frame.

### Claims

`configs/coinspin_claims.yaml`. The spin counts are computed by walking a lap through `rolling_pose`, the function the renderer draws, and checking at every sample that the coins touch and that the arc swept on the fixed rim equals the arc swept on the rolling rim. The SAT year, ratio, choices, and intended answer are cited (Scientific American, 2023). The tropical year (365.24219 days) is cited; the spins against the stars and the 23 h 56 m 4 s day are computed from it. `screen_counts` is every integer any counter shows; regenerate it from `fc_sat.coinspin_schedule.screen_counts()` if the schedule changes, or `facts` fails.

### Verify

`verify` checks the claims, the style law, that every spin bell lands on a frame where the arrow points straight up and the counter steps by one, that the coins touch on every rolling frame and stay inside the stage, that the rolling coin never jumps, the safe zone, overlaps, the six-element text limit, that every digit on screen names a claim, the loop seam SSIM, the frame 0 coin radius in pixels, luminance flashes, and with `--out` the file's size, rate, frame count, loudness, true peak, kick onset, spin onsets, and silent tail. The report is `out/coinspin_verify_report.md`.

### Posting

`postkit` writes `out/coinspin_postkit.md` and stills in `out/coinspin_thumbs/`. Upload with #Shorts, audience not made for kids. The music is original and the drawing is made with code. The Sun and Earth are not drawn to scale. Pin comment A (a coin 10x wider gives 11 spins), keep the reply, and post comment B (inside a ring 3x wider it spins 2 times) under the first wrong answer. Read Viewed vs Swiped away against `out/coinspin_retention_map.md`. If swipe-away is high, post hook D ("THE SAT GOT / THIS WRONG.") as its own upload at least a day later.

### Phone checklist

- The question is readable with the sound off, and the coin is already rolling on frame 0.
- The counter reads 1 while the coin is only halfway around.
- At frame 672 the coin is visibly not home yet while the counter says 3.
- The three gold ticks and the gold orbit read as 3 + 1.
- The loop feels seamless.
