#!/usr/bin/env python3
"""Why does your voice sound so weird on recordings?

    python make_voice.py voice                 # Eleven v4 take (cached) -> out/voice/vo.wav + word times
    python make_voice.py stills                # contact sheet
    python make_voice.py full                  # clean 1080x1920 60 fps
    python make_voice.py full --captions --out out/voice/voice_captions.mp4
    python make_voice.py thumb                 # out/voice/thumbnail.png
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "voice"
SCRIPT = ROOT / "configs" / "voice_script.txt"
VOICE_ID = "FGY2WhTYpPnrIDTdsKH5"  # Laura, the channel narrator
SEED = 21
STABILITY = 0.4

NUMBERS = {"re-recorded": "RE-RECORDED"}
KEEP: dict[str, set[str]] = {}
ACCENT = {"NOBODY", "ONCE", "EX", "THAT", "SKULL", "BASS", "BOOST", "AIR", "NASAL", "FILTER", "TWIST", "SECRETLY",
          "MORE", "ATTRACTIVE", "UNFILTERED", "HATE", "RE-RECORDED"}

_painter = None


def _init(captions: bool = False) -> None:
    global _painter
    from fc_sat.voice.cues import load
    from fc_sat.voice.draw import Painter
    from fc_sat.rule37.captions import Captions

    cues = load(OUT / "vo.words.json")
    cap = Captions(OUT / "vo.words.json", cues.end, NUMBERS, ACCENT, KEEP) if captions else None
    _painter = Painter(cues, cap)


def _frame(i: int) -> np.ndarray:
    return _painter.render(i)


def voice() -> None:
    from fc_sat.rule37.vo_edit import build
    from fc_sat.rule37.voice import synth

    text = SCRIPT.read_text().strip().replace("\n", " ")
    mp3, meta = synth(text, VOICE_ID, stability=STABILITY, seed=SEED)
    z, words = build(mp3, meta, OUT / "vo.wav")
    print(f"vo {len(z) / 48000:.2f}s, {len(words)} words")


def stills(frames: list[int] | None, captions: bool) -> None:
    from fc_sat.voice.cues import load

    cues = load(OUT / "vo.words.json")
    _init(captions)
    folder = OUT / "stills"
    folder.mkdir(parents=True, exist_ok=True)
    picks = frames or list(range(0, cues.n_frames, 45))
    thumbs = []
    for i in picks:
        img = Image.fromarray(_frame(i)[:, :, ::-1])
        if frames:
            img.save(folder / f"f{i:04d}.png")
        thumbs.append(img.resize((216, 384), Image.Resampling.BOX))
    cols = 10
    sheet = Image.new("RGB", (216 * cols, 384 * ((len(thumbs) + cols - 1) // cols)))
    for k, th in enumerate(thumbs):
        sheet.paste(th, ((k % cols) * 216, (k // cols) * 384))
    sheet.save(OUT / "contact.jpg", quality=85)
    print(f"wrote {OUT / 'contact.jpg'} ({len(thumbs)} frames)")


def full(out: Path, captions: bool) -> None:
    from fc_sat.encode import pipe_raw_bgr
    from fc_sat.voice.audio import render_mix
    from fc_sat.voice.cues import FPS, load

    cues = load(OUT / "vo.words.json")
    wav = OUT / "mix.wav"
    render_mix(cues, OUT / "vo.wav", wav)
    with mp.get_context("spawn").Pool(max(1, mp.cpu_count() - 1), initializer=_init, initargs=(captions,)) as pool:
        it = pool.imap(_frame, range(cues.n_frames), chunksize=8)
        pipe_raw_bgr(it, out, width=1080, height=1920, fps=FPS, audio_path=wav, n_frames=cues.n_frames, pool=pool)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["voice", "stills", "full", "thumb"])
    ap.add_argument("--frames", default="")
    ap.add_argument("--out", default=str(OUT / "voice.mp4"))
    ap.add_argument("--captions", action="store_true")
    a = ap.parse_args()
    if a.mode == "voice":
        voice()
    elif a.mode == "stills":
        stills([int(x) for x in a.frames.split(",")] if a.frames else None, a.captions)
    elif a.mode == "thumb":
        from fc_sat.voice.cues import load
        from fc_sat.voice.draw import Painter
        from fc_sat.voice.thumb import render

        render(Painter(load(OUT / "vo.words.json")), OUT / "thumbnail.png")
    else:
        full(Path(a.out), a.captions)


if __name__ == "__main__":
    main()
