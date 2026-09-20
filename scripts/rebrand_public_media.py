#!/usr/bin/env python3
"""Replace only the broadcast brand text in reviewed public evidence media.

The source footage, simulation pixels, timing, and telemetry are retained.  This
script touches only the dark title band that originally carried the project
name, so the result remains an honest copy of the reviewed experiment evidence.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "site" / "source_assets"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
RACE_IMAGES = {
    "drift.png": "OSRACER | High-speed entry + 180-degree hairpin | mujoco",
    "isaac-race.png": "OSRACER | isaac | Bahrain International Circuit",
    "mujoco-race.png": "OSRACER | mujoco | Bahrain International Circuit",
    "perturb.png": "OSRACER | mujoco | Bahrain International Circuit",
}
STILL_IMAGES = {
    "mujoco-preview.png": ("mujoco", None),
}
VIDEOS = {
    "isaac-bahrain.mp4": "race_isaac",
    "mujoco-bahrain.mp4": "race_mujoco",
    "mujoco-usability.mp4": "mujoco",
    "openusd-usability.mp4": "openusd",
}


def draw_brand(image: Image.Image, profile: str, title: str | None = None) -> Image.Image:
    """Paint a same-band, same-font OSRACER label without altering evidence."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    if profile == "race":
        draw.rectangle((0, 0, canvas.width, 42), fill=(15, 22, 34))
        draw.text((18, 10), title, font=ImageFont.truetype(FONT, 22), fill="white")
    elif profile == "mujoco":
        draw.rectangle((0, 0, canvas.width, 65), fill=(13, 20, 30))
        draw.text((38, 20), "OSRACER  /  MuJoCo", font=ImageFont.truetype(FONT, 32), fill=(238, 244, 250))
    elif profile == "openusd":
        draw.rectangle((0, 0, 650, 64), fill=(13, 20, 31))
        draw.text((28, 13), "OSRACER  |  OpenUSD / Isaac Sim 6.0.1", font=ImageFont.truetype(FONT, 26), fill="white")
    else:
        raise ValueError(f"Unknown media profile: {profile}")
    return canvas


def rebrand_image(path: Path, title: str) -> None:
    with Image.open(path) as source:
        rewritten = draw_brand(source, "race", title)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as temporary:
        destination = Path(temporary.name)
    rewritten.save(destination, format="PNG", optimize=True)
    destination.replace(path)


def video_filter(profile: str) -> str:
    if profile in {"race_isaac", "race_mujoco"}:
        title = "OSRACER | isaac | Bahrain International Circuit" if profile == "race_isaac" else "OSRACER | mujoco | Bahrain International Circuit"
        return (
            "drawbox=x=0:y=0:w=1280:h=42:color=0x0f1622:t=fill,"
            f"drawtext=fontfile={FONT}:text='{title}':x=18:y=10:fontsize=22:fontcolor=white"
        )
    if profile == "mujoco":
        return (
            "drawbox=x=28:y=10:w=267:h=50:color=0x0d141e:t=fill,"
            f"drawtext=fontfile={FONT}:text='OSRACER  /  MuJoCo':x=38:y=20:fontsize=32:fontcolor=0xeef4fa"
        )
    if profile == "openusd":
        return (
            "drawbox=x=18:y=4:w=542:h=53:color=0x0d141f:t=fill,"
            f"drawtext=fontfile={FONT}:text='OSRACER  |  OpenUSD / Isaac Sim 6.0.1':x=28:y=13:fontsize=26:fontcolor=white"
        )
    raise ValueError(f"Unknown media profile: {profile}")


def rebrand_video(path: Path, profile: str) -> None:
    destination = path.with_suffix(".osracer.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
        "-map", "0:v:0", "-map", "0:a?", "-vf", video_filter(profile),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(destination), "-f", "null", "-"], check=True)
    destination.replace(path)


def main() -> None:
    for name, title in RACE_IMAGES.items():
        rebrand_image(MEDIA / name, title)
    for name, (profile, title) in STILL_IMAGES.items():
        path = MEDIA / name
        with Image.open(path) as source:
            rewritten = draw_brand(source, profile, title)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as temporary:
            destination = Path(temporary.name)
        rewritten.save(destination, format="PNG", optimize=True)
        destination.replace(path)
    for name, profile in VIDEOS.items():
        rebrand_video(MEDIA / name, profile)
    print("Rebranded 5 screenshots and 4 videos as OSRACER.")


if __name__ == "__main__":
    main()
