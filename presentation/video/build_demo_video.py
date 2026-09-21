#!/usr/bin/env python3
"""Build a subtitle-burned, male-narrated OSRACER technical walkthrough."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "presentation/video/build"
WORK = OUT / "work"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
MPL_FONT = FontProperties(fname=FONT)
W, H, FPS = 1280, 720, 30

SEGMENTS = [
    ("title", "OSRACER：从训练到双引擎赛道验证", "证据驱动的技术分享演示",
     "欢迎来到 OSRACER 技术汇报。本次分享围绕一个清晰的问题展开：在约束明确的观测和物理条件下，策略究竟已经具备哪些能力？我们将从训练设计、固定验证条件，到 MuJoCo 和 Isaac Sim 的逐赛道结果，说明已有证据，也说明尚未解决的边界。"),
    ("contract", "先固定实验契约，再谈策略好坏", "控制变量：观测、赛道集合、种子、验证器和引擎口径",
     "首先固定实验契约。策略只接收轮速、转角和十五赫兹单线激光，不使用全局定位，也不依赖轨迹回放。资格验证固定为二十四条 RC 赛道、双车起始条件和同一审计规则。MuJoCo 与 Isaac Sim 分别统计，七点二五和九米每秒也分别统计，这样每一个比较都有明确前提。"),
    ("assets", "资产层验证：先证明它能进入目标引擎", "OpenUSD / Isaac Sim 与 MuJoCo",
     "进入训练之前，先验证资产能否在目标引擎中正确工作。OpenUSD 在 Isaac Sim 中完成网格、刚体、关节和短程 PhysX 步进；MuJoCo 的 robot.xml 与 scene.xml 各完成五百步，并且没有数值警告。这证明的是资产层可用性，而不是固定基座模型已经具备可靠驾驶能力。"),
    ("exporter", "SolidWorks URDF Exporter Pro：CAD 导出到仿真", "面向 OpenUSD、MJCF、URDF 与 ROS 工作流",
     "OSRACER 的导出链路使用 SolidWorks URDF Exporter Pro。它面向 SolidWorks 装配体，支持面向 OpenUSD、MJCF、URDF 和 ROS 的多目标导出，并保留关节、坐标系、惯量、命名和元数据等工程信息。工具还支持外观与碰撞几何的选择、网格简化和文件大小反馈，帮助开发者更快把 CAD 资产带入仿真。我们也欢迎需要 OpenUSD、MJCF 或 URDF 的开发者试用它；使用中遇到问题请提交 Issue，欢迎补充测试、文档和代码，一起让工具服务更多开发者。"),
    ("iteration", "策略迭代：固定资格协议下的 MuJoCo 通过圈数", "版本链，而不是伪装成单因子因果实验",
     "现在看策略迭代。图中只放入同一开发赛道集合和相同审计口径下的 MuJoCo 批次，因此纵轴是二十四条赛道中，同时满足有效整圈和有效超车的数量。v7 到 v10 逐步加入脱困、近场处理和有界侧向避让。需要强调的是，这是一条工程版本链，不是单因素因果实验；每次策略调整后都重新运行完整资格集。"),
    ("matrix", "冻结 v10c：24 条赛道逐项对照", "同一策略版本，双引擎分别审计",
     "冻结到 v10c 后，MuJoCo 的二十四条赛道全部完成有效整圈和一次有效超车。Isaac Sim 在相同的七点二五米每秒配置下有二十二条有效。剩余两条的原因也明确保留：Spa 是起步阶段的车辆间接触，Suzuka 是桥面区域的车辆姿态失稳。矩阵让每条赛道都保持可见，而不是只展示 Bahrain 的一个代表样例。"),
    ("speed", "控制变量示例：只提高巡航速度", "v10c、赛道集合与验证器不变",
     "这里给出一个真正的控制变量示例：保持 v10c 策略、赛道集合和审计器不变，只把巡航速度从七点二五提高到九米每秒。MuJoCo 仍然二十四条全部有效，且所有单圈更快，峰值为八点九七米每秒。Isaac Sim 则从二十二条降为二十条，新增 Austin 和 Las Vegas 的停滞。当前证据表明，七点二五米每秒仍是共享配置在 Isaac Sim 中更稳妥的工作点。"),
    ("race", "原生录像：Bahrain 双引擎闭环运行", "MuJoCo（左）与 Isaac Sim（右）",
     "这是 Bahrain 的原生闭环录像。左右两侧分别对应 MuJoCo 和 Isaac Sim，画面展示策略输入、决策、安全约束和与对手车辆的交互过程。是否通过并不由画面单独决定，而由完整轨迹审计判定。有效超车需要完整的相对位置变化，瞬时名次波动不会被计入结果。"),
    ("boundary", "漂移和主动观测：展示 A/B，也展示失败", "成立条件和失败边界必须同时可见",
     "回头弯漂移同样需要对照实验。后轮增速设置为三点五时，最大侧滑达到二十七度；设置为零后只剩四点三六度。因此，当前漂移动作依赖后轮增速，不能直接外推到单电机四驱实车。再看主动观测：加入二厘米噪声、百分之五丢束和五十毫秒延迟后，组合感知扰动十次中十次失败。这些失败记录为下一轮感知与策略改进提供了具体目标。"),
    ("close", "如何在现场直接复演", "从训练记录到原生录像，再到逐赛道审计",
     "现场复演时，先固定 checkpoint、策略源码哈希、赛道版本、引擎和种子；随后播放单赛道原生录像，并查看对应轨迹和验证记录；最后再运行全赛道资格，而不是用一个样例代替统计。下一步将聚焦噪声条件下的扫描匹配、Isaac 桥接触、实车转向反馈，以及与漂移目标相匹配的动作硬件。"),
]


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def probe(path: Path) -> float:
    result = subprocess.run(
        ("ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)),
        capture_output=True, check=True, text=True,
    )
    return float(result.stdout.strip())


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def base() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), "#10283e")
    draw = ImageDraw.Draw(image)
    for y in range(H):
        color = (14, 29 + int(y * 25 / H), 46 + int(y * 65 / H))
        draw.line((0, y, W, y), fill=color)
    draw.rectangle((0, 0, W, 72), fill="#091722")
    draw.text((44, 20), "OSRACER", font=f(31, True), fill="#7ae4d8")
    draw.text((242, 25), "TECHNICAL WALKTHROUGH", font=f(22), fill="#edf5f8")
    return image, draw


def wrap(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, size: int, color: str) -> int:
    line = ""
    for char in text:
        candidate = line + char
        if draw.textlength(candidate, font=f(size)) > width and line:
            draw.text((x, y), line, font=f(size), fill=color)
            y += size + 12
            line = char
        else:
            line = candidate
    if line:
        draw.text((x, y), line, font=f(size), fill=color)
        y += size + 12
    return y


def card(path: Path, title: str, subtitle: str, lines: list[str]) -> None:
    image, draw = base()
    y = wrap(draw, title, 70, 126, 1100, 46, "white") + 16
    y = wrap(draw, subtitle, 70, y, 1100, 26, "#82e2d8") + 30
    for item in lines:
        draw.ellipse((78, y + 11, 91, y + 24), fill="#eb7e58")
        y = wrap(draw, item, 114, y, 1030, 29, "#eff6f9") + 21
    draw.text((70, 666), "Evidence-led autonomous racing · conditions and failure boundaries stay visible", font=f(17), fill="#afc7d7")
    image.save(path)


def asset_card(path: Path) -> None:
    image, draw = base()
    draw.text((70, 115), "OpenUSD / Isaac Sim 与 MuJoCo 的资产层验证", font=f(39, True), fill="white")
    entries = [
        ("site/source_assets/openusd-preview.png", "OpenUSD / Isaac Sim 6.0.1", "20 meshes · 10 rigid bodies · 9 joints"),
        ("site/source_assets/mujoco-preview.png", "MuJoCo 3.10.0", "robot.xml / scene.xml · 500 steps"),
    ]
    for index, (rel, label, note) in enumerate(entries):
        tile = Image.open(ROOT / rel).convert("RGB")
        tile.thumbnail((535, 380))
        x = 70 + index * 575
        draw.rounded_rectangle((x - 6, 184, x + 541, 570), radius=15, fill="#07121d", outline="#4188b0", width=2)
        image.paste(tile, (x + (535 - tile.width) // 2, 187 + (377 - tile.height) // 2))
        draw.text((x, 604), label, font=f(26, True), fill="#82e2d8")
        draw.text((x, 646), note, font=f(18), fill="#edf5f8")
    image.save(path)


def records(name: str) -> list[dict]:
    return json.loads((ROOT / "output/racing" / name).read_text(encoding="utf-8"))


def iteration(path: Path) -> None:
    items = [("v7", "c03v7_escape_dev24_summary.json"), ("v7b", "c03v7b_escape_dev24_summary.json"),
             ("v7c", "c03v7c_escape_dev24_summary.json"), ("v8a", "c03v8a_escape_dev24_summary.json"),
             ("v8c", "c03v8c_escape_dev24_summary.json"), ("v9f", "c03v9f_dev24_summary.json"),
             ("v9n", "c03v9n_dev24_summary.json"), ("v10a", "c03v10a_dev24_summary.json"),
             ("v10b", "c03v10b_dev24_summary.json"), ("v10c", "c03v10c_dev24_summary.json")]
    values = [sum(row["valid_lap"] for row in records(file)) for _, file in items]
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="#10283e")
    ax.set_facecolor("#10283e")
    ax.plot([x[0] for x in items], values, "-o", lw=3, ms=9, color="#72ded3")
    ax.fill_between(range(len(values)), values, color="#2578bd", alpha=.25)
    ax.set_ylim(0, 25); ax.set_yticks(range(0, 25, 4)); ax.grid(axis="y", alpha=.3)
    ax.set_title("MuJoCo 策略版本链：固定开发资格协议", color="white", fontsize=23, fontweight="bold", pad=20, fontproperties=MPL_FONT)
    ax.set_ylabel("有效整圈 + 有效超车（24 条赛道）", color="white", fontsize=14, fontproperties=MPL_FONT)
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_color("#6c8da5")
    for i, value in enumerate(values): ax.text(i, value + .6, str(value) + "/24", ha="center", color="white", fontweight="bold")
    ax.text(.01, -.14, "版本链含多项策略与安全层改动；每次重新执行资格集，不作为单因子因果宣称。", transform=ax.transAxes, color="#b7cfdd", fontsize=12, fontproperties=MPL_FONT)
    fig.tight_layout(); fig.savefig(path, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


def engine_records(engine: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for file in (ROOT / "output/racing" / engine).glob("*_c03v10c_dev24.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            result[data[0]["track"]] = data[0]
    return result


def matrix(path: Path) -> None:
    mu, isaac = engine_records("mujoco"), engine_records("isaac")
    tracks = sorted(set(mu) | set(isaac))
    values = [[int(mu.get(t, {}).get("valid_lap", False)), int(isaac.get(t, {}).get("valid_lap", False))] for t in tracks]
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="#10283e")
    ax.set_facecolor("#10283e")
    ax.imshow(values, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#ca5a50", "#1aa78f"]), vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["MuJoCo", "Isaac Sim"], color="white", fontsize=15, fontweight="bold")
    ax.set_yticks(range(len(tracks)), [t.replace("_", " ").title() for t in tracks], color="white", fontsize=10, fontproperties=MPL_FONT)
    ax.set_title("冻结 v10c · 7.25 m/s · 24 条赛道逐项资格结果", color="white", fontsize=22, fontweight="bold", pad=20, fontproperties=MPL_FONT)
    for row, track in enumerate(tracks):
        for col, data in enumerate((mu.get(track, {}), isaac.get(track, {}))):
            label = "PASS" if data.get("valid_lap") else (data.get("failure") or "FAIL").upper()
            ax.text(col, row, label, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(.01, -.08, "绿：有效整圈 + 有效超车；红：保留失败原因。", transform=ax.transAxes, color="#b7cfdd", fontsize=12, fontproperties=MPL_FONT)
    fig.tight_layout(); fig.savefig(path, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


def speed(path: Path) -> None:
    x = [0, 1]
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="#10283e")
    ax.set_facecolor("#10283e")
    a = ax.bar([i - .18 for i in x], [24, 22], .36, color="#72ded3", label="7.25 m/s")
    b = ax.bar([i + .18 for i in x], [24, 20], .36, color="#eb8058", label="9.0 m/s")
    ax.set_ylim(0, 25); ax.set_xticks(x, ["MuJoCo", "Isaac Sim"], color="white", fontsize=17, fontweight="bold"); ax.tick_params(colors="white")
    ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
    ax.set_title("v10c 的速度控制变量：只有巡航速度改变", color="white", fontsize=23, fontweight="bold", pad=20, fontproperties=MPL_FONT)
    ax.legend(facecolor="#173249", labelcolor="white", fontsize=14)
    for bars in (a, b):
        for bar in bars: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.6, str(int(bar.get_height()))+"/24", ha="center", color="white", fontweight="bold", fontsize=16)
    ax.text(.01, -.12, "MuJoCo 9.0 m/s：24/24、峰值 8.97 m/s；Isaac 9.0 m/s：20/24、新增 Austin 与 Las Vegas 停滞。", transform=ax.transAxes, color="#b7cfdd", fontsize=12, fontproperties=MPL_FONT)
    fig.tight_layout(); fig.savefig(path, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


def boundary(path: Path) -> None:
    image, draw = base()
    draw.text((70, 115), "把成立条件和失败边界同时放进画面", font=f(39, True), fill="white")
    panels = [("回头弯 A/B", "后轮增速 3.5", "最大侧滑 27.0°", "#1aa78f"),
              ("实车等效动作", "后轮增速 0", "最大侧滑 4.36°", "#eb8058"),
              ("组合感知扰动", "2 cm 噪声 + 5% 丢束 + 50 ms 延迟", "10 / 10 失败", "#ca5a50")]
    for index, (title, setting, result, color) in enumerate(panels):
        x = 70 + 400 * index
        draw.rounded_rectangle((x, 205, x + 350, 535), radius=18, fill="#173249", outline=color, width=4)
        draw.text((x + 27, 242), title, font=f(28, True), fill="white")
        wrap(draw, setting, x + 27, 315, 295, 22, "#d5e4eb")
        draw.text((x + 27, 425), result, font=f(32, True), fill=color)
    draw.text((70, 628), "结论：当前漂移依赖后轮增速；扫描匹配在组合扰动下失效。", font=f(23), fill="#edf5f8")
    image.save(path)


def tts(name: str, text: str) -> tuple[Path, float]:
    executable = os.environ.get("EDGE_TTS") or shutil.which("edge-tts")
    if not executable: raise RuntimeError("Set EDGE_TTS to an edge-tts executable")
    source, target = WORK / (name + ".txt"), WORK / (name + ".mp3")
    source.write_text(text, encoding="utf-8")
    run(executable, "--voice", "zh-CN-YunyangNeural", "--rate=-8%", "--file", str(source), "--write-media", str(target))
    return target, probe(target)


def still(image: Path, output: Path, seconds: float) -> None:
    run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(image), "-t", "{:.3f}".format(seconds),
        "-vf", "scale={}:{}".format(W, H), "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(output))


def split(output: Path, seconds: float) -> None:
    left, right = ROOT / "site/source_assets/mujoco-bahrain.mp4", ROOT / "site/source_assets/isaac-bahrain.mp4"
    graph = ("[0:v]scale=620:600:force_original_aspect_ratio=decrease,pad=620:600:(ow-iw)/2:(oh-ih)/2:color=0x08131f[l];"
             "[1:v]scale=620:600:force_original_aspect_ratio=decrease,pad=620:600:(ow-iw)/2:(oh-ih)/2:color=0x08131f[r];"
             "[l][r]hstack,pad=1280:720:30:90:color=0x08131f,drawbox=x=0:y=0:w=1280:h=70:color=0x091722:t=fill,"
             "drawtext=fontfile=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc:text='OSRACER | Bahrain representative closed-loop replay':x=35:y=20:fontsize=25:fontcolor=white,"
             "drawtext=fontfile=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc:text='MuJoCo':x=270:y=655:fontsize=25:fontcolor=0x80e2d8,"
             "drawtext=fontfile=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc:text='Isaac Sim':x=894:y=655:fontsize=25:fontcolor=0x80e2d8,format=yuv420p")
    run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-stream_loop", "-1", "-i", str(left), "-stream_loop", "-1", "-i", str(right),
        "-t", "{:.3f}".format(seconds), "-filter_complex", graph, "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "20", str(output))


def subtitles(lengths: dict[str, float]) -> Path:
    def stamp(value: float) -> str:
        ms = round(value * 1000); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
        return "{:02}:{:02}:{:02},{:03}".format(h, m, s, ms)
    cursor, number, lines = 0.0, 1, []
    for name, _, _, text in SEGMENTS:
        parts = [part.strip() for part in text.replace("，", "，|").replace("。", "。|").split("|") if part.strip()]
        weight = sum(len(part) for part in parts)
        for part in parts:
            span = lengths[name] * len(part) / weight
            lines.extend((str(number), "{} --> {}".format(stamp(cursor), stamp(cursor + span)), part, ""))
            cursor += span; number += 1
    output = WORK / "burned_in_subtitles.srt"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    if OUT.exists(): shutil.rmtree(OUT)
    WORK.mkdir(parents=True)
    audio, lengths = [], {}
    for name, _, _, text in SEGMENTS:
        path, length = tts(name, text); audio.append(path); lengths[name] = length; print(name, "{:.2f}s".format(length))
    videos = []
    for name, title, subtitle, _ in SEGMENTS:
        image, video = WORK / (name + ".png"), WORK / (name + ".mp4")
        if name == "assets": asset_card(image)
        elif name == "iteration": iteration(image)
        elif name == "matrix": matrix(image)
        elif name == "speed": speed(image)
        elif name == "boundary": boundary(image)
        elif name == "race": split(video, lengths[name]); videos.append(video); continue
        elif name == "contract": card(image, title, subtitle, ["高层策略：轮速、转角、15 Hz 单线激光；禁止全局位置与轨迹回放。",
                                                       "资格协议：24 条 RC 赛道、双车条件、有效整圈与有效超车审计。",
                                                       "引擎与巡航速度是显式变量，分开统计。"])
        elif name == "exporter": card(image, title, subtitle, ["从 SolidWorks 装配体导出 OpenUSD、MJCF、URDF 与 ROS 资产。",
                                                               "保留关节、坐标系、惯量、命名与元数据；支持外观/碰撞几何选择和网格简化。",
                                                               "欢迎试用：github.com/osrbot/solidworks_urdf_exporter_pro",
                                                               "欢迎提交 Issue、测试、文档和代码贡献。"])
        elif name == "close": card(image, title, subtitle, ["固定 checkpoint、策略哈希、赛道版本、引擎和种子。",
                                                            "播放原生录像，同时保留轨迹、指标、哈希和解码检查。",
                                                            "再执行全赛道资格；把成功与失败一起展示。"])
        else: card(image, title, subtitle, ["训练策略、固定条件与双引擎验证共同组成可复现证据。", "本片所有结论均保留成立条件与失败边界。"])
        still(image, video, lengths[name]); videos.append(video)
    video_list, audio_list = WORK / "video_list.txt", WORK / "audio_list.txt"
    video_list.write_text("".join("file '{}'\n".format(path) for path in videos), encoding="utf-8")
    audio_list.write_text("".join("file '{}'\n".format(path) for path in audio), encoding="utf-8")
    silent, narration = WORK / "visuals.mp4", OUT / "osracer_training_walkthrough_zh_narration.mp3"
    run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(video_list), "-c", "copy", str(silent))
    run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(audio_list), "-c", "copy", str(narration))
    srt, final = subtitles(lengths), OUT / "osracer_training_walkthrough_zh.mp4"
    style = "FontName=Noto Sans CJK SC,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H80101B27,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=32"
    run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(narration), "-vf",
        "subtitles={}:fontsdir=/usr/share/fonts/opentype/noto:force_style='{}'".format(srt, style), "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(final))
    run("ffmpeg", "-v", "error", "-xerror", "-i", str(final), "-f", "null", "-")
    srt.unlink()
    print("rendered {} ({:.2f}s)".format(final, probe(final)))


if __name__ == "__main__":
    main()
