#!/usr/bin/env python3
"""Build a side-by-side native-video comparison and evidence-linked summary."""
import hashlib
import json
import subprocess
from hairpin_common import OUT

def main():
    verification=json.loads((OUT/'verification.json').read_text())
    if not verification['passed']:
        raise RuntimeError('Complete verify_hairpin.py successfully before packaging')
    reports={e:json.loads((OUT/f'{e}_evaluation.json').read_text()) for e in ('isaac','mujoco')}
    duration=max(r['recorded_episode']['duration_s'] for r in reports.values())
    destination=OUT/'sim2sim_comparison.mp4'
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error',
        '-i',str(OUT/'isaac_hairpin.mp4'),'-i',str(OUT/'mujoco_hairpin.mp4'),
        '-filter_complex','[0:v]scale=960:540,setsar=1,tpad=stop_mode=clone:stop_duration=10[a];'
        '[1:v]scale=960:540,setsar=1,tpad=stop_mode=clone:stop_duration=10[b];[a][b]hstack=inputs=2[v]',
        '-map','[v]','-t',str(duration),'-r','30','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],check=True)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(destination),'-f','null','-'],check=True)
    meta=dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
              duration_s=duration,width=1920,height=540,fps=30,decode_verified=True,
              presentation='Independent native simulations aligned at first control step; shorter completed run holds its last frame.')
    (OUT/'comparison.json').write_text(json.dumps(meta,indent=2))
    lines=['# OSRACER 180° 回头弯：训练与 sim2sim 结果','',
        '已在 Isaac Sim 6.0.1 / PhysX 中完成 100 个 CEM 训练回合、15 个固定条件的候选选择回合，并把同一策略直接迁移到 MuJoCo。四轮速度目标与左右转向关节位置目标以 30 Hz 更新。','',
        '[并排视频](sim2sim_comparison.mp4) · [Isaac 原生视频](isaac_hairpin.mp4) · [MuJoCo 原生视频](mujoco_hairpin.mp4)','',
        '| 引擎 | 最终评估成功 | 平均耗时 | 平均 RMS 横向误差 | 最坏横向误差 |',
        '|---|---:|---:|---:|---:|']
    for engine,r in reports.items():
        episodes=r['episodes']; n=len(episodes)
        lines.append(f"| {engine} {r['runtime_version']} | {sum(e['success'] for e in episodes)}/{n} | "
            f"{sum(e['duration_s'] for e in episodes)/n:.2f} s | "
            f"{100*sum(e['rms_cte_m'] for e in episodes)/n:.2f} cm | "
            f"{100*max(e['max_abs_cte_m'] for e in episodes):.2f} cm |")
    lines += ['', '每个引擎测试一个名义初始状态和九个未参与训练的位姿扰动（横向 ±2.5 cm、航向 ±0.03 rad）。两段录制回合也均通过，录像与不渲染的轨迹一致性检查、原始资产哈希检查和视频完整解码检查通过。','',
        '策略采用有控制先验的 CEM 参数优化，不是 PPO 神经网络。观测使用已知路径与理想车辆定位，加上轮速和转角反馈。测试范围为半径 0.8 m、平地、固定质量/摩擦的低速 180° 回头弯；尚未验证高速漂移、不同路面或实车迁移。成功表示通过终点容差，不要求停车。','',
        f"策略 SHA256：`{verification['checkpoint_sha256']}`。MuJoCo 未进行策略重训或参数选择。",'',
        '[策略参数](policy.json) · [训练历史](training_history.json) · [最终候选选择](checkpoint_selection.json) · [Isaac 指标](isaac_evaluation.json) · [MuJoCo 指标](mujoco_evaluation.json) · [完整校验](verification.json)','',
        '[实现与复现说明](../../scripts/README_hairpin.md)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(destination)

if __name__=='__main__':main()
