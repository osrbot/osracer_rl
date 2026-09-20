"""Package verified fast-entry native videos and a precise speed/slip report."""
import argparse,json,subprocess,hashlib
import numpy as np
from hairpin_fast_common import OUT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial-slip',action='store_true',
        help='Package explicitly incomplete drift results; all driving and evidence checks must still pass')
    args=parser.parse_args()
    verify=json.loads((OUT/'verification.json').read_text())
    failed=[name for name,passed in verify['checks'].items() if not passed]
    if not verify['passed']:
        allowed=set()
        for engine,data in verify['engines'].items():
            for seed,result in data['traces'].items():
                failures=[name for name,passed in result['checks'].items() if not passed]
                if failures==['qualifying_drift_at_least_point1_s']:
                    allowed.add(f'{engine}_seed{seed}_independent_success')
        if (not args.allow_partial_slip or verify['errors'] or not failed
                or not set(failed).issubset(allowed)):
            raise RuntimeError('High-speed evidence has not passed verification')
    reports={e:json.loads((OUT/f'{e}_evaluation.json').read_text()) for e in ('isaac','mujoco')}
    duration=max(float(verify['engines'][e]['video_probe']['duration']) for e in reports)
    video=OUT/'sim2sim_fast_comparison.mp4'
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error',
        '-i',str(OUT/'isaac_fast_hairpin.mp4'),'-i',str(OUT/'mujoco_fast_hairpin.mp4'),
        '-filter_complex','[0:v]scale=960:540,setsar=1,tpad=stop_mode=clone:stop_duration=10[a];'
        '[1:v]scale=960:540,setsar=1,tpad=stop_mode=clone:stop_duration=10[b];[a][b]hstack=inputs=2[v]',
        '-map','[v]','-t',str(duration),'-an','-r','30','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    slow=OUT/'sim2sim_fast_slowmotion.mp4'
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(video),
        '-vf',"setpts=4*PTS,drawbox=x=0:y=32:w=iw:h=30:color=0x0f1622:t=fill,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='0.25x SLOW MOTION | Timestamps show simulation time':x=20:y=37:fontsize=18:fontcolor=white",
        '-an','-r','30','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(slow)],check=True)
    media=[]
    for file in (video,slow):
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(file),'-f','null','-'],check=True)
        media.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),decode_verified=True))
    (OUT/'comparison.json').write_text(json.dumps(media,indent=2))
    lines=['# NEORACER 高速接近 + 180° 侧滑回头弯','',
        '[正常速度并排视频](sim2sim_fast_comparison.mp4) · [0.25 倍速观察侧滑](sim2sim_fast_slowmotion.mp4)','',
        '同一策略在 Isaac Sim 6.0.1 中训练后直接用于 MuJoCo。策略目标是以至少 5 m/s 接近弯道，再制动、利用后轴侧滑通过 180° 回头弯并加速驶出。弯道中心线半径仍为 0.8 m；前后直道各 6 m，道路宽 1.2 m。','',
        '**5–10 m/s 是接近弯道的目标范围，不是弯心速度。** 下面将制动前峰值、弯道入口和弯心实测速度分别列出，数值来自后轴参考点的真实平移速度。','',
        '| 引擎 | 最终验收 | 接近峰值 | 弯道入口速度 | 弯心速度范围 | 后轴侧滑峰值 |',
        '|---|---:|---:|---:|---:|---:|']
    for e,r in reports.items():
        x=verify['engines'][e]['recorded'];a,b=x['apex_speed_range_m_s']
        episodes=r['episodes']
        lines.append(f"| {e} | {sum(t['success'] for t in episodes)}/{len(episodes)} | {x['entry_peak_m_s']:.2f} m/s | "
            f"{x['speed_at_turn_entry_x6_progress6_m_s']:.2f} m/s | {a:.2f}–{b:.2f} m/s | {abs(np.degrees(x['minimum_rear_slip_rad'])):.1f}° |")
    lines += ['', '表中速度与侧滑来自所录制的名义回合；成功数覆盖一个名义条件和九个未参与训练的初始位姿扰动。验收要求完成弯道、制动前峰值至少 5 m/s，以及速度大于 1.2 m/s 时后轴过度转向侧滑超过约 20°，累计至少 0.1 秒。','']
    if not verify['passed']:
        lines += ['**完整侧滑验收尚未全部通过。** 两端各 10/10 完成高速接近和掉头；Isaac 侧滑验收 10/10，MuJoCo 9/10。MuJoCo 种子 301 的峰值侧滑仅约 19.6°，合格时长为 0，保留为失败回合。`verification.json` 保持 `passed=false`；未更改验收阈值或隐藏失败。所展示的是种子 0 名义回合，两端均通过。','']
    lines += [
        '两端另做的零转向加速探针均达到约 10 m/s。这只验证了模型的速度能力，尚未证明能以 10 m/s 通过这个小半径回头弯。按 μ≈1 的平面摩擦近似，持续 5 m/s 转弯需要约 2.55 m 半径，10 m/s 需要约 10.19 m；漂移也不能消除所需的横向力。','',
        '本轮采用明确但未经过实车测量的每个前轮 ±0.45 rad 转向限位；对应低速无侧滑 Ackermann 理论半径约 0.701 m。后轴侧滑段的轨迹圆拟合如下，属于短时轨迹拟合，不代表整段都是恒定半径，也不代表测出了实车最小半径：','']
    for e in reports:
        fits=verify['engines'][e]['recorded']['circle_fits']
        good=[f for f in fits if f['reliable']]
        if good:
            f=min(good,key=lambda x:x['radius_m'])
            lines.append(f"- {e}：{f['radius_m']:.3f} m，拟合径向 RMS {1000*f['rms_radial_residual_m']:.1f} mm，覆盖 {np.degrees(f['angle_coverage_rad']):.1f}°。")
        else:lines.append(f'- {e}：没有满足本轮可靠性门槛的侧滑段圆拟合。')
    lines += ['', '修正项：Isaac 关节最大角速度与车轮刚体最大角速度均显式配置为 30000 度/秒；物理更新 480 Hz，策略控制 60 Hz。侧滑使用后轴参考点速度，避免将质心在普通转弯中的横向速度误报为甩尾。', '',
        '采用 CEM 优化参数化闭环策略。前后轮可独立给速度目标是本轮的模拟执行器假设；定位、路径、轮速和转向角来自仿真。保留原始 CAD 质量、惯量和视觉资产，轮胎使用圆柱刚体接触模型。两端共享参数，无 MuJoCo 策略训练；该验证尚不覆盖实车电机、差速器和轮胎辨识。','',
        '开发记录：原朝向门控版本在 MuJoCo 的侧滑验收先后为 2/10、7/10、8/10，证据保存在 attempt1–3。最终 policy_version=2 改用一次触发的侧滑建立—保持—恢复状态机，并根据实际侧滑角调节后轴相对轮速；在 Isaac 中重新训练，最终测试使用新种子 0、300–308。正式验收门槛始终为约 20°、0.1 秒。MuJoCo 曾用于开发验收，因此不声称它是开发过程完全未见的引擎。','',
        '[Isaac 原生视频](isaac_fast_hairpin.mp4) · [MuJoCo 原生视频](mujoco_fast_hairpin.mp4) · [策略](policy.json) · [完整验收](verification.json) · [训练历史](training_history.json) · [复现说明](../../scripts/README_hairpin_fast.md)','',
        f"最终检查点 SHA256：`{verify['checkpoint_sha256']}`。正常速度视频为 1 倍播放；并排视频中先完成的一侧保持最后一帧。",'']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(video)

if __name__=='__main__':main()
