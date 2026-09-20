#!/usr/bin/env python3
"""Record the unchanged OSRACER MJCF with native MuJoCo EGL rendering.
Run: MUJOCO_GL=egl python3 scripts/record_mujoco.py
Uses installed MuJoCo, or the existing local uv wheel cache; downloads nothing.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, glob, json, math, argparse, subprocess, hashlib
from pathlib import Path
# Reuse cached wheels when this host's system Python has no MuJoCo installation.
for package in ('mujoco', 'OpenGL'):
    try:
        __import__(package)
    except ImportError:
        candidates = glob.glob(str(Path.home()/'.cache/uv/archive-v0'/('*')/package))
        if candidates:
            sys.path.insert(0, str(Path(candidates[0]).parent))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'output/usability/mujoco');p.add_argument('--seconds',type=float,default=18);args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
asset=ROOT/'OSRACER/MuJoCo/osracer_description'
report={'engine':'MuJoCo','version':mujoco.__version__,'python':sys.executable,'render_backend':os.environ['MUJOCO_GL'],'source':str(asset),'source_files_sha256':{},'original_load_and_step':{},'limitations':['Fixed base: no free joint, so this asset cannot drive through the world.','No floor/world geometry; contact behavior and vehicle driving are not validated.','Six source actuators exist, but the stock controls do not reliably track targets in the short test. Controller tuning and contact diagnosis remain necessary.','The articulation segment uses explicitly labeled qpos + mj_forward kinematics; it does not demonstrate actuator tracking.','Finite short-run state and zero numerical warnings do not establish long-horizon physical fidelity.'],'presentation_changes':['In-memory offscreen buffer resolution and lighting only; source assets unchanged.','Collision geom group 3 is hidden from rendering; all physics collision geoms remain enabled.','Camera and video text are presentation overlays.']}
for filename in ('robot.xml','scene.xml'):
    path=asset/filename;report['source_files_sha256'][filename]=hashlib.sha256(path.read_bytes()).hexdigest()
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    for _ in range(500):mujoco.mj_step(model,data)
    report['original_load_and_step'][filename]={'loaded':True,'steps':500,'simulated_seconds':float(data.time),'finite_state':bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),'warnings':data.warning.number.tolist(),'final_qpos':data.qpos.tolist(),'contacts':int(data.ncon)}
m=mujoco.MjModel.from_xml_path(str(asset/'scene.xml'));d=mujoco.MjData(m)
names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(m.njnt)]
anames=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,j) for j in range(m.nu)]
report['model']={'bodies_including_world':m.nbody,'joints':m.njnt,'nq':m.nq,'nv':m.nv,'actuators':m.nu,'geoms':m.ngeom,'meshes':m.nmesh,'sensors':m.nsensor,'timestep':m.opt.timestep,'joint_names':names,'actuator_names':anames,'root_free_joint':False}
# Quantify a real, unchanged-model actuator test independently of the pose demo.
vel=[]
for i in range(1000):
    for a,n in enumerate(anames):d.ctrl[a]=0.3 if 'steering' in n else (-1.2 if 'right_rear' in n else 1.2)
    mujoco.mj_step(m,d);vel.append(d.qvel.copy())
report['original_control_test']={'steps':1000,'simulated_seconds':float(d.time),'controls':d.ctrl.tolist(),'final_qpos':d.qpos.tolist(),'final_qvel':d.qvel.tolist(),'last_half_second_mean_qvel':np.mean(vel[-250:],axis=0).tolist(),'max_abs_qvel':float(np.max(np.abs(vel))),'warnings':d.warning.number.tolist(),'finite_state':bool(np.isfinite(vel).all()),'tracking_validation':'not passed: significant target discrepancies; not a working driving controller'}
m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
m.vis.headlight.ambient[:]=.4;m.vis.headlight.diffuse[:]=.8
renderer=mujoco.Renderer(m,height=720,width=1280)
option=mujoco.MjvOption();option.geomgroup[3]=0
cam=mujoco.MjvCamera();cam.lookat[:]=[.14,0,.055];cam.distance=.8
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
f=ImageFont.truetype(font,22);small=ImageFont.truetype(font,17);big=ImageFont.truetype(font,32)
video=args.out/'osracer_mujoco_usability.mp4'
cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
frames=int(args.seconds*30);mujoco.mj_resetData(m,d);dynamic_started=False
for i in range(frames):
    t=i/30;phase=t/args.seconds
    if phase<1/3:
        d.qpos[:]=0;mujoco.mj_forward(m,d)
        title='01 / SOURCE MODEL  -  MULTI-ANGLE INSPECTION';line='Native MuJoCo render  |  Source materials and geometry';cam.azimuth=135+phase*540;cam.elevation=-24
    elif phase<7/9:
        tau=(phase-1/3)*args.seconds
        for j,n in enumerate(names):d.qpos[m.jnt_qposadr[j]]=.45*math.sin(tau*1.7) if 'steering' in n else tau*2*(-1 if 'right_rear' in n else 1)
        mujoco.mj_forward(m,d)
        title='02 / ARTICULATION  -  KINEMATIC DEMONSTRATION';line='qpos + mj_forward  |  Steering +/- 0.45 rad  |  Wheel rotation';cam.azimuth=35+tau*14;cam.elevation=-28
    else:
        if not dynamic_started:mujoco.mj_resetData(m,d);dynamic_started=True
        target=(phase-7/9)*args.seconds
        while d.time<target:
            for a,n in enumerate(anames):d.ctrl[a]=.3 if 'steering' in n else (-1.2 if 'right_rear' in n else 1.2)
            mujoco.mj_step(m,d)
        title='03 / ORIGINAL ACTUATORS  -  SHORT DYNAMICS TEST';line='mj_step + source actuators  |  Target tracking requires tuning';cam.azimuth=55;cam.elevation=-25
    renderer.update_scene(d,camera=cam,scene_option=option)
    im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,1280,119),fill=(13,20,30));draw.text((38,20),'OSRACER  /  MuJoCo',font=big,fill=(238,244,250));draw.text((40,74),title,font=f,fill=(92,216,213))
    draw.rectangle((0,608,1280,720),fill=(13,20,30));draw.text((40,623),line,font=f,fill=(235,239,245));draw.text((40,660),'6 joints  /  6 actuators  /  fixed base  /  no floor  |  Not a vehicle-driving validation',font=small,fill=(168,180,196))
    draw.rectangle((0,712,int(1280*(i+1)/frames),720),fill=(92,216,213))
    proc.stdin.write(im.tobytes())
    if i in (60,270,480):im.save(args.out/f'preview_{i:04d}.png')
    if i%90==0:print(f'Rendered {i}/{frames}',flush=True)
proc.stdin.close();rc=proc.wait();renderer.close()
if rc:raise RuntimeError(f'ffmpeg failed: {rc}')
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
report['video']={'path':str(video),'width':1280,'height':720,'fps':30,'frames':frames,'duration_seconds':args.seconds,'segments':[{'seconds':[0,args.seconds/3],'mode':'static source orbit'},{'seconds':[args.seconds/3,args.seconds*7/9],'mode':'explicit kinematic articulation'},{'seconds':[args.seconds*7/9,args.seconds],'mode':'original actuator dynamics; tracking not validated'}],'ffprobe':probe,'decode_exit_code':decode.returncode,'decode_stderr':decode.stderr}
report['source_files_unchanged']=all(hashlib.sha256((asset/n).read_bytes()).hexdigest()==s for n,s in report['source_files_sha256'].items())
(args.out/'report.json').write_text(json.dumps(report,indent=2))
assert decode.returncode == 0, f'Video decode failed: {decode.stderr}'
assert report['source_files_unchanged'], 'Source XML changed during recording'
assert all(v['finite_state'] for v in report['original_load_and_step'].values()), 'Nonfinite zero-control state'
assert report['original_control_test']['finite_state'], 'Nonfinite original-control state'
print(json.dumps({'video':str(video),'report':str(args.out/'report.json'),'decode_exit_code':decode.returncode}))
