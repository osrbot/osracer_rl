#!/usr/bin/env python3
"""Record the supplied OpenUSD in Isaac Sim; edits stay in the session layer."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, default=root/'output/usability/openusd')
p.add_argument('--preview-only', action='store_true')
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = root/'OSRACER/USD/osracer_description/robot.usd'
report = {'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'runtime': 'Isaac Sim 6.0.1', 'status': 'started', 'fps': 30, 'resolution': [1280,720],
          'scope': 'Native USD load/render, short zero-control physics smoke, fixture-based joint position demonstration; no driving controller validation.'}
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width':1280, 'height':720,
                     'renderer':'RayTracedLighting', 'multi_gpu':False})
import numpy as np
import omni.replicator.core as rep
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import open_stage, get_current_stage
encoder = None
try:
    if not open_stage(str(source)):
        raise RuntimeError('OpenUSD entrypoint failed to open')
    stage = get_current_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    prims = list(stage.Traverse())
    meshes = [x for x in prims if x.IsA(UsdGeom.Mesh)]
    report['asset_counts'] = {'meshes':len(meshes), 'rigid_bodies':sum(x.HasAPI(UsdPhysics.RigidBodyAPI) for x in prims),
                              'physics_joints':sum(x.IsA(UsdPhysics.Joint) for x in prims)}
    report['resolved_meshes'] = sum(bool(UsdGeom.Mesh(x).GetPointsAttr().Get()) for x in meshes)
    assert report['resolved_meshes'] == 20, report
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render']).ComputeWorldBound(stage.GetPrimAtPath('/Robot')).ComputeAlignedRange()
    lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    center = (lo+hi)/2
    report['bounds_m'] = [lo.tolist(), hi.tolist()]
    print('USD_ASSET_LOADED', report['asset_counts'], 'bounds', report['bounds_m'], flush=True)
    world = World(physics_dt=1/240, rendering_dt=1/30, stage_units_in_meters=1.0)
    robot = world.scene.add(SingleArticulation(prim_path='/Robot', name='osracer'))
    world.reset()
    start_pos, _ = robot.get_world_pose()
    for _ in range(60):
        world.step(render=False)
    end_pos, end_quat = robot.get_world_pose()
    q = robot.get_joint_positions()
    report['native_physics_smoke'] = {'steps':60, 'duration_s':0.25, 'finite':bool(np.isfinite(np.r_[end_pos,end_quat,q]).all()),
                                    'start_position':start_pos.tolist(), 'end_position':end_pos.tolist(), 'dof_names':list(robot.dof_names),
                                    'note':'Original source has no floor/world fixture; gravity is active. This is a short smoke test only.'}
    assert report['native_physics_smoke']['finite']
    print('PHYSICS_SMOKE', report['native_physics_smoke'], flush=True)
    world.stop()
    world.reset()
    world.pause()
    # Reset restores source transforms before fixture-only articulation initialization.
    anchor = UsdPhysics.FixedJoint.Define(stage, '/Demo/BaseFixture')
    anchor.CreateBody1Rel().SetTargets(['/Robot/Links/base_link'])
    world.reset()
    world.pause()
    robot.set_joint_positions(np.zeros(robot.num_dof, dtype=np.float32))
    report['limitations'] = ['PhysX reports that the 10 dynamic triangle-mesh colliders are unsupported and falls back to convexHull; original collision attributes are unchanged.', 'Source robotType metadata is quadruped although the asset is a car.', 'Wheel velocity drives have no authored damping; this recording does not validate wheel drive tracking.']
    report['demo_setup'] = {'fixture':'World fixed joint added only to session layer',
                            'motion':'Joint positions assigned directly; kinematic articulation demonstration, physics paused.'}
    camera = UsdGeom.Camera.Define(stage, '/Demo/Camera')
    camera.CreateFocalLengthAttr(35)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    camera_op = UsdGeom.Xformable(camera).AddTransformOp()
    dome = UsdLux.DomeLight.Define(stage, '/Demo/Dome')
    dome.CreateIntensityAttr(500)
    dome.CreateColorAttr(Gf.Vec3f(0.85,0.91,1.0))
    key = UsdLux.DistantLight.Define(stage, '/Demo/Key')
    key.CreateIntensityAttr(1800)
    key.CreateAngleAttr(15)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-30,-25,-45))
    ground = UsdGeom.Cube.Define(stage, '/Demo/Floor')
    ground.CreateSizeAttr(1)
    xf = UsdGeom.Xformable(ground)
    xf.AddTranslateOp().Set(Gf.Vec3d(float(center[0]),float(center[1]),float(lo[2]-0.015)))
    xf.AddScaleOp().Set(Gf.Vec3f(200,200,0.02))
    material = UsdShade.Material.Define(stage,'/Demo/FloorMaterial')
    shader = UsdShade.Shader.Define(stage,'/Demo/FloorMaterial/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.045,0.06,0.085))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(0.9)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material)
    render_product = rep.create.render_product(str(camera.GetPath()), (1280,720))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([render_product])
    radius = max(float(np.linalg.norm(hi-lo))*1.75, 0.8)
    def camera_at(angle, close=1.0):
        eye = center+np.array([radius*close*math.cos(angle),radius*close*math.sin(angle),radius*close*0.65])
        camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*center),Gf.Vec3d(0,0,1)).GetInverse())
    def capture():
        world.render()
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
        pixels=np.asarray(rgb.get_data())
        if pixels.shape[:2]!=(720,1280): raise RuntimeError(f'Bad RGB shape: {pixels.shape}')
        return Image.fromarray(np.ascontiguousarray(pixels[:,:,:3]))
    camera_at(-0.9)
    for _ in range(40): world.render()
    frame=capture()
    assert np.asarray(frame).std()>3, 'Empty frame'
    frame.save(args.output/'preview.png')
    print('PREVIEW_READY',flush=True)
    if not args.preview_only:
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',24)
        small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',19)
        video=args.output/'OSRACER_OpenUSD_IsaacSim.mp4'
        encoder=subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24',
                                  '-s','1280x720','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18',
                                  '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        traces=[]
        for i in range(600):
            if i<360:
                camera_at(-0.9+2*math.pi*i/360)
                caption='原始 OpenUSD 资产加载 · 多角度渲染'
                detail='10 连杆 / 9 关节 / 20 网格已解析；物理测试中碰撞体自动回退为凸包'
            else:
                t=(i-360)/30
                camera_at(-0.65+0.35*math.sin(t*0.5),0.92)
                pose=np.array([0.4*math.sin(t*1.6) if 'steering' in n else ((2*t+math.pi)%(2*math.pi)-math.pi)*(-1 if 'right_rear' in n else 1) for n in robot.dof_names],dtype=np.float32)
                robot.set_joint_positions(pose)
                traces.append({'frame':i,'q_rad':robot.get_joint_positions().tolist()})
                caption='转向与车轮关节可动性 · 运动学演示'
                detail='测试夹具固定车身；直接设置关节位置，非闭环驾驶或驱动力验证'
            frame=capture()
            draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,64),fill=(13,20,31))
            draw.text((28,13),'OSRACER  |  OpenUSD / Isaac Sim 6.0.1',font=font,fill='white')
            draw.text((1120,19),f'{i/30:04.1f} / 20s',font=small,fill=(141,216,240))
            draw.rectangle((0,624,1280,720),fill=(13,20,31))
            draw.text((28,635),caption,font=font,fill=(145,224,242))
            draw.text((28,675),detail,font=small,fill='white')
            encoder.stdin.write(frame.tobytes())
            if i in (0,180,359,420,599):frame.save(args.output/f'frame_{i:04d}.png')
            if i%60==0:print('RECORD',i,'/600',flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=60)!=0:raise RuntimeError('ffmpeg encode failed')
        encoder=None
        decode = subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'], capture_output=True, text=True)
        if decode.returncode: raise RuntimeError(decode.stderr)
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(video)]))
        assert int(probe['streams'][0]['nb_read_frames']) == 600
        report['decode_verified'] = True
        report['ffprobe'] = probe
        report['video']={'file':str(video),'frames':600,'duration_s':20,'sha256':hashlib.sha256(video.read_bytes()).hexdigest()}
        (args.output/'joint_trace.json').write_text(json.dumps(traces,indent=2))
    report['source_unchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==report['sha256']
    assert report['source_unchanged']
    report['status']='passed_with_limitations'
except Exception as error:
    report['status']='failed'
    report['error']=repr(error)
    raise
finally:
    (args.output/'validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if encoder is not None:
        encoder.stdin.close()
        encoder.wait(timeout=60)
    app.close()
