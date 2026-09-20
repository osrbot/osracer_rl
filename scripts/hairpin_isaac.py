"""Isaac Sim 6.0.1 native PhysX vehicle backend; launch via run_isaac.sh."""
import numpy as np
from hairpin_common import ROOT, OUT, WHEELS, STEERS, SIGNS, POINTS

class IsaacEnv:
    def __init__(self, render=False, output=None, physics_dt=1/240, control_dt=1/30,
                 steering_limit=None, road_points=None, camera=None, max_joint_velocity_deg=5000.,
                 road_width=.58,center_points=None):
        self.version='6.0.1'
        self.output=OUT if output is None else output
        self.substeps=round(control_dt/physics_dt)
        from isaacsim import SimulationApp
        self.app = SimulationApp({'headless': True, 'width':1280, 'height':720,
            'renderer':'RayTracedLighting', 'multi_gpu':False,
            'extra_args':['--/app/asyncRendering=false']})
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Sdf, UsdLux
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import open_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        self.Action = ArticulationAction
        open_stage(str(ROOT/'OSRACER/USD/osracer_description/robot.usd'))
        stage = get_current_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        self.stage = stage
        self.world = World(physics_dt=physics_dt, rendering_dt=1/30, stage_units_in_meters=1.)
        self.world.scene.add_default_ground_plane(z_position=0, static_friction=1., dynamic_friction=1., restitution=0.)
        # USD drive angular stiffness is authored per degree; controller gains
        # below use the public tensor API's SI radians instead.
        for p in list(stage.Traverse()):
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and max_joint_velocity_deg>5000.:
                body_api=PhysxSchema.PhysxRigidBodyAPI.Apply(p)
                body_api.CreateMaxAngularVelocityAttr(max_joint_velocity_deg)
            if p.HasAPI(UsdPhysics.CollisionAPI):
                if p.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull')
                if 'wheel_link' in str(p.GetPath()):
                    UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
            if p.IsA(UsdPhysics.RevoluteJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(p)
                api.CreateJointFrictionAttr(0.)
                api.CreateMaxJointVelocityAttr(max_joint_velocity_deg)
                if steering_limit is not None and 'steering' in p.GetName():
                    joint=UsdPhysics.RevoluteJoint(p)
                    joint.CreateLowerLimitAttr(float(-np.degrees(steering_limit)))
                    joint.CreateUpperLimitAttr(float(np.degrees(steering_limit)))
        mat = UsdShade.Material.Define(stage, '/World/TireMaterial')
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(1.); pm.CreateDynamicFrictionAttr(1.); pm.CreateRestitutionAttr(0.)
        for name in WHEELS:
            link = stage.GetPrimAtPath('/Robot/Links/' + name.replace('_joint', '_link'))
            center = UsdPhysics.MassAPI(link).GetCenterOfMassAttr().Get()
            tire = UsdGeom.Cylinder.Define(stage, str(link.GetPath()) + '/DrivingCollider')
            tire.CreateRadiusAttr(.045); tire.CreateHeightAttr(.04); tire.CreateAxisAttr('Y')
            UsdGeom.Xformable(tire).AddTranslateOp().Set(Gf.Vec3d(*center))
            tire.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(tire.GetPrim())
            phys = PhysxSchema.PhysxCollisionAPI.Apply(tire.GetPrim())
            phys.CreateContactOffsetAttr(.001); phys.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(tire.GetPrim()).Bind(mat, materialPurpose='physics')
        self.robot = self.world.scene.add(SingleArticulation(prim_path='/Robot', name='osracer'))
        self.world.reset()
        self.wids = np.array([self.robot.dof_names.index(n) for n in WHEELS])
        self.sids = np.array([self.robot.dof_names.index(n) for n in STEERS])
        self.controller = self.robot.get_articulation_controller()
        kp = np.zeros(self.robot.num_dof); kd = np.zeros(self.robot.num_dof)
        kp[self.sids] = 3.; kd[self.sids] = .08; kd[self.wids] = .03
        self.controller.set_gains(kps=kp, kds=kd)
        efforts = np.full(self.robot.num_dof, .3); efforts[self.sids] = 1.
        self.robot._articulation_view.set_max_efforts(efforts[None, :])
        self.robot.set_solver_position_iteration_count(16)
        self.robot.set_solver_velocity_iteration_count(4)
        self.robot.set_enabled_self_collisions(False)
        self.rgb = None
        if render:
            import omni.replicator.core as rep
            self.rep = rep
            # Noncolliding native USD road ribbons and dashed centerline.
            def strip(points, width, name, color, z):
                tangent = np.gradient(points, axis=0)
                tangent /= np.linalg.norm(tangent, axis=1)[:, None]
                normal = np.stack([-tangent[:,1], tangent[:,0]], axis=1)
                verts = np.concatenate([np.c_[points+normal*width/2, np.full(len(points),z)],
                                        np.c_[points-normal*width/2, np.full(len(points),z)]])
                n = len(points)
                mesh = UsdGeom.Mesh.Define(stage, name)
                mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in verts])
                mesh.CreateFaceVertexCountsAttr([4]*(n-1))
                mesh.CreateFaceVertexIndicesAttr([j for i in range(n-1) for j in (i,i+1,n+i+1,n+i)])
                mesh.CreateDoubleSidedAttr(True); mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            strip(np.array([[-15.,0.],[15.,0.]]),30.,'/World/VisualGround',(.18,.25,.24),.0001)
            road_points=np.concatenate([np.c_[np.linspace(-.55,0,30,endpoint=False),np.zeros(30)],
                POINTS[::5],np.c_[np.linspace(0,-.55,30),np.full(30,1.6)]]) if road_points is None else road_points
            strip(road_points, road_width, '/World/Road', (.045,.06,.085), .002)
            markings=POINTS if center_points is None else center_points
            for i in range(0,len(markings)-20,50):
                strip(markings[i:i+22:3], .015, f'/World/Line{i}', (.95,.75,.2), .003)
            camera_cfg=camera or ((1,-1.,4.2),(1,.8,0))
            cam = UsdGeom.Camera.Define(stage, '/World/Camera')
            cam.CreateFocalLengthAttr(35); cam.CreateHorizontalApertureAttr(36)
            cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
            xf = UsdGeom.Xformable(cam).AddTransformOp()
            xf.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*camera_cfg[0]),Gf.Vec3d(*camera_cfg[1]),Gf.Vec3d(0,0,1)).GetInverse())
            dome=UsdLux.DomeLight.Define(stage,'/World/Dome');dome.CreateIntensityAttr(700)
            key=UsdLux.DistantLight.Define(stage,'/World/Sun');key.CreateIntensityAttr(1500)
            UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35,-25,-30))
            rp=rep.create.render_product(str(cam.GetPath()),(1280,720))
            self.rgb=rep.AnnotatorRegistry.get_annotator('rgb');self.rgb.attach([rp])
        self.output.mkdir(parents=True, exist_ok=True)
        stage.GetSessionLayer().Export(str(self.output/'isaac_session.usda'))
        print('ISAAC_READY', self.robot.dof_names, flush=True)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        y, yaw = (0., 0.) if seed == 0 else (rng.uniform(-.025,.025),rng.uniform(-.03,.03))
        self.robot.set_world_pose(np.array([0., y, .055]), np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)]))
        self.robot.set_world_velocity(np.zeros(6))
        self.robot.set_joint_positions(np.zeros(self.robot.num_dof))
        self.robot.set_joint_velocities(np.zeros(self.robot.num_dof))
        self._targets(np.zeros(4), np.zeros(2))
        for _ in range(round((100/240)/self.world.get_physics_dt())): self.world.step(render=False)
        return self.state()

    def _targets(self, wheels, steers):
        self.robot.apply_action(self.Action(joint_velocities=np.asarray(wheels,dtype=np.float32),joint_indices=self.wids))
        self.robot.apply_action(self.Action(joint_positions=np.asarray(steers,dtype=np.float32),joint_indices=self.sids))

    def step(self, wheel_targets, steering_targets):
        self._targets(wheel_targets, steering_targets)
        for _ in range(self.substeps): self.world.step(render=False)
        return self.state()

    def state(self):
        p, q = self.robot.get_world_pose()
        w,x,y,z = q
        yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        roll = np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x),-1,1))
        v = self.robot.get_linear_velocity(); a = self.robot.get_angular_velocity()
        return dict(x=float(p[0]),y=float(p[1]),z=float(p[2]),yaw=float(yaw),
            roll=float(roll),pitch=float(pitch),vx=float(v[0]),vy=float(v[1]),yaw_rate=float(a[2]),
            wheel_vel=self.robot.get_joint_velocities()[self.wids]*SIGNS,
            steer_pos=self.robot.get_joint_positions()[self.sids])

    def render(self):
        before = self.robot.get_world_pose()[0].copy()
        self.world.render()
        self.rep.orchestrator.step(rt_subframes=2, delta_time=0., pause_timeline=False)
        after = self.robot.get_world_pose()[0]
        if np.linalg.norm(after-before) > 1e-6:
            raise RuntimeError('Rendering advanced vehicle dynamics')
        pixels=np.asarray(self.rgb.get_data())
        if pixels.shape[:2] != (720,1280): raise RuntimeError(f'Invalid RGB {pixels.shape}')
        return np.ascontiguousarray(pixels[:,:,:3])

    def close(self):
        self.app.close()

if __name__ == '__main__':
    env=IsaacEnv()
    try:
        print('RESET',env.reset(),flush=True)
        for i in range(120):
            state=env.step(np.array([8,8,8,-8]),np.zeros(2))
            if i%30==29: print('DRIVE',i,state,flush=True)
    finally: env.close()
