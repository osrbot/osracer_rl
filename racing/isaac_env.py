"""Native multi-vehicle PhysX backend. Run with scripts/run_isaac.sh.

The CAD asset is referenced, never edited. Physics defaults to 480 Hz; actions stay 60 Hz.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WHEELS = ['left_front_wheel_joint', 'right_front_wheel_joint',
          'left_rear_wheel_joint', 'right_rear_wheel_joint']
STEERS = ['left_steering_hinge_joint', 'right_steering_hinge_joint']
SIGNS = np.array([1., 1., 1., -1.])


class RaceIsaacEnv:
    @staticmethod
    def create_app():
        from isaacsim import SimulationApp
        return SimulationApp({'headless': True, 'width': 1280, 'height': 720,
            'limit_cpu_threads': 4,
            'renderer': 'RayTracedLighting', 'multi_gpu': False,
            'extra_args': ['--/app/asyncRendering=false']})

    def __init__(self, track, num_cars=2, render=False, tire_representation='native_cylinder',simulation_app=None,physics_hz=480):
        if isinstance(physics_hz, bool) or not isinstance(physics_hz, (int, np.integer)) or physics_hz < 60 or physics_hz % 60:
            raise ValueError('physics_hz must be an integer multiple of the 60 Hz control rate')
        self.physics_hz = int(physics_hz)
        self.physics_substeps = self.physics_hz // 60
        self.settle_steps = 25 * self.physics_substeps  # Preserve 25 / 60 seconds of settling.
        self._owns_app=simulation_app is None
        self.app=simulation_app if simulation_app is not None else self.create_app()
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, UsdLux
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import create_new_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        create_new_stage()
        self.track, self.num_cars, self.Action = track, num_cars, ArticulationAction
        self.stage = stage = get_current_stage()
        self.world = World(physics_dt=1/self.physics_hz, rendering_dt=1/30, stage_units_in_meters=1.)
        self.world.scene.add_default_ground_plane(z_position=0, static_friction=1., dynamic_friction=1., restitution=0.)
        material = UsdShade.Material.Define(stage, '/World/TireMaterial')
        pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        pm.CreateStaticFrictionAttr(1.); pm.CreateDynamicFrictionAttr(1.); pm.CreateRestitutionAttr(0.)
        self.robots = []
        for i in range(num_cars):
            path = f'/World/Car{i}'
            stage.DefinePrim(path, 'Xform').GetReferences().AddReference(
                str(ROOT/'OSRACER/USD/osracer_description/robot.usd'), '/Robot')
        for prim in list(stage.Traverse()):
            if not str(prim.GetPath()).startswith('/World/Car'): continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxAngularVelocityAttr(30000.)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
                if 'wheel_link' in str(prim.GetPath()):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(prim)
                api.CreateJointFrictionAttr(0.); api.CreateMaxJointVelocityAttr(30000.)
                if 'steering' in prim.GetName():
                    joint = UsdPhysics.RevoluteJoint(prim)
                    joint.CreateLowerLimitAttr(float(-np.degrees(.45)))
                    joint.CreateUpperLimitAttr(float(np.degrees(.45)))
        for i in range(num_cars):
            target=UsdGeom.Cube.Define(stage,f'/World/Car{i}/Links/base_link/LidarTarget')
            target.CreateSizeAttr(1.)
            tx=UsdGeom.Xformable(target)
            tx.AddTranslateOp().Set(Gf.Vec3d(.10,0,.325))
            tx.AddScaleOp().Set(Gf.Vec3f(.10,.16,.55))
            target.CreateDisplayColorAttr([Gf.Vec3f(*((.1,.8,.95) if i==0 else (.95,.25,.1)))])
            UsdPhysics.CollisionAPI.Apply(target.GetPrim())
            for name in WHEELS:
                link = stage.GetPrimAtPath(f'/World/Car{i}/Links/'+name.replace('_joint','_link'))
                center = UsdPhysics.MassAPI(link).GetCenterOfMassAttr().Get()
                tire_path=str(link.GetPath())+'/DrivingCollider'
                colliders=[]
                if tire_representation in ('native_cylinder','rounded_cylinder'):
                    tire=UsdGeom.Cylinder.Define(stage,tire_path)
                    margin=.001 if tire_representation=='rounded_cylinder' else 0.
                    tire.CreateRadiusAttr(.045-margin);tire.CreateHeightAttr(.04-2*margin);tire.CreateAxisAttr('Y')
                    if margin:
                        from pxr import Sdf
                        tire.GetPrim().CreateAttribute('physxConvexGeometry:margin',Sdf.ValueTypeNames.Float).Set(margin)
                elif tire_representation in ('convex120','convex120_scaled','convex120_precise','sdf120'):
                    # Same 45 mm radius / 40 mm width, with at most 15.5 um
                    # radial deficit. Mass and inertia remain authored on the
                    # unchanged CAD link; only the contact representation changes.
                    sides=120
                    angle=np.arange(sides)*2*np.pi/sides
                    ring=np.c_[.045*np.cos(angle),np.zeros(sides),.045*np.sin(angle)]
                    vertices=np.vstack([ring+[0.,-.02,0.],ring+[0.,.02,0.]])
                    tire=UsdGeom.Mesh.Define(stage,tire_path)
                    cooking_scale=.001 if tire_representation=='convex120_scaled' else 1.
                    tire.CreatePointsAttr([Gf.Vec3f(*map(float,v/cooking_scale)) for v in vertices])
                    faces=[list(range(sides)),list(range(2*sides-1,sides-1,-1))]
                    faces.extend([[j,j+sides,(j+1)%sides+sides,(j+1)%sides] for j in range(sides)])
                    tire.CreateFaceVertexCountsAttr([len(f) for f in faces])
                    tire.CreateFaceVertexIndicesAttr([j for f in faces for j in f])
                    tire.CreateSubdivisionSchemeAttr('none')
                    UsdPhysics.MeshCollisionAPI.Apply(tire.GetPrim()).CreateApproximationAttr('sdf' if tire_representation=='sdf120' else 'convexHull')
                    if tire_representation=='sdf120':
                        sdf=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(tire.GetPrim());sdf.CreateSdfResolutionAttr(256)
                    hull=PhysxSchema.PhysxConvexHullCollisionAPI.Apply(tire.GetPrim())
                    hull.CreateHullVertexLimitAttr(240)
                    if tire_representation=='convex120_precise':hull.CreateMinThicknessAttr(0.)
                elif tire_representation=='convex120_compound':
                    # Exact triangular-prism tiling avoids the cooker's coarse
                    # reduction of one complete circular convex hull.
                    tire=UsdGeom.Xform.Define(stage,tire_path)
                    sides=120
                    for j in range(sides):
                        angles=np.array([j,j+1])*2*np.pi/sides
                        ring=np.vstack([[0.,0.,0.],np.c_[.045*np.cos(angles),np.zeros(2),.045*np.sin(angles)]])
                        vertices=np.vstack([ring+[0.,-.02,0.],ring+[0.,.02,0.]])
                        part=UsdGeom.Mesh.Define(stage,tire_path+f'/Part{j}')
                        part.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in vertices])
                        faces=[[0,1,2],[5,4,3],[0,3,4,1],[1,4,5,2],[2,5,3,0]]
                        part.CreateFaceVertexCountsAttr([len(f) for f in faces])
                        part.CreateFaceVertexIndicesAttr([k for f in faces for k in f])
                        part.CreateSubdivisionSchemeAttr('none')
                        UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr('convexHull')
                        hull=PhysxSchema.PhysxConvexHullCollisionAPI.Apply(part.GetPrim())
                        hull.CreateHullVertexLimitAttr(64);hull.CreateMinThicknessAttr(0.)
                        colliders.append(part.GetPrim())
                else:
                    raise ValueError(f'Unknown tire representation: {tire_representation}')
                UsdGeom.Xformable(tire).AddTranslateOp().Set(Gf.Vec3d(*center))
                if tire_representation=='convex120_scaled':
                    UsdGeom.Xformable(tire).AddScaleOp().Set(Gf.Vec3f(.001))
                tire.CreateVisibilityAttr('invisible')
                for collider in colliders or [tire.GetPrim()]:
                    UsdPhysics.CollisionAPI.Apply(collider)
                    api=PhysxSchema.PhysxCollisionAPI.Apply(collider)
                    api.CreateContactOffsetAttr(.001); api.CreateRestOffsetAttr(0.)
                    UsdShade.MaterialBindingAPI.Apply(collider).Bind(material, materialPurpose='physics')
            self.robots.append(self.world.scene.add(SingleArticulation(
                prim_path=f'/World/Car{i}', name=f'osracer{i}')))
        # Boundaries are physical boxes with the same height used by ray queries.
        elevated=getattr(track,'has_elevation',False)
        boundaries=track.boundary_segments_3d() if elevated else track.boundary_segments()
        for i, segment in enumerate(boundaries):
            a, b = segment
            d = b-a
            wall = UsdGeom.Cube.Define(stage, f'/World/Barriers/Wall{i}')
            wall.CreateSizeAttr(1.)
            xf=UsdGeom.Xformable(wall)
            z=float((a[2]+b[2])/2) if elevated else 0.
            xf.AddTranslateOp().Set(Gf.Vec3d(float((a[0]+b[0])/2),float((a[1]+b[1])/2),z+.3))
            pitch=-float(np.degrees(np.arctan2(d[2],np.linalg.norm(d[:2])))) if elevated else 0.
            xf.AddRotateXYZOp().Set(Gf.Vec3f(0.,pitch,float(np.degrees(np.arctan2(d[1],d[0])))))
            xf.AddScaleOp().Set(Gf.Vec3f(float(np.linalg.norm(d)+.003),.04,.6))
            wall.CreateDisplayColorAttr([Gf.Vec3f(*((.8,.1,.1) if i%4<2 else (.9,.9,.9)))])
            UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
        self._road_visual()
        if elevated:
            triangles=np.asarray(track.road_triangles_3d())
            deck=UsdGeom.Mesh.Define(stage,'/World/BridgeDeck')
            deck.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in triangles.reshape(-1,3)])
            deck.CreateFaceVertexCountsAttr([3]*len(triangles))
            deck.CreateFaceVertexIndicesAttr(list(range(3*len(triangles))))
            deck.CreateDoubleSidedAttr(True)
            deck.CreateDisplayColorAttr([Gf.Vec3f(.055,.065,.08)])
            thickness=track.metadata.get('bridge',{}).get('deck_thickness',.08)
            # One welded, closed rigid deck uses the shared upper surface.
            # Separate triangle prisms introduce internal contact edges.
            top=triangles[triangles[:,:,2].max(axis=1)>1e-8]
            vertices,indices=np.unique(np.round(top.reshape(-1,3),9),axis=0,return_inverse=True)
            faces=indices.reshape(-1,3);n=len(vertices)
            edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
            _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
            boundary=edges[counts[inverse]==1]
            shell_faces=[faces,faces[:,::-1]+n]
            for u,v in boundary:shell_faces.append(np.array([[u,u+n,v+n],[u,v+n,v]]))
            points=np.vstack([vertices,vertices-np.array([0.,0.,thickness])])
            mesh=UsdGeom.Mesh.Define(stage,'/World/BridgeContacts/Surface')
            mesh.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in points])
            mesh.CreateFaceVertexCountsAttr([3]*sum(len(x) for x in shell_faces))
            mesh.CreateFaceVertexIndicesAttr(np.concatenate(shell_faces).reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
            mesh.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
            contact=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            contact.CreateContactOffsetAttr(.001);contact.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose='physics')
            # Only car/wall events are race collisions. Filtering report pairs
            # retains those physical contacts while excluding ordinary road
            # support contacts from the reporting/material lookup path.
            bodies=[p for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Car')
                    and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            walls=[p.GetPath() for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Barriers/')
                   and p.HasAPI(UsdPhysics.CollisionAPI)]
            for body in bodies:
                own=str(body.GetPath()).split('/')[2]
                others=[p.GetPath() for p in bodies if str(p.GetPath()).split('/')[2]!=own]
                PhysxSchema.PhysxContactReportAPI(body).CreateReportPairsRel().SetTargets(walls+others)
        self.world.reset()
        self.wids=[]; self.sids=[]; self.root_com=[]
        for robot in self.robots:
            wi=np.array([robot.dof_names.index(n) for n in WHEELS]); si=np.array([robot.dof_names.index(n) for n in STEERS])
            self.wids.append(wi); self.sids.append(si)
            kp=np.zeros(robot.num_dof); kd=kp.copy()
            kp[si]=3.; kd[si]=.08; kd[wi]=.03
            robot.get_articulation_controller().set_gains(kps=kp,kds=kd)
            effort=np.full(robot.num_dof,.3); effort[si]=1.
            robot._articulation_view.set_max_efforts(effort[None,:])
            robot.set_solver_position_iteration_count(16); robot.set_solver_velocity_iteration_count(4)
            robot.set_enabled_self_collisions(False)
            av=robot._articulation_view
            self.root_com.append(av.get_body_coms()[0][0,av.get_body_index('base_link')].copy())
        self.contacts=np.zeros(num_cars,dtype=int)
        self.car_contacts=np.zeros(num_cars,dtype=int)
        self.wall_contacts=np.zeros(num_cars,dtype=int)
        import omni.physx
        self.contact_subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        self.rgb=None
        self.render_product=None
        if render:
            import omni.replicator.core as rep
            self.rep=rep
            cam=UsdGeom.Camera.Define(stage,'/World/Camera')
            cam.CreateFocalLengthAttr(35); cam.CreateHorizontalApertureAttr(36)
            cam.CreateClippingRangeAttr(Gf.Vec2f(.01,2000))
            UsdGeom.Xformable(cam).AddTransformOp()
            UsdLux.DomeLight.Define(stage,'/World/Dome').CreateIntensityAttr(700)
            sun=UsdLux.DistantLight.Define(stage,'/World/Sun');sun.CreateIntensityAttr(1500)
            UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35,-25,-30))
            rp=rep.create.render_product('/World/Camera',(1280,720))
            self.render_product=rp
            self.rgb=rep.AnnotatorRegistry.get_annotator('rgb');self.rgb.attach([rp])
        out=ROOT/'output/racing/isaac';out.mkdir(parents=True,exist_ok=True)
        stage.GetRootLayer().Export(str(out/f'{track.id}.usda'))
        print('RACE_ISAAC_READY',track.id,num_cars,flush=True)

    def _road_visual(self):
        from pxr import UsdGeom,Gf
        if getattr(self.track,'has_elevation',False):return
        obj=self.track.path.parent/'track.obj'
        if obj.exists():
            vertices=[];indices=[];counts=[]
            for line in obj.read_text().splitlines():
                fields=line.split()
                if not fields:continue
                if fields[0]=='v':vertices.append([float(x) for x in fields[1:4]])
                if fields[0]=='f':
                    face=[int(x.split('/')[0])-1 for x in fields[1:]]
                    counts.append(len(face));indices.extend(face)
            mesh=UsdGeom.Mesh.Define(self.stage,'/World/Road')
            mesh.CreatePointsAttr([Gf.Vec3f(v[0],v[1],v[2]+.001) for v in vertices])
            mesh.CreateFaceVertexCountsAttr(counts);mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateDoubleSidedAttr(True);mesh.CreateDisplayColorAttr([Gf.Vec3f(.055,.065,.08)])
            return
        p=self.track.points
        tangent=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0)
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-8)
        n=np.c_[-tangent[:,1],tangent[:,0]]
        vertices=np.vstack([np.c_[p+n*self.track.width/2,np.full(len(p),.001)],np.c_[p-n*self.track.width/2,np.full(len(p),.001)]])
        mesh=UsdGeom.Mesh.Define(self.stage,'/World/Road')
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices]);N=len(p)
        mesh.CreateFaceVertexCountsAttr([4]*N)
        mesh.CreateFaceVertexIndicesAttr([j for i in range(N) for j in (i,(i+1)%N,(i+1)%N+N,i+N)])
        mesh.CreateDoubleSidedAttr(True);mesh.CreateDisplayColorAttr([Gf.Vec3f(.055,.065,.08)])

    def _on_contact(self,headers,data):
        from pxr import PhysicsSchemaTools
        for h in headers:
            paths=[str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in ('actor0','actor1')]
            cars=[i for i in range(self.num_cars) if any(f'/World/Car{i}/' in p for p in paths)]
            wall=any('/Barriers/' in p for p in paths)
            if wall or len(cars)>1:
                for i in cars:
                    self.contacts[i]+=1
                    self.wall_contacts[i]+=int(wall);self.car_contacts[i]+=int(len(cars)>1)

    def reset(self,seed=0,starts=None):
        rng=np.random.default_rng(seed)
        starts=starts or [(2.*i,0.) for i in range(self.num_cars)]
        for i,(robot,(s,offset)) in enumerate(zip(self.robots,starts)):
            p,yaw=self.track.at(s,offset)
            z=0.;grade=0.
            if getattr(self.track,'has_elevation',False):
                point,yaw,grade=self.track.at3d(s,offset);p=point[:2];z=float(point[2])
            if seed: p=p+rng.uniform(-.015,.015,2);yaw+=rng.uniform(-.015,.015)
            pitch=-np.arctan(grade)
            cy,sy=np.cos(yaw/2),np.sin(yaw/2);cp,sp=np.cos(pitch/2),np.sin(pitch/2)
            robot.set_world_pose(np.array([p[0],p[1],z+.055]),np.array([cy*cp,-sy*sp,cy*sp,sy*cp]))
            robot.set_world_velocity(np.zeros(6));robot.set_joint_positions(np.zeros(robot.num_dof));robot.set_joint_velocities(np.zeros(robot.num_dof))
            self._target(i,np.zeros(4),np.zeros(2))
        for _ in range(self.settle_steps):self.world.step(render=False)
        self.contacts.fill(0);self.car_contacts.fill(0);self.wall_contacts.fill(0)
        return self.states()

    def _target(self,i,wheels,steers):
        r=self.robots[i]
        r.apply_action(self.Action(joint_velocities=np.asarray(wheels,dtype=np.float32),joint_indices=self.wids[i]))
        r.apply_action(self.Action(joint_positions=np.asarray(steers,dtype=np.float32),joint_indices=self.sids[i]))

    def step(self,actions):
        self.contacts.fill(0);self.car_contacts.fill(0);self.wall_contacts.fill(0)
        for i,(w,s) in enumerate(actions):self._target(i,w,s)
        for _ in range(self.physics_substeps):self.world.step(render=False)
        return self.states()

    def states(self):
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        from isaacsim.core.utils.xforms import get_world_pose
        result=[]
        for i,r in enumerate(self.robots):
            p,q=r.get_world_pose();w,x,y,z=q;rot=quat_to_rot_matrix(q)
            v=r.get_linear_velocity()-np.cross(r.get_angular_velocity(),rot@self.root_com[i])
            local=rot.T@v
            laser_p,laser_q=get_world_pose(f'/World/Car{i}/Links/laser')
            raw=r.get_joint_velocities()[self.wids[i]]
            result.append(dict(x=float(p[0]),y=float(p[1]),z=float(p[2]),
                yaw=float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))),
                roll=float(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))),
                pitch=float(np.arcsin(np.clip(2*(w*y-z*x),-1,1))),
                vx=float(v[0]),vy=float(v[1]),vz=float(v[2]),yaw_rate=float(r.get_angular_velocity()[2]),
                rear_slip_beta=float(np.arctan2(local[1],local[0])),wheel_vel=raw*SIGNS,wheel_vel_raw=raw,
                steer_pos=r.get_joint_positions()[self.sids[i]],
                lidar_pose={'position':np.asarray(laser_p),'rotation':quat_to_rot_matrix(laser_q)},
                lidar_target={'position':p+rot@np.array([.10,0,.325]),'rotation':rot,
                              'half_size':np.array([.05,.08,.275])},
                collision=bool(self.contacts[i]),collision_count=int(self.contacts[i]),
                car_collision=bool(self.car_contacts[i]),barrier_collision=bool(self.wall_contacts[i])))
        return result

    def render(self):
        from pxr import Gf,UsdGeom
        p,_=self.robots[0].get_world_pose()
        matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(float(p[0])-1.5,float(p[1])-3.5,float(p[2])+5.5),Gf.Vec3d(float(p[0])+.5,float(p[1]),float(p[2])),Gf.Vec3d(0,0,1)).GetInverse()
        UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/Camera')).GetOrderedXformOps()[0].Set(matrix)
        before=p.copy();self.world.render()
        self.rep.orchestrator.step(rt_subframes=2,delta_time=0.,pause_timeline=False)
        if np.linalg.norm(self.robots[0].get_world_pose()[0]-before)>1e-6:raise RuntimeError('Rendering advanced physics')
        pixels=np.asarray(self.rgb.get_data())
        if pixels.shape[:2]!=(720,1280):raise RuntimeError(f'Bad frame {pixels.shape}')
        return np.ascontiguousarray(pixels[:,:,:3])

    def close(self):
        self.contact_subscription=None
        if self._owns_app:
            self.app.close()
        else:
            # The next track creates a new stage and World. Keep only Kit and
            # its compiled renderer alive; no physics scene or actor is reused.
            self.world.stop()
            if self.rgb is not None:self.rgb.detach()
            if self.render_product is not None:self.render_product.destroy()
            from isaacsim.core.api import World
            World.clear_instance()
