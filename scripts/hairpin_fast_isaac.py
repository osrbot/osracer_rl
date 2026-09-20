"""480 Hz PhysX dynamics, 60 Hz controls, physical steering stops."""
import numpy as np
from hairpin_fast_common import OUT,POINTS,STEER_LIMIT
from hairpin_isaac import IsaacEnv

class FastIsaacEnv(IsaacEnv):
    def __init__(self,render=False):
        super().__init__(render=render,output=OUT,physics_dt=1/480,control_dt=1/60,
                         steering_limit=STEER_LIMIT,max_joint_velocity_deg=30000.,
                         road_points=POINTS,center_points=POINTS,road_width=1.2,
                         camera=((5.5,-1.5,5.),(5.5,.8,0)))
        av=self.robot._articulation_view
        self.root_com=av.get_body_coms()[0][0,av.get_body_index('base_link')].copy()

    def state(self):
        state=super().state()
        # Root linear velocity is reported at the root body's center of mass.
        # Transform the COM offset and subtract omega cross offset for base-origin velocity.
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        _,quat=self.robot.get_world_pose()
        rotation=quat_to_rot_matrix(quat)
        offset=rotation@self.root_com
        velocity=self.robot.get_linear_velocity()-np.cross(self.robot.get_angular_velocity(),offset)
        state['cg_vx'],state['cg_vy']=state['vx'],state['vy']
        state['vx'],state['vy']=float(velocity[0]),float(velocity[1])
        local=rotation.T@velocity
        state['rear_slip_beta']=float(np.arctan2(local[1],local[0]))
        return state

    def render(self):
        from pxr import Gf,UsdGeom
        pos,_=self.robot.get_world_pose()
        center=float(np.clip(pos[0],1.5,5.5))
        cam=self.stage.GetPrimAtPath('/World/Camera')
        UsdGeom.Xformable(cam).GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(center,-1.4,4.8),Gf.Vec3d(center,.8,0),Gf.Vec3d(0,0,1)).GetInverse())
        return super().render()
