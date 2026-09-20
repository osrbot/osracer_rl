"""Raw native PhysX support contacts for bounded diagnostic fixtures only."""
from pathlib import Path
import numpy as np

def attach(env, maximum_control_tick=240):
    from pxr import PhysxSchema, UsdPhysics, PhysicsSchemaTools
    import omni.physx
    from scripts.race_bridge_probe import serial
    targets=[p.GetPath() for p in env.stage.Traverse()
             if (str(p.GetPath()).startswith('/World/PlanarBridge/') or
                 str(p.GetPath()).startswith('/World/BridgeContacts/') or
                 str(p.GetPath()).startswith('/World/defaultGroundPlane/'))
             and p.HasAPI(UsdPhysics.CollisionAPI)]
    for p in env.stage.Traverse():
        if str(p.GetPath()).startswith('/World/Car') and p.HasAPI(UsdPhysics.RigidBodyAPI):
            api=PhysxSchema.PhysxContactReportAPI.Apply(p)
            existing=api.GetReportPairsRel().GetTargets()
            api.CreateReportPairsRel().SetTargets(list(dict.fromkeys(existing+targets)))
    result={'records':[], 'control_tick':[0], 'targets':[str(p) for p in targets]}
    def callback(headers,data):
        if result['control_tick'][0]>maximum_control_tick:return
        for h in headers:
            paths={k:str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in
                   ('actor0','actor1','collider0','collider1')}
            if not any('PlanarBridge' in p or 'BridgeContacts' in p or 'defaultGroundPlane' in p for p in paths.values()):continue
            contacts=[]
            for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
                contacts.append({k:serial(np.asarray(getattr(d,k))) for k in
                                 ('position','normal','impulse','separation')})
            result['records'].append({'control_tick':result['control_tick'][0],**paths,'contacts':contacts})
    subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(callback)
    return result,subscription
