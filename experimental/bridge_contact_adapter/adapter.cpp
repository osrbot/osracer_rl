// Diagnostic only. No scene queries or non-box/non-plane contacts are claimed.
#include <PxPhysicsAPI.h>
#include <geomutils/PxContactBuffer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/base/tf/token.h>
#include <carb/Framework.h>
#include "IPhysxCustomGeometry.h"
#include "IPhysx.h"
#include "contact_kernel.hpp"
#include "geometry_dispatch.hpp"
#include "mesh_surface.hpp"
#include <atomic>
#include <cfloat>
CARB_FRAMEWORK_GLOBALS("bridge.contact.diagnostic")
namespace {
using namespace physx;
using namespace bridge_contact;
std::atomic<unsigned long> created{0},calls{0},analytic{0},fallback{0},unsupported{0};
omni::physx::IPhysxCustomGeometry* iface=nullptr;
size_t registry=0;
const PxCustomGeometry::Type* known_wheel_type=nullptr;
long cached_stage=-1;
std::atomic<unsigned long> ray_calls{0},overlap_calls{0},sweep_calls{0},query_failures{0};
T transform(const PxTransform& p){T t=T::Identity();t.translation()=V(p.p.x,p.p.y,p.p.z);t.linear()=Eigen::Quaterniond(p.q.w,p.q.x,p.q.y,p.q.z).toRotationMatrix();return t;}
PxVec3 vec(const V& v){return PxVec3(float(v.x()),float(v.y()),float(v.z()));}
void* create(pxr::SdfPath,long stage,const PxCustomGeometry::Type& type,void*){++created;known_wheel_type=&type;if(stage!=cached_stage){clear_mesh_cache();cached_stage=stage;}return new int(1);}
void release(pxr::SdfPath,void* object){delete static_cast<int*>(object);}
PxBounds3 bounds(const PxGeometry&,void*){return PxBounds3(PxVec3(-.045f,-.02f,-.045f),PxVec3(.045f,.02f,.045f));}
bool contacts(const PxGeometry&,const PxGeometry& b,const PxTransform& a_pose,const PxTransform& b_pose,float margin,float,float,PxContactBuffer& buffer,void*,void*){
 ++calls;auto pts=dispatch_contacts(a_pose,b,b_pose,margin,known_wheel_type);
 for(auto& p:pts){buffer.contact(vec(p.point),vec(p.normal),float(p.separation));if(p.analytic)++analytic;else ++fallback;}
 return !pts.empty();
}
PxU32 raycast(const PxVec3& origin,const PxVec3& direction,const PxGeometry&,const PxTransform& pose,float max_distance,PxHitFlags,PxU32 max_hits,PxGeomRaycastHit* hits,PxU32,PxRaycastThreadContext*,void*){
 ++ray_calls;if(!max_hits)return 0;
 PxVec3 oo=pose.transformInv(origin),dd=pose.q.rotateInv(direction);V o(oo.x,oo.y,oo.z),d(dd.x,dd.y,dd.z),normal;double best=max_distance+1.;
 auto accept=[&](double t,V n){if(t>=0&&t<best&&t<=max_distance){best=t;normal=n;}};
 if(o.x()*o.x()+o.z()*o.z()<=.045*.045&&std::abs(o.y())<=.02){best=0;normal=-d;}
 double aa=d.x()*d.x()+d.z()*d.z(),bb=2*(o.x()*d.x()+o.z()*d.z()),cc=o.x()*o.x()+o.z()*o.z()-.045*.045;
 double discr=bb*bb-4*aa*cc;
 if(aa>1e-20&&discr>=0)for(double sign:{-1.,1.}){double t=(-bb+sign*std::sqrt(discr))/(2*aa);V p=o+t*d;if(std::abs(p.y())<=.02+1e-12)accept(t,V(p.x(),0,p.z()).normalized());}
 if(std::abs(d.y())>1e-20)for(double sign:{-1.,1.}){double t=(sign*.02-o.y())/d.y();V p=o+t*d;if(p.x()*p.x()+p.z()*p.z()<=.045*.045+1e-12)accept(t,V(0,sign,0));}
 if(best>max_distance)return 0;
 hits[0].distance=float(best);hits[0].position=origin+direction*float(best);hits[0].normal=pose.q.rotate(vec(normal));hits[0].flags=PxHitFlag::ePOSITION|PxHitFlag::eNORMAL;hits[0].faceIndex=0xffffffffu;return 1;
}
bool overlap(const PxGeometry&,const PxTransform& p0,const PxGeometry& g1,const PxTransform& p1,PxOverlapThreadContext*,void*){
 ++overlap_calls;auto pts=dispatch_contacts(p0,g1,p1,0,known_wheel_type);for(auto& p:pts)if(p.separation<=0)return true;return false;
}
bool sweep(const PxVec3& direction,float max_distance,const PxGeometry&,const PxTransform& p0,const PxGeometry& g1,const PxTransform& p1,PxGeomSweepHit& hit,PxHitFlags,float inflation,PxSweepThreadContext*,void*){
 ++sweep_calls;double travelled=0;
 for(int iteration=0;iteration<256;++iteration){PxTransform moving=p1;moving.p+=direction*float(travelled);
  auto pts=dispatch_contacts(p0,g1,moving,max_distance+inflation+1.,known_wheel_type);
  if(pts.empty())return false;
  const Contact* nearest=&pts[0];for(auto& p:pts)if(p.separation<nearest->separation)nearest=&p;
  double gap=nearest->separation-inflation;
  if(gap<=1e-7){hit.distance=float(travelled);hit.position=vec(nearest->point);hit.normal=-vec(nearest->normal);hit.flags=PxHitFlag::ePOSITION|PxHitFlag::eNORMAL;hit.faceIndex=0xffffffffu;return true;}
  travelled+=std::max(gap,1e-8);if(travelled>max_distance)return false;
 }
 ++query_failures;return false;
}
void mass(const PxGeometry&,PxMassProperties& out,void*){float m=float(M_PI*.045*.045*.04);float iy=m*.045f*.045f/2;float ix=m*(3*.045f*.045f+.04f*.04f)/12;out=PxMassProperties(m,PxMat33(PxVec3(ix,0,0),PxVec3(0,iy,0),PxVec3(0,0,ix)),PxVec3(0));}
bool pcm(const PxGeometry&,float& threshold,void*){threshold=FLT_EPSILON;return false;}
}
extern "C" size_t bridge_register(){
 if(registry)return registry;
 g_carbFramework=carb::acquireFramework("bridge.contact.diagnostic");
 if(!g_carbFramework)return 0;
 iface=g_carbFramework->tryAcquireInterface<omni::physx::IPhysxCustomGeometry>();if(!iface)return 0;
 omni::physx::ICustomGeometryCallback cb;
 cb.createCustomGeometryFn=create;cb.releaseCustomGeometryFn=release;cb.computeCustomGeometryLocalBoundsFn=bounds;cb.generateCustomGeometryContactsFn=contacts;cb.raycastCustomGeometryFn=raycast;cb.overlapCustomGeometryFn=overlap;cb.sweepCustomGeometryFn=sweep;cb.computeCustomGeometryMassPropertiesFn=mass;cb.useCustomGeometryPersistentContactManifoldFn=pcm;
 registry=iface->registerCustomGeometry(pxr::TfToken("BridgeCylinderAPI"),cb);return registry;
}
extern "C" void bridge_unregister(){if(iface&&registry){iface->unregisterCustomGeometry(registry);registry=0;clear_mesh_cache();}}
extern "C" void bridge_extended_counters(unsigned long* out){out[0]=numerical_failures();out[1]=ray_calls;out[2]=overlap_calls;out[3]=sweep_calls;out[4]=query_failures;}
extern "C" void bridge_counters(unsigned long* out){out[0]=created;out[1]=calls;out[2]=analytic;out[3]=fallback;out[4]=unsupported_pairs();}

// Read-only runtime qualification: actual PhysX shape and scene, not USD attrs.
extern "C" int bridge_inspect_shape(const char* path,unsigned long* out){
 g_carbFramework=carb::acquireFramework("bridge.contact.diagnostic");
 auto* api=g_carbFramework->tryAcquireInterface<omni::physx::IPhysx>();if(!api)return -1;
 auto* shape=static_cast<physx::PxShape*>(api->getPhysXPtr(pxr::SdfPath(path),omni::physx::ePTShape));
 if(!shape)return -2;
 const auto& g=shape->getGeometry();out[0]=g.getType();out[1]=static_cast<physx::PxU32>(shape->getFlags());out[2]=out[3]=out[4]=out[5]=0;
 if(g.getType()==physx::PxGeometryType::eTRIANGLEMESH){const auto& mesh=static_cast<const physx::PxTriangleMeshGeometry&>(g);physx::PxU32 x,y,z;mesh.triangleMesh->getSDFDimensions(x,y,z);out[2]=x;out[3]=y;out[4]=z;out[5]=mesh.triangleMesh->getSDF()!=nullptr;}
 auto* actor=shape->getActor();if(actor&&actor->getScene()){out[6]=static_cast<physx::PxU32>(actor->getScene()->getFlags());out[7]=actor->getScene()->getFlags().isSet(physx::PxSceneFlag::eENABLE_GPU_DYNAMICS);out[8]=actor->getScene()->getFlags().isSet(physx::PxSceneFlag::eENABLE_PCM);}
 return 0;
}
extern "C" unsigned int bridge_query_selftest(){
 using namespace physx;unsigned int errors=0;PxBoxGeometry dummy(.1f,.1f,.1f);PxTransform identity(PxIdentity);PxGeomRaycastHit ray;
 if(!raycast(PxVec3(.1f,0,0),PxVec3(-1,0,0),dummy,identity,1,PxHitFlag::eDEFAULT,1,&ray,sizeof(ray),nullptr,nullptr)||std::abs(ray.distance-.055f)>1e-6||ray.normal.x<.999f)errors|=1;
 if(!raycast(PxVec3(0,.1f,0),PxVec3(0,-1,0),dummy,identity,1,PxHitFlag::eDEFAULT,1,&ray,sizeof(ray),nullptr,nullptr)||std::abs(ray.distance-.08f)>1e-6||ray.normal.y<.999f)errors|=2;
 if(!raycast(PxVec3(0),PxVec3(1,0,0),dummy,identity,1,PxHitFlag::eDEFAULT,1,&ray,sizeof(ray),nullptr,nullptr)||ray.distance!=0)errors|=4;
 if(raycast(PxVec3(.1f,.1f,0),PxVec3(-1,0,0),dummy,identity,1,PxHitFlag::eDEFAULT,1,&ray,sizeof(ray),nullptr,nullptr))errors|=8;
 PxSphereGeometry sphere(.04f);
 if(!overlap(dummy,identity,sphere,PxTransform(PxVec3(.08f,0,0)),nullptr,nullptr))errors|=16;
 if(overlap(dummy,identity,sphere,PxTransform(PxVec3(.1f,0,0)),nullptr,nullptr))errors|=32;
 PxGeomSweepHit hit;
 if(!sweep(PxVec3(-1,0,0),.3f,dummy,identity,sphere,PxTransform(PxVec3(.2f,0,0)),hit,PxHitFlag::eDEFAULT,0,nullptr,nullptr)||std::abs(hit.distance-.115f)>2e-6||hit.normal.x<.999)errors|=64;
 return errors;
}

extern "C" unsigned long bridge_rejected_internal_edges(){return rejected_internal_edges();}
