#include "geometry_dispatch.hpp"
#include "mesh_surface.hpp"
#include <atomic>
#include <array>
#include <mutex>
#include <unordered_map>
#include <memory>
namespace bridge_contact {
using namespace physx;
namespace {
std::atomic<unsigned long> unsupported{0};
V vector(const PxVec3& p){return V(p.x,p.y,p.z);}
T transform(const PxTransform& p){T t=T::Identity();t.translation()=vector(p.p);t.linear()=Eigen::Quaterniond(p.q.w,p.q.x,p.q.y,p.q.z).toRotationMatrix();return t;}
struct Mesh {PxTransform pose;PxMeshScale scale;std::shared_ptr<TriangleSurface> surface;};
std::mutex mutex;
std::unordered_map<const PxTriangleMesh*,std::shared_ptr<Mesh>> cache;
std::shared_ptr<Mesh> mesh_data(const PxTriangleMeshGeometry& g,const PxTransform& pose){
 std::lock_guard<std::mutex> lock(mutex);
 auto found=cache.find(g.triangleMesh);
 if(found!=cache.end()&&found->second->pose==pose&&found->second->scale.scale==g.scale.scale&&found->second->scale.rotation==g.scale.rotation)return found->second;
 auto out=std::make_shared<Mesh>();out->pose=pose;out->scale=g.scale;
 const auto* vertices=g.triangleMesh->getVertices();const void* indices=g.triangleMesh->getTriangles();
 bool short_indices=g.triangleMesh->getTriangleMeshFlags().isSet(PxTriangleMeshFlag::e16_BIT_INDICES);
 PxMat33 scale=g.scale.toMat33();
 std::vector<std::array<V,3>> triangles;
 for(PxU32 j=0;j<g.triangleMesh->getNbTriangles();++j){std::array<V,3> tri;
  for(int k=0;k<3;++k){PxU32 i=short_indices?static_cast<const PxU16*>(indices)[3*j+k]:static_cast<const PxU32*>(indices)[3*j+k];tri[k]=vector(pose.transform(scale*vertices[i]));}
  triangles.push_back(tri);
 }
 out->surface=std::make_shared<TriangleSurface>(triangles);cache[g.triangleMesh]=out;return out;
}
bool make_proxy(const PxGeometry& g,const PxTransform& pose,ConvexProxy& p,const PxCustomGeometry::Type* known){
 p.pose=transform(pose);
 auto axis_x_to_y=[&](){p.pose.linear()*=Eigen::AngleAxisd(-M_PI/2,V::UnitZ()).toRotationMatrix();};
 switch(g.getType()){
 case PxGeometryType::eSPHERE:p.kind=ConvexProxy::Sphere;p.parameters=V(static_cast<const PxSphereGeometry&>(g).radius,0,0);return true;
 case PxGeometryType::eBOX:p.kind=ConvexProxy::Box;p.parameters=vector(static_cast<const PxBoxGeometry&>(g).halfExtents);return true;
 case PxGeometryType::eCAPSULE:{auto& c=static_cast<const PxCapsuleGeometry&>(g);p.kind=ConvexProxy::Capsule;p.parameters=V(c.radius,c.halfHeight,0);axis_x_to_y();return true;}
 case PxGeometryType::eCONVEXMESH:{auto& c=static_cast<const PxConvexMeshGeometry&>(g);p.kind=ConvexProxy::Vertices;auto scale=c.scale.toMat33();for(PxU32 i=0;i<c.convexMesh->getNbVertices();++i)p.vertices.push_back(vector(scale*c.convexMesh->getVertices()[i]));return true;}
 case PxGeometryType::eCUSTOM:{auto& c=static_cast<const PxCustomGeometry&>(g);if(!known||c.callbacks->getCustomType()!=*known)return false;p.kind=ConvexProxy::Cylinder;p.parameters=V(.045,.02,0);return true;}
 case PxGeometryType::eCONVEXCORE:{auto& c=static_cast<const PxConvexCoreGeometry&>(g);p.round_margin=c.getMargin();
  switch(c.getCoreType()){
  case PxConvexCore::ePOINT:p.kind=ConvexProxy::Sphere;p.parameters=V(0,0,0);return true;
  case PxConvexCore::eSEGMENT:p.kind=ConvexProxy::Capsule;p.parameters=V(0,c.getCore<PxConvexCore::Segment>().length/2,0);axis_x_to_y();return true;
  case PxConvexCore::eBOX:p.kind=ConvexProxy::Box;p.parameters=vector(c.getCore<PxConvexCore::Box>().extents)/2;return true;
  case PxConvexCore::eELLIPSOID:p.kind=ConvexProxy::Ellipsoid;p.parameters=vector(c.getCore<PxConvexCore::Ellipsoid>().radii);return true;
  case PxConvexCore::eCYLINDER:{auto& v=c.getCore<PxConvexCore::Cylinder>();p.kind=ConvexProxy::Cylinder;p.parameters=V(v.radius,v.height/2,0);axis_x_to_y();return true;}
  case PxConvexCore::eCONE:{auto& v=c.getCore<PxConvexCore::Cone>();p.kind=ConvexProxy::Cone;p.parameters=V(v.radius,v.height/2,0);axis_x_to_y();return true;}
  default:return false;}
 }
 default:return false;
 }
}
}
unsigned long unsupported_pairs(){return unsupported.load();}
void clear_mesh_cache(){std::lock_guard<std::mutex> lock(mutex);cache.clear();}
std::vector<Contact> dispatch_contacts(const PxTransform& wp,const PxGeometry& g,const PxTransform& op,double margin,const PxCustomGeometry::Type* known){
 T wheel=transform(wp);
 if(g.getType()==PxGeometryType::eBOX)return cylinder_box(wheel,transform(op),vector(static_cast<const PxBoxGeometry&>(g).halfExtents),.045,.02,margin);
 if(g.getType()==PxGeometryType::ePLANE){T p=transform(op);V n=p.linear().col(0);return cylinder_plane(wheel,n,n.dot(p.translation()),.045,.02,margin);}
 if(g.getType()==PxGeometryType::eTRIANGLEMESH){
  auto mesh=mesh_data(static_cast<const PxTriangleMeshGeometry&>(g),op);
  return cylinder_surface(wheel,*mesh->surface,.045,.02,margin);
 }
 ConvexProxy a,b;a.pose=wheel;
 if(!make_proxy(g,op,b,known)){++unsupported;return {};}
 return convex_contacts(a,b,margin);
}
}
