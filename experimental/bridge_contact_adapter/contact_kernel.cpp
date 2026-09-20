#include "contact_kernel.hpp"
#include <fcl/fcl.h>
#include <algorithm>
#include <ccd/ccd.h>
#include <atomic>
#include <cmath>
namespace bridge_contact {
namespace {
struct Shape {T pose; V half; bool cylinder; double radius,height;const std::vector<V>* vertices=nullptr;int primitive=0;double rounding=0;};
std::atomic<unsigned long> failures{0};
void support(const void* obj,const ccd_vec3_t* dir,ccd_vec3_t* out){
 const auto& s=*static_cast<const Shape*>(obj);V d=s.pose.linear().transpose()*V(ccdVec3X(dir),ccdVec3Y(dir),ccdVec3Z(dir)),v;
 if(s.vertices){double best=-1e300;for(const V& p:*s.vertices){double score=d.dot(p);if(score>best){best=score;v=p;}}}
 else if(s.primitive==1){v=d.normalized()*s.radius;}
 else if(s.primitive==2){v=d.normalized()*s.radius;v.y()+=d.y()>=0?s.height:-s.height;}
 else if(s.primitive==3){V q=s.half.cwiseProduct(d);v=s.half.cwiseProduct(q)/std::max(q.norm(),1e-30);}
 else if(s.primitive==4){double len=std::hypot(d.x(),d.z());V base(len>0?s.radius*d.x()/len:0,-s.height,len>0?s.radius*d.z()/len:0),apex(0,s.height,0);v=d.dot(base)>d.dot(apex)?base:apex;}
 else if(s.cylinder){double len=std::hypot(d.x(),d.z());v=V(len>0?s.radius*d.x()/len:0,d.y()>=0?s.height:-s.height,len>0?s.radius*d.z()/len:0);}
 else for(int j=0;j<3;++j)v[j]=d[j]>=0?s.half[j]:-s.half[j];
 if(s.rounding>0 && d.squaredNorm()>0)v+=s.rounding*d.normalized();
 v=s.pose*v;ccdVec3Set(out,v.x(),v.y(),v.z());
}
void center(const void* obj,ccd_vec3_t* out){const V& v=static_cast<const Shape*>(obj)->pose.translation();ccdVec3Set(out,v.x(),v.y(),v.z());}
}

std::vector<Contact> cylinder_plane(const T& tf,const V& n,double d,double r,double h,double margin) {
 const V axis=tf.linear().col(1),c=tf.translation();
 const double axial=axis.dot(n); const V radial=n-axis*axial;
 const double radial_norm=radial.norm();
 V p=c;
 if(radial_norm>1e-14)p-=r*radial/radial_norm;
 std::vector<Contact> points;
 for(double side:{-1.,1.}){V candidate=p+side*h*axis;double separation=candidate.dot(n)-d;if(separation<=margin)points.push_back({candidate,n,separation,true});}
 return points;
}
std::vector<Contact> cylinder_box(const T& ct,const T& bt,const V& half,double r,double h,double margin) {
 const T local=bt.inverse()*ct; const V c=local.translation();
 // Select the closest exposed box face; accept its analytical support only
 // when the support point/segment is genuinely inside this finite face.
 int face=0; double best=-1e300;
 for(int j=0;j<3;++j){const double gap=std::abs(c[j])-half[j];if(gap>best){face=j;best=gap;}}
 if(best>=0.){
  V n=V::Zero();n[face]=c[face]>=0?1.:-1.;
  auto pts=cylinder_plane(local,n,half[face],r,h,margin);
  if(!pts.empty()){
   if(pts.size()==2){
    const V a=pts[0].point,b=pts[1].point,delta=b-a;double lo=0,hi=1;
    for(int j=0;j<3;++j)if(j!=face){
     if(std::abs(delta[j])<1e-14){if(std::abs(a[j])>half[j]+1e-12)hi=-1.;}
     else {double t0=(-half[j]-a[j])/delta[j],t1=(half[j]-a[j])/delta[j];if(t0>t1)std::swap(t0,t1);lo=std::max(lo,t0);hi=std::min(hi,t1);}
    }
    if(lo<=hi){double s0=pts[0].separation,ds=pts[1].separation-s0;pts[0].point=a+lo*delta;pts[1].point=a+hi*delta;pts[0].separation=s0+lo*ds;pts[1].separation=s0+hi*ds;}
    else pts.clear();
   } else {
    for(int j=0;j<3;++j)if(j!=face && std::abs(pts[0].point[j])>half[j]+1e-12){pts.clear();break;}
   }
   if(!pts.empty()){for(auto& p:pts){p.point=bt*p.point;p.normal=bt.linear()*p.normal;}return pts;}
  }
 }
 // Cylinder/edge/corner cases use an independent double-precision convex
 // narrow phase. FCL's Cylinder support is analytical, not a polygon mesh.
 auto cy=std::make_shared<fcl::Cylinderd>(r,2*h);
 // Intersect only remote box volume with a strict superset of the cylinder's
 // AABB. New clipping faces lie at least (margin + 1 cm) outside the entire
 // cylinder, so they cannot produce contacts within the requested margin.
 // This preserves every possible local contact while avoiding ill-conditioned
 // EPA on a 55 m by 8 cm box. It does not alter the simulated box.
 V extent;
 const V axis=local.linear().col(1);
 for(int j=0;j<3;++j)extent[j]=h*std::abs(axis[j])+r*std::sqrt(std::max(0.,1-axis[j]*axis[j]))+margin+.01;
 V lower=(-half).cwiseMax(c-extent),upper=half.cwiseMin(c+extent);
 if((lower.array()>upper.array()).any())return {};
 const V clipped_center=(lower+upper)/2,clipped_size=upper-lower;
 auto bx=std::make_shared<fcl::Boxd>(clipped_size[0],clipped_size[1],clipped_size[2]);
 T clipped_box=bt;clipped_box.translation()=bt*clipped_center;
 T ft=ct;ft.linear()=ct.linear()*Eigen::AngleAxisd(-M_PI/2,V::UnitX()).toRotationMatrix();
 fcl::CollisionObjectd co(cy,ft),bo(bx,clipped_box);
 // FCL 0.7's penetration wrapper does not expose EPA tolerance. Use the
 // same analytical support with direct libccd and explicit metre tolerances.
 Shape cshape{ct,V::Zero(),true,r,h},bshape{clipped_box,clipped_size/2,false,0,0};
 ccd_t ccd;CCD_INIT(&ccd);ccd.support1=support;ccd.support2=support;ccd.center1=center;ccd.center2=center;
 ccd.max_iterations=1000;ccd.epa_tolerance=1e-14;ccd.dist_tolerance=1e-14;
 ccd_real_t depth;ccd_vec3_t direction,position;std::vector<Contact> out;
 if(ccdGJKPenetration(&cshape,&bshape,&ccd,&depth,&direction,&position)==0){
  out.push_back({V(ccdVec3X(&position),ccdVec3Y(&position),ccdVec3Z(&position)),-V(ccdVec3X(&direction),ccdVec3Y(&direction),ccdVec3Z(&direction)), -depth,false});
 }
 if(!out.empty())return out;
 fcl::DistanceRequestd dr(true,false,0,0,1e-10,fcl::GST_LIBCCD);fcl::DistanceResultd dist;
 double distance=fcl::distance(&co,&bo,dr,dist);
 if(distance>=0 && distance<=margin){V normal=dist.nearest_points[0]-dist.nearest_points[1];if(normal.norm()>1e-12)out.push_back({dist.nearest_points[0],normal.normalized(),distance,false});}
 return out;
}
unsigned long numerical_failures(){return failures.load();}
std::vector<Contact> convex_contacts(const ConvexProxy& a,const ConvexProxy& b,double margin){
 auto convert=[](const ConvexProxy& p){Shape s{p.pose,p.parameters,p.kind==ConvexProxy::Cylinder,p.parameters.x(),p.parameters.y()};s.rounding=p.round_margin;
  if(p.kind==ConvexProxy::Vertices)s.vertices=&p.vertices;
  if(p.kind>=ConvexProxy::Sphere)s.primitive=int(p.kind)-int(ConvexProxy::Sphere)+1;
  return s;};
 Shape aa=convert(a),bb=convert(b);
 // Pair-local coordinates retain physical metres while reducing cancellation.
 const V origin=(aa.pose.translation()+bb.pose.translation())/2;
 aa.pose.translation()-=origin;bb.pose.translation()-=origin;
 ccd_t ccd;CCD_INIT(&ccd);ccd.support1=support;ccd.support2=support;ccd.center1=center;ccd.center2=center;
 ccd.max_iterations=1000;ccd.epa_tolerance=1e-14;ccd.dist_tolerance=1e-14;
 ccd_real_t depth;ccd_vec3_t direction,position;
 const int code=ccdGJKPenetration(&aa,&bb,&ccd,&depth,&direction,&position);
 if(code==0){V p(ccdVec3X(&position),ccdVec3Y(&position),ccdVec3Z(&position)),n(-ccdVec3X(&direction),-ccdVec3Y(&direction),-ccdVec3Z(&direction));
  if(p.allFinite()&&n.allFinite()&&n.norm()>.5&&std::isfinite(depth))return {{p+origin,n.normalized(),-depth,false}};
  ++failures;return {};
 }
 double distance;V p1,p2;
 bool separated=fcl::detail::GJKDistance<double>(&aa,support,&bb,support,1000,1e-12,&distance,&p1,&p2);
 if(separated&&distance>=0&&distance<=margin){V n=p1-p2;if(n.norm()>1e-12)return {{p1+origin,n.normalized(),distance,false}};}
 if(code==-2 || (!separated&&ccdGJKIntersect(&aa,&bb,&ccd)))++failures;
 return {};
}
std::vector<Contact> cylinder_triangle(const T& tf,const V& a,const V& b,const V& c,double r,double h,double margin){
 V original=(b-a).cross(c-a);if(original.norm()<1e-14)return {};
 original.normalize();V n=original;if(n.dot(tf.translation()-a)<0)n=-n;
 auto pts=cylinder_plane(tf,n,n.dot(a),r,h,margin);
 const V vertices[3]={a,b,c};
 if(pts.size()==2){V start=pts[0].point,delta=pts[1].point-start;double lo=0,hi=1;
  for(int j=0;j<3;++j){V inward=original.cross(vertices[(j+1)%3]-vertices[j]);double base=inward.dot(start-vertices[j]),rate=inward.dot(delta);
   if(std::abs(rate)<1e-14){if(base< -1e-12)hi=-1.;}
   else if(rate>0)lo=std::max(lo,-base/rate);else hi=std::min(hi,-base/rate);
  }
  if(lo<=hi){double s0=pts[0].separation,ds=pts[1].separation-s0;pts[0].point=start+lo*delta;pts[1].point=start+hi*delta;pts[0].separation=s0+lo*ds;pts[1].separation=s0+hi*ds;return pts;}
 }else if(pts.size()==1){bool inside=true;for(int j=0;j<3;++j)inside &= original.cross(vertices[(j+1)%3]-vertices[j]).dot(pts[0].point-vertices[j])>=-1e-12;if(inside)return pts;}
 // Clip only outside a strict superset of the cylinder envelope; each new
 // polygon edge is farther than contact_margin + 1 cm from the cylinder.
 std::vector<V> polygon={a,b,c};V axis=tf.linear().col(1),extent;
 for(int j=0;j<3;++j)extent[j]=h*std::abs(axis[j])+r*std::sqrt(std::max(0.,1-axis[j]*axis[j]))+margin+.01;
 for(int j=0;j<3;++j)for(double side:{-1.,1.}){
  double bound=tf.translation()[j]+side*extent[j];std::vector<V> next;
  for(size_t k=0;k<polygon.size();++k){V u=polygon[k],v=polygon[(k+1)%polygon.size()];double du=side*(u[j]-bound),dv=side*(v[j]-bound);if(du<=0)next.push_back(u);if((du<0&&dv>0)||(du>0&&dv<0))next.push_back(u+(v-u)*(du/(du-dv)));}
  polygon=std::move(next);
 }
 if(polygon.empty())return {};
 ConvexProxy wheel,tri;wheel.pose=tf;wheel.parameters=V(r,h,0);tri.kind=ConvexProxy::Vertices;tri.vertices=polygon;tri.pose.translation()=V::Zero();for(const V& p:polygon)tri.pose.translation()+=p/double(polygon.size());
 for(V& p:tri.vertices)p-=tri.pose.translation();
 return convex_contacts(wheel,tri,margin);
}

}
