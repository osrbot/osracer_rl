#include "contact_kernel.hpp"
#include "mesh_surface.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
using namespace bridge_contact;
int main(){
 T box=T::Identity(),wheel=T::Identity();V half(27.5,1.5,.04);box.translation()[2]=-.04;
 int cases=0;
 // Wheel spin must not rotate a flat-face contact normal.
 for(int i=0;i<360;++i){wheel=T::Identity();wheel.translation()=V(0,0,.0445);wheel.linear()=Eigen::AngleAxisd(i*M_PI/180,V::UnitY()).toRotationMatrix();auto q=cylinder_box(wheel,box,half);assert(q.size()==2);for(auto c:q){assert(c.analytic);assert((c.normal-V::UnitZ()).norm()<1e-12);assert(std::abs(c.separation+.0005)<1e-12);}++cases;}
 wheel=T::Identity();wheel.translation()=V(0,0,.0445);
 T incline=T::Identity();incline.linear()=Eigen::AngleAxisd(-std::atan(.045),V::UnitY()).toRotationMatrix();
 auto q=cylinder_box(incline*wheel,incline*box,half);assert(q.size()==2);for(auto c:q)assert((c.normal-incline.linear().col(2)).norm()<1e-12);++cases;
 // Actual finite face: no phantom contact outside either end or side.
 for(V p:{V(28,0,.0445),V(0,2,.0445),V(0,0,-.5),V(0,0,.5)}){wheel.translation()=p;assert(cylinder_box(wheel,box,half).empty());++cases;}
 // Underside contact points down; underpass farther away is empty.
 wheel.translation()=V(0,0,-.1245);q=cylinder_box(wheel,box,half);assert(!q.empty());for(auto c:q)assert(c.normal.z()<-.999);++cases;
 // Partially overlapping tread is clipped to the physical side boundary.
 wheel.translation()=V(0,1.51,.0445);q=cylinder_box(wheel,box,half);assert(q.size()==2);for(auto c:q)assert(c.point.y()<=1.5+1e-12);++cases;
 // Rounded cylinder surface reaches the deck end corner despite its bottom
 // support lying outside: must use the geometric edge fallback.
 wheel.translation()=V(27.51,0,.042);q=cylinder_box(wheel,box,half);assert(!q.empty());for(auto c:q){assert(!c.analytic);assert(c.normal.allFinite());assert(c.normal.x()>0.);assert(c.normal.z()>0.);assert(c.separation<0.);const V expected=V(.01,0,.042).normalized();std::cerr<<"edge normal "<<c.normal.transpose()<<" expected "<<expected.transpose()<<" sep "<<c.separation<<" expected "<<(std::hypot(.01,.042)-.045)<<"\n";assert((c.normal-expected).norm()<std::sin(.05*M_PI/180));assert(std::abs(c.separation-(std::hypot(.01,.042)-.045))<1e-6);}++cases;
 // A shorter box with identical nearby end/side/top faces must produce the
 // same local query. Every removed face is farther than the cylinder AABB
 // plus contact margin: no newly introduced face can support the cylinder.
 T short_box=box;short_box.translation().x()=27.;
 auto short_contacts=cylinder_box(wheel,short_box,V(.5,1.5,.04));
 assert(short_contacts.size()==q.size());
 for(size_t j=0;j<q.size();++j){assert((q[j].normal-short_contacts[j].normal).norm()<1e-8);assert(std::abs(q[j].separation-short_contacts[j].separation)<1e-8);}++cases;

 // True finite triangles: face, seam, end edge and underpass clearance.
 wheel=T::Identity();wheel.translation()=V(0,0,.0445);
 auto tri=cylinder_triangle(wheel,V(-1,-1,0),V(1,-1,0),V(0,1,0));assert(!tri.empty());for(auto c:tri){assert(c.analytic);assert((c.normal-V::UnitZ()).norm()<1e-12);assert(std::abs(c.separation+.0005)<1e-12);}++cases;
 wheel.translation()=V(2,0,.0445);assert(cylinder_triangle(wheel,V(-1,-1,0),V(1,-1,0),V(0,1,0)).empty());++cases;
 wheel.translation()=V(.01,0,.042);tri=cylinder_triangle(wheel,V(-1,-1,0),V(0,-1,0),V(0,1,0));assert(!tri.empty());for(auto c:tri){std::cerr<<"triangle edge "<<c.normal.transpose()<<" sep "<<c.separation<<"\n";assert((c.normal-V(.01,0,.042).normalized()).norm()<std::sin(.05*M_PI/180));assert(std::abs(c.separation-(std::hypot(.01,.042)-.045))<1e-6);}++cases;
 wheel.translation()=V(0,0,.045);assert(cylinder_triangle(wheel,V(-1,-1,.9),V(1,-1,.9),V(0,1,.9)).empty());++cases;
 wheel.translation()=V(0,0,.0445);auto seam1=cylinder_triangle(wheel,V(-1,-1,0),V(1,-1,0),V(1,1,0)),seam2=cylinder_triangle(wheel,V(-1,-1,0),V(1,1,0),V(-1,1,0));assert(!seam1.empty()&&!seam2.empty());for(auto c:seam1)assert((c.normal-V::UnitZ()).norm()<1e-12);for(auto c:seam2)assert((c.normal-V::UnitZ()).norm()<1e-12);++cases;
 // Convex bodies and another real wheel cannot silently pass through.
 ConvexProxy ca,cb;cb.pose.translation()=V(.089,0,0);
 for(auto kind:{ConvexProxy::Cylinder,ConvexProxy::Sphere,ConvexProxy::Capsule}){cb.kind=kind;cb.parameters=V(.045,.02,0);auto contacts=convex_contacts(ca,cb,.001);assert(!contacts.empty());for(auto c:contacts){assert(c.normal.x()<-.999);assert(std::abs(c.separation+.001)<1e-6);}++cases;}
 cb.kind=ConvexProxy::Vertices;cb.pose=T::Identity();cb.pose.translation().z()=-.04;for(double x:{-.2,.2})for(double y:{-.2,.2})for(double z:{-.04,.04})cb.vertices.push_back(V(x,y,z));ca.pose=wheel;
 auto poly=convex_contacts(ca,cb,.001);assert(!poly.empty());for(auto c:poly){assert(c.normal.z()>.999);assert(std::abs(c.separation+.0005)<1e-6);}++cases;

 TriangleSurface flat({{V(-1,-1,0),V(1,-1,0),V(1,1,0)},{V(-1,-1,0),V(1,1,0),V(-1,1,0)}});
 for(int i=-10;i<=10;++i){wheel=T::Identity();wheel.translation()=V(i*.005,0,.0445);auto contacts=cylinder_surface(wheel,flat);assert(!contacts.empty());for(auto c:contacts){assert(c.analytic);assert((c.normal-V::UnitZ()).norm()<1e-12);assert(std::abs(c.separation+.0005)<1e-12);}++cases;}
 std::vector<V> vv={V(-1,-1,-.08),V(0,-1,-.08),V(0,1,-.08),V(-1,1,-.08),V(-1,-1,0),V(0,-1,0),V(0,1,0),V(-1,1,0)};
 std::vector<std::array<int,3>> ff={{4,5,6},{4,6,7},{0,2,1},{0,3,2},{1,2,6},{1,6,5},{0,4,7},{0,7,3},{0,1,5},{0,5,4},{3,7,6},{3,6,2}};
 std::vector<std::array<V,3>> cube_triangles;for(auto f:ff)cube_triangles.push_back({vv[f[0]],vv[f[1]],vv[f[2]]});TriangleSurface solid(cube_triangles);
 wheel.translation()=V(.01,0,.042);auto corner=cylinder_surface(wheel,solid);assert(!corner.empty());for(auto c:corner){assert((c.normal-V(.01,0,.042).normalized()).norm()<std::sin(.05*M_PI/180));assert(std::abs(c.separation-(std::hypot(.01,.042)-.045))<1e-6);}++cases;
 wheel.translation()=V(-.5,0,-.5);assert(cylinder_surface(wheel,solid).empty());++cases;
 assert(rejected_internal_edges()>0);
 assert(numerical_failures()==0);
 std::cout<<"{\"passed\":true,\"geometry_cases\":"<<cases<<",\"spin_invariance_samples\":360,\"edge_fallback\":\"FCL libccd double precision\"}\n";
}
