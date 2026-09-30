#pragma once
#include <Eigen/Geometry>
#include <vector>
namespace bridge_contact {
using V=Eigen::Vector3d;
using T=Eigen::Isometry3d;
struct Contact { V point,normal; double separation; bool analytic; };
// Exact finite cylinder: local Y is its axis. Box is finite in all directions.
std::vector<Contact> cylinder_box(const T& cylinder,const T& box,const V& half,
                                 double radius=.045,double half_width=.02,double margin=.001);
std::vector<Contact> cylinder_plane(const T& cylinder,const V& normal,double offset,
                                   double radius=.045,double half_width=.02,double margin=.001);
}
namespace bridge_contact {
struct ConvexProxy {
 enum Kind {Cylinder,Box,Vertices,Sphere,Capsule,Ellipsoid,Cone};
 Kind kind=Cylinder; T pose=T::Identity(); V parameters=V(.045,.02,0);
 std::vector<V> vertices; double round_margin=0;
};
// Generic exact support geometry, with numerical GJK/EPA edge contacts.
std::vector<Contact> convex_contacts(const ConvexProxy& a,const ConvexProxy& b,double contact_margin);
std::vector<Contact> cylinder_triangle(const T& cylinder,const V& a,const V& b,const V& c,double radius=.045,double half_width=.02,double margin=.001);
unsigned long numerical_failures();
}
