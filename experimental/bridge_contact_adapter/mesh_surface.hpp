#pragma once
#include "contact_kernel.hpp"
#include <array>
namespace bridge_contact {
struct SurfaceTriangle {std::array<V,3> points;std::array<int,3> indices,adjacent;V lo,hi,normal;};
struct TriangleSurface {
 std::vector<SurfaceTriangle> triangles;std::vector<V> vertices;std::vector<std::vector<int>> incident;
 explicit TriangleSurface(const std::vector<std::array<V,3>>& points);
};
std::vector<Contact> cylinder_surface(const T& wheel,const TriangleSurface& surface,double radius=.045,double half_width=.02,double margin=.001);
unsigned long rejected_internal_edges();
}
