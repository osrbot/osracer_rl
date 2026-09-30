#include "mesh_surface.hpp"
#include <map>
#include <tuple>
#include <atomic>
#include <Eigen/LU>
namespace bridge_contact {
namespace {
std::atomic<unsigned long> rejected{0};
double segment_distance(const V& p,const V& a,const V& b){V d=b-a;double t=std::clamp((p-a).dot(d)/std::max(d.squaredNorm(),1e-30),0.,1.);return (p-a-t*d).norm();}
bool cone(const V& normal,const std::vector<V>& generators){
 constexpr double tolerance=.001; // <0.058 degree numerical GJK edge tolerance.
 for(const V& n:generators)if((normal-n).norm()<tolerance)return true;
 for(size_t i=0;i<generators.size();++i)for(size_t j=i+1;j<generators.size();++j){
  const V& a=generators[i];const V& b=generators[j];double d=a.dot(b),det=1-d*d;if(det<1e-12)continue;
  double x=(a.dot(normal)-d*b.dot(normal))/det,y=(b.dot(normal)-d*a.dot(normal))/det;
  if(x>=-1e-8&&y>=-1e-8&&(x*a+y*b-normal).norm()<tolerance)return true;
 }
 for(size_t i=0;i<generators.size();++i)for(size_t j=i+1;j<generators.size();++j)for(size_t k=j+1;k<generators.size();++k){
  Eigen::Matrix3d m;m.col(0)=generators[i];m.col(1)=generators[j];m.col(2)=generators[k];if(std::abs(m.determinant())<1e-12)continue;V x=m.inverse()*normal;
  if(x.minCoeff()>=-1e-8&&(m*x-normal).norm()<tolerance)return true;
 }
 return false;
}
// Positive is a concave crease (the adjacent triangle enters this face's
// outside half-space). Such an internal edge is not a solid boundary feature.
double dihedral_side(const TriangleSurface& s,const SurfaceTriangle& t,int edge){
 int other=t.adjacent[edge];if(other<0)return 0;
 for(int v:s.triangles[other].indices)if(v!=t.indices[edge]&&v!=t.indices[(edge+1)%3])return t.normal.dot(s.vertices[v]-t.points[edge]);
 return 0;
}
bool exposed_edge_normal(const TriangleSurface& s,const SurfaceTriangle& t,const Contact& contact,const V& center,double bound){
 for(int edge=0;edge<3;++edge){if(segment_distance(center,t.points[edge],t.points[(edge+1)%3])>bound)continue;
  int other=t.adjacent[edge];if(other<0)return true; // Real finite boundary.
  const auto& neighbor=s.triangles[other];double side=dihedral_side(s,t,edge);
  if(side>=-1e-8)continue; // Coplanar or concave, therefore not an exposed ridge.
  if(cone(contact.normal,{t.normal,neighbor.normal}))return true;
 }
 for(int vertex:t.indices){if((center-s.vertices[vertex]).norm()>bound)continue;
  bool convex=true;std::vector<V> normals;
  for(int index:s.incident[vertex]){const auto& neighbor=s.triangles[index];normals.push_back(neighbor.normal);
   for(int edge=0;edge<3;++edge)if(neighbor.indices[edge]==vertex||neighbor.indices[(edge+1)%3]==vertex){if(neighbor.adjacent[edge]<0)return true;if(dihedral_side(s,neighbor,edge)>1e-8)convex=false;}
  }
  if(convex&&cone(contact.normal,normals))return true;
 }
 return false;
}
}
TriangleSurface::TriangleSurface(const std::vector<std::array<V,3>>& points){
 std::map<std::tuple<double,double,double>,int> ids;std::map<std::pair<int,int>,std::vector<std::pair<int,int>>> edges;
 for(const auto& p:points){SurfaceTriangle t;t.points=p;t.adjacent={-1,-1,-1};t.normal=(p[1]-p[0]).cross(p[2]-p[0]);if(t.normal.norm()<1e-14)continue;t.normal.normalize();t.lo=p[0].cwiseMin(p[1]).cwiseMin(p[2]);t.hi=p[0].cwiseMax(p[1]).cwiseMax(p[2]);
  for(int j=0;j<3;++j){auto key=std::make_tuple(p[j].x(),p[j].y(),p[j].z());auto found=ids.find(key);if(found==ids.end()){int id=vertices.size();vertices.push_back(p[j]);incident.push_back({});ids[key]=id;t.indices[j]=id;}else t.indices[j]=found->second;incident[t.indices[j]].push_back(triangles.size());}
  int index=triangles.size();triangles.push_back(t);for(int j=0;j<3;++j){auto key=std::minmax(t.indices[j],t.indices[(j+1)%3]);edges[key].push_back({index,j});}
 }
 for(const auto& [edge,pairs]:edges)if(pairs.size()==2){triangles[pairs[0].first].adjacent[pairs[0].second]=pairs[1].first;triangles[pairs[1].first].adjacent[pairs[1].second]=pairs[0].first;}
}
unsigned long rejected_internal_edges(){return rejected.load();}
std::vector<Contact> cylinder_surface(const T& wheel,const TriangleSurface& s,double r,double h,double margin){
 V axis=wheel.linear().col(1),extent;for(int j=0;j<3;++j)extent[j]=h*std::abs(axis[j])+r*std::sqrt(std::max(0.,1-axis[j]*axis[j]))+margin;
 V lo=wheel.translation()-extent,hi=wheel.translation()+extent;double bound=std::hypot(r,h)+margin+1e-6;std::vector<Contact> out;
 for(const auto& t:s.triangles){if((t.lo.array()>hi.array()).any()||(t.hi.array()<lo.array()).any())continue;
  for(auto p:cylinder_triangle(wheel,t.points[0],t.points[1],t.points[2],r,h,margin)){
   if(!p.analytic&&!exposed_edge_normal(s,t,p,wheel.translation(),bound)){++rejected;continue;}
   bool duplicate=false;for(const auto& old:out)if((old.point-p.point).squaredNorm()<1e-12&&old.normal.dot(p.normal)>.999999){duplicate=true;break;}
   if(!duplicate)out.push_back(p);
  }
 }
 return out;
}
}
