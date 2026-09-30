#pragma once
#include <PxPhysicsAPI.h>
#include "contact_kernel.hpp"
namespace bridge_contact {
std::vector<Contact> dispatch_contacts(const physx::PxTransform& wheel,const physx::PxGeometry& other,const physx::PxTransform& pose,double margin,const physx::PxCustomGeometry::Type* known_wheel_type);
unsigned long unsupported_pairs();
void clear_mesh_cache();
}
