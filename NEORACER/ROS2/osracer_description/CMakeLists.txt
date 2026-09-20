cmake_minimum_required(VERSION 3.16)
project(osracer_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch meshes textures urdf
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
