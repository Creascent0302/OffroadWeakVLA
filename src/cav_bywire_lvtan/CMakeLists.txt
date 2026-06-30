cmake_minimum_required(VERSION 3.8)
project(cav_bywire_lvtan)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(cav_msgs REQUIRED)
find_package(can_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  include
  /opt/xproj_deploy/include_public
)

file(GLOB SRC "src/*.cpp" "src/*/*.cpp")

add_executable(${PROJECT_NAME} ${SRC})

ament_target_dependencies(${PROJECT_NAME}
  cav_msgs
  can_msgs
  rclcpp
  std_msgs
)

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
