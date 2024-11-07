cmake_minimum_required(VERSION 3.8)
project(detect_openvn)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
# Temukan paket OpenVINO
find_package(OpenVINO REQUIRED)
# Temukan paket OpenCV
find_package(OpenCV REQUIRED)
find_package(detect_messages REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

ament_export_dependencies(rosidl_default_runtime)

# rosidl_generate_interfaces(${PROJECT_NAME}
#   "../detect_messages/msg/DetectObject.msg"  
# )

add_executable(main src/main1.cpp)

ament_target_dependencies(main
  rclcpp
  detect_messages
  OpenCV
  OpenVINO
)

install(TARGETS
  main
  DESTINATION lib/${PROJECT_NAME}
)

# Tambahkan direktori include untuk OpenCV
target_include_directories(main PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

# Tautkan library OpenCV ke target main
target_link_libraries(main
  ${OpenCV_LIBRARIES}
  openvino::runtime
)

# install(DIRECTORY include/
#   DESTINATION include
# )
# install(TARGETS main
#   DESTINATION lib/${PROJECT_NAME})

# ament_export_include_directories(include)
ament_export_dependencies(rclcpp detect_messages)

ament_package()
