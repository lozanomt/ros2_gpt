cmake_minimum_required(VERSION 3.8)
project(ros2_gpt)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable for each node
add_executable(input_publisher src/input_publisher_node.cpp)
ament_target_dependencies(input_publisher rclcpp std_msgs)

add_executable(output_subscriber src/output_subscriber_node.cpp)
ament_target_dependencies(output_subscriber rclcpp std_msgs)

# Install Cpp executables
install(TARGETS
  input_publisher
  output_subscriber
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(include)

# Install Python executables
install(PROGRAMS
  src/gpt_trainer_node.py
  src/gpt_inference_node.py
  src/wikipedia_downloader.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install data and models folders
install(DIRECTORY
  data
  models
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(FILES
  launch.xml
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()