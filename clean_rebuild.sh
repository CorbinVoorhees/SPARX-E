sudo rm -rf build/ install/ log/

# unset these env vars because they might have the wrong info if a pkg was deleted. 
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH

# resource the ros kilted setup
source /opt/ros/kilted/setup.bash

# now we rebuild
colcon build