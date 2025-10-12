
cmake_minimum_required(VERSION 3.4)
project(mp4module)

#add_subdirectory(pybind11)
# Find pybind11 configuration files
find_package(pybind11 REQUIRED)
# Create the Python module using the imported pybind11::module target
# add_executable(mp4Readerlib MpReader.cpp) # Your original C++ library
# Create the Python module
pybind11_add_module(mp4module SHARED Mp4PythonWrapper.cpp  MpReader.cpp)


# Add the include directory for the mp4module target
# Since mp4module.h is used both to compile the module itself and by Python users, use PUBLIC
target_include_directories(mp4module PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

#target_include_directories(mp4module PUBLIC &PWD/include/)
