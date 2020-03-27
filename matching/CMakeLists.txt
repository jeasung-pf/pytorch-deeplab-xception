# Software License Agreement (BSD License)
#
# Copyright 2020 JeaSung (jeasung@peoplefund.co.kr). All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.5.1)

project(matching CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

option(HAVE_OPENCV_GPU "Check whether to compile in CPU/GPU mode" OFF)
option(DEBUG_DESC_PROGRESS "Execute verbosely" OFF)
option(PR_DEBUG "Execute verbosely" OFF)
if (HAVE_OPENCV_GPU)
    add_definitions(-DHAVE_OPENCV_GPU)
    message(STATUS "Will be built with GPU support")
endif ()
if (DEBUG_DESC_PROGRESS)
    add_definitions(-DDEBUG_DESC_PROGRESS)
    message(STATUS "Will print messages verbosely")
endif ()
if (PR_DEBUG)
    add_definitions(-DPR_DEBUG)
endif ()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

# One way to build a projects that uses gRPC is to just include the
# entire gRPC project tree via "add_subdirectory".
# This approach is very simple to use, but the are some potential
# disadvantages:
# * it includes gRPC's CMakeLists.txt directly into your build script
#   without and that can make gRPC's internal setting interfere with your
#   own build.
# * depending on what's installed on your system, the contents of submodules
#   in gRPC's third_party/* might need to be available (and there might be
#   additional prerequisites required to build them). Consider using
#   the gRPC_*_PROVIDER options to fine-tune the expected behavior.
#
# A more robust approach to add dependency on gRPC is using
# cmake's ExternalProject_Add (see cmake_externalproject/CMakeLists.txt).

# Include the gRPC's cmake build (normally grpc source code would live
# in a git submodule called "third_party/grpc", but this example lives in
# the same repository as gRPC sources, so we just look a few directories up)
add_subdirectory(./grpc ${CMAKE_CURRENT_BINARY_DIR}/grpc/cmake/build)
message(STATUS "Using gRPC via add_subdirectory.")

# After using add_subdirectory, we can now use the grpc targets directly from
# this build.
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
if (CMAKE_CROSSCOMPILING)
    find_program(_PROTOBUF_PROTOC protoc)
else ()
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif ()
set(_GRPC_GRPCPP_UNSECURE grpc++_unsecure)
if (CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else ()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif ()


# Proto file
get_filename_component(matching_proto "${CMAKE_CURRENT_BINARY_DIR}/../protocol/matching.proto" ABSOLUTE)
get_filename_component(gateway_proto "${CMAKE_CURRENT_BINARY_DIR}/../../gateway/protocol/gateway.proto" ABSOLUTE)
get_filename_component(matching_proto_path "${matching_proto}" PATH)
get_filename_component(gateway_proto_path "${gateway_proto}" PATH)

# Generated sources of current matching server
set(matching_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/matching.pb.cc")
set(matching_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/matching.pb.h")
set(matching_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/matching.grpc.pb.cc")
set(matching_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/matching.grpc.pb.h")
# Generated sources of gateway server for reference
set(gateway_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/gateway.pb.cc")
set(gateway_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/gateway.pb.h")

#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gateway/protocol)
#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matching/protocol)

add_custom_command(
        OUTPUT "${gateway_proto_srcs}" "${gateway_proto_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}/"
        -I"${gateway_proto_path}"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/grpc/third_party/googleapis"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/grpc/third_party/protobuf/src"
        "${gateway_proto}"
        DEPENDS "${gateway_proto}")
add_custom_command(
        OUTPUT "${matching_proto_srcs}" "${matching_proto_hdrs}" "${matching_grpc_srcs}" "${matching_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}/"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}/"
        -I"${matching_proto_path}"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/../gateway/protocol"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/grpc/third_party/googleapis"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/grpc/third_party/protobuf/src"
        -I"${CMAKE_CURRENT_BINARY_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${matching_proto}"
        DEPENDS "${matching_proto}")
message(STATUS ${OpenCV_LIBS})

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/grpc/third_party/googleapis/gens")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/grpc/third_party/protobuf/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/grpc/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${OpenCV_INCLUDE_DIRS}")

add_library(utility STATIC
        ${gateway_proto_srcs}
        ${gateway_proto_hdrs}
        ${matching_proto_srcs}
        ${matching_proto_hdrs}
        ${matching_grpc_srcs}
        ${matching_grpc_hdrs})
add_library(vocdata STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/data/VOCData.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/data/VOCData.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/data/VOCDataProtected.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/data/VOCDataUtility.cpp)
add_library(params STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/params.cpp)
add_library(defines STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/defines.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/defines.cpp)

# Targets greeter_[async_](client|server)
foreach (_target
        matching_client matching_server train_classifier warpHomography)
    add_executable(${_target} "${_target}.cpp"
            ${CMAKE_CURRENT_SOURCE_DIR}/src/Base.cpp
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/Bootstrap.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/Classifier.cpp
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/defines.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp)
    target_link_libraries(${_target}
            ${_GRPC_GRPCPP_UNSECURE}
            ${_PROTOBUF_LIBPROTOBUF}
            ${OpenCV_LIBS}
            utility
            vocdata
            params
            defines)
endforeach ()

