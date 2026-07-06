cmake_minimum_required(VERSION 3.20.0)

project(ondrix LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_library(ONDRIX_TERMINFO_LIBRARY NAMES terminfo tinfo curses ncurses ncursesw)
if(ONDRIX_TERMINFO_LIBRARY AND NOT TARGET Terminfo::terminfo)
  add_library(Terminfo::terminfo UNKNOWN IMPORTED)
  set_target_properties(Terminfo::terminfo PROPERTIES
    IMPORTED_LOCATION "${ONDRIX_TERMINFO_LIBRARY}")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")

if(NOT LLVM_VERSION_MAJOR EQUAL 17)
  message(FATAL_ERROR "ondrix currently requires LLVM/MLIR 17.x; found ${LLVM_PACKAGE_VERSION}")
endif()

if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL "17.0.6")
  message(WARNING "ondrix is tested with LLVM/MLIR 17.0.6; found ${LLVM_PACKAGE_VERSION}")
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(AddOndrix)

set(ONDRIX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ONDRIX_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${ONDRIX_SOURCE_DIR}/include)
include_directories(${ONDRIX_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
