cmake_minimum_required(VERSION 3.18)

project(CubeCobraRecommenderGenerator VERSION "0.0.1")

if(SKBUILD)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LBIRARY "${PYTHON_LIBRARY}")
  set(DUMMY "${PYTHON_VERSION_STRING}")  # Not needed, silences a warning
endif()

set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT
)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(extern/concurrentqueue)
add_subdirectory(extern/mio)

pybind11_add_module(generator MODULE cpp/generator.cpp)

target_compile_definitions(generator PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_include_directories(generator PRIVATE extern/pcg-cpp/include)

target_link_libraries(generator PRIVATE concurrentqueue mio::mio)

target_compile_features(generator PRIVATE cxx_std_20)

install(TARGETS generator DESTINATION .)
