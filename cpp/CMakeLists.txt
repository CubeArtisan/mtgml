cmake_minimum_required(VERSION 3.18)

project(MtgmlNativeGenerators VERSION "0.0.1")

set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

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

pybind11_add_module(draftbot_generator MODULE src/draftbots_generator.cpp)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(draftbot_generator PUBLIC -fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(draftbot_generator PUBLIC -fcolor-diagnostics)
endif ()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_compile_options(draftbot_generator PRIVATE /W2 /arch:AVX2)
else()
  target_compile_options(draftbot_generator PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-missing-braces -mavx2)
endif()
target_compile_definitions(draftbot_generator PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(draftbot_generator PRIVATE extern/pcg-cpp/include include)
target_link_libraries(draftbot_generator PRIVATE concurrentqueue mio::mio)
target_compile_features(draftbot_generator PRIVATE cxx_std_20)
install(TARGETS draftbot_generator DESTINATION .)

pybind11_add_module(recommender_generator MODULE src/recommender_generator.cpp)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(recommender_generator PUBLIC -fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(recommender_generator PUBLIC -fcolor-diagnostics)
endif ()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_compile_options(recommender_generator PRIVATE /W2 /arch:AVX2)
else()
  target_compile_options(recommender_generator PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-missing-braces -mavx2)
endif()
target_compile_definitions(recommender_generator PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(recommender_generator PRIVATE extern/pcg-cpp/include include)
target_link_libraries(recommender_generator PRIVATE concurrentqueue mio::mio)
target_compile_features(recommender_generator PRIVATE cxx_std_20)
install(TARGETS recommender_generator DESTINATION .)

pybind11_add_module(adj_mtx_generator MODULE src/adj_mtx_generator.cpp)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(adj_mtx_generator PUBLIC -fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(adj_mtx_generator PUBLIC -fcolor-diagnostics)
endif ()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_compile_options(adj_mtx_generator PRIVATE /W2 /arch:AVX2)
else()
  target_compile_options(adj_mtx_generator PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-missing-braces -mavx2)
endif()
target_compile_definitions(adj_mtx_generator PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(adj_mtx_generator PRIVATE extern/pcg-cpp/include include)
target_link_libraries(adj_mtx_generator PRIVATE concurrentqueue mio::mio)
target_compile_features(adj_mtx_generator PRIVATE cxx_std_20)
install(TARGETS adj_mtx_generator DESTINATION .)
