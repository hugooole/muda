# project
cmake_minimum_required(VERSION 3.20)

# include options and install dirs
include(cmake/options.cmake)

project(muda LANGUAGES CXX CUDA)
# must include after project()
include(GNUInstallDirs)

# to remove warning
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()


# ----------------------------------------------------------------------------
# muda
# ----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

add_library(muda INTERFACE)

# c++ standard >= 17
target_compile_features(muda INTERFACE cxx_std_17)

# include directories
if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    target_include_directories(muda INTERFACE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endif()
target_include_directories(muda INTERFACE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


# link libraries
target_link_libraries(muda INTERFACE CUDA::cudart)

# compile options
target_compile_options(muda INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 20012>
)

if(WIN32)
    target_compile_definitions(muda INTERFACE 
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 4819>
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 4068>
    )
endif()

if(MUDA_FORCE_CHECK)
    target_compile_definitions(muda INTERFACE "-DMUDA_CHECK_ON=1")
else()
    if(MUDA_WITH_CHECK)
        target_compile_definitions(muda INTERFACE
            $<$<CONFIG:Debug>:MUDA_CHECK_ON=1>
            $<$<CONFIG:RelWithDebInfo>:MUDA_CHECK_ON=1>)
    else()
        target_compile_definitions(muda INTERFACE "-DMUDA_CHECK_ON=0")
    endif()
endif()

if(MUDA_WITH_COMPUTE_GRAPH)
    target_compile_definitions(muda INTERFACE "-DMUDA_COMPUTE_GRAPH_ON=1")
else()
    target_compile_definitions(muda INTERFACE "-DMUDA_COMPUTE_GRAPH_ON=0")
endif()

if(MUDA_WITH_NVTX3)
    target_compile_definitions(muda INTERFACE "-DMUDA_NVTX3_ON=1")
else()
    target_compile_definitions(muda INTERFACE "-DMUDA_NVTX3_ON=0")
endif()

# header files
file(GLOB_RECURSE MUDA_HEADER_FILES "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/*.inl" "${PROJECT_SOURCE_DIR}/src/*.cuh")

# installation
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/muda" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS muda EXPORT mudaTargets)
install(EXPORT mudaTargets
    FILE mudaTargets.cmake
    NAMESPACE muda::
    DESTINATION share/muda
)
install(FILES
    "${PROJECT_SOURCE_DIR}/cmake/muda-config.cmake"
    DESTINATION share/muda
)

source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "Header Files" FILES ${MUDA_HEADER_FILES})


# ----------------------------------------------------------------------------
# example and test
# ----------------------------------------------------------------------------
if(MUDA_BUILD_EXAMPLE)
    add_subdirectory("example")
endif()

if(MUDA_BUILD_TEST)
    add_subdirectory("test")
endif()
