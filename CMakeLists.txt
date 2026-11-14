cmake_minimum_required(VERSION 3.20)
project(keel_audio_formats VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_AUDIO_FORMATS_NO_EXCEPTIONS_RTTI "Compile keel_audio_formats without C++ exceptions and RTTI" OFF)
option(KEEL_AUDIO_FORMATS_BUILD_TESTS "Build keel_audio_formats unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})
option(KEEL_AUDIO_FORMATS_UNITY_BUILD "Compile keel_audio_formats with CMake UNITY_BUILD enabled" OFF)

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    set(KEEL_NO_EXCEPTIONS_RTTI ${KEEL_AUDIO_FORMATS_NO_EXCEPTIONS_RTTI} CACHE BOOL "" FORCE)
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        042c102203a5d07389a1e974b0a9c342503519c6
    )
    FetchContent_MakeAvailable(keel)
endif()

add_library(keel_audio_formats STATIC
    src/audio.cpp
    src/wav.cpp
    src/flac.cpp
    src/mp3.cpp
    src/aiff.cpp
    src/au.cpp
    src/midi.cpp
)
add_library(keel::audio_formats ALIAS keel_audio_formats)

target_link_libraries(keel_audio_formats PUBLIC keel::keel)
target_compile_features(keel_audio_formats PUBLIC cxx_std_20)
set_target_properties(keel_audio_formats PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
if(KEEL_AUDIO_FORMATS_UNITY_BUILD)
    set_target_properties(keel_audio_formats PROPERTIES UNITY_BUILD ON)
endif()

target_include_directories(keel_audio_formats PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(keel_audio_formats PRIVATE /W4 /utf-8)
    if(KEEL_AUDIO_FORMATS_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_audio_formats PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_audio_formats PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_audio_formats PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_AUDIO_FORMATS_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_audio_formats PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_audio_formats EXPORT keel_audio_formatsTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_audio_formatsTargets
    FILE keel_audio_formatsTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_audio_formats
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_audio_formatsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_audio_formatsConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_audio_formats
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_audio_formatsConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_audio_formatsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_audio_formatsConfigVersion.cmake"
    DESTINATION lib/cmake/keel_audio_formats
)

if(KEEL_AUDIO_FORMATS_BUILD_TESTS)
    enable_testing()

    add_executable(test_audio_formats tests/test_audio_formats.cpp)
    target_link_libraries(test_audio_formats PRIVATE keel::audio_formats)
    if(MSVC)
        target_compile_options(test_audio_formats PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_audio_formats PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME test_audio_formats COMMAND test_audio_formats)

    get_target_property(KEEL_CORE_INC_DIRS keel::keel INTERFACE_INCLUDE_DIRECTORIES)
    set(KEEL_CORE_INC_CLEAN "")
    foreach(IDIR IN LISTS KEEL_CORE_INC_DIRS)
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
        if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
            list(APPEND KEEL_CORE_INC_CLEAN "${IDIR_STRIPPED}")
        endif()
    endforeach()
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_CORE_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
