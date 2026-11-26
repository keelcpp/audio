cmake_minimum_required(VERSION 3.20)
project(keel_audio VERSION 0.2.0 LANGUAGES CXX)

option(KEEL_AUDIO_NO_EXCEPTIONS_RTTI "Compile keel_audio without C++ exceptions and RTTI" OFF)
option(KEEL_AUDIO_BUILD_TESTS "Build keel_audio unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    if(KEEL_AUDIO_NO_EXCEPTIONS_RTTI)
        set(KEEL_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        4b37783925fe7af7a64daaff90f268df267fbbf3
    )
    FetchContent_MakeAvailable(keel)
endif()

if(NOT TARGET keel::interp)
    find_package(keel_interp QUIET CONFIG)
    if(TARGET keel::keel_interp AND NOT TARGET keel::interp)
        add_library(keel::interp ALIAS keel::keel_interp)
    endif()
endif()
if(NOT TARGET keel::interp)
    include(FetchContent)
    set(KEEL_INTERP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../interp/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_INTERP)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_INTERP "${CMAKE_CURRENT_SOURCE_DIR}/../interp" CACHE PATH "Local keel_interp checkout")
    endif()
    if(KEEL_AUDIO_NO_EXCEPTIONS_RTTI)
        set(KEEL_INTERP_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel_interp
        GIT_REPOSITORY https://github.com/keelcpp/interp.git
        GIT_TAG        72d23db74ec41ee929c8535ac419624461f3eb4d
    )
    FetchContent_MakeAvailable(keel_interp)
endif()

if(NOT TARGET keel::fft)
    find_package(keel_fft QUIET CONFIG)
    if(TARGET keel::keel_fft AND NOT TARGET keel::fft)
        add_library(keel::fft ALIAS keel::keel_fft)
    endif()
endif()
if(NOT TARGET keel::fft)
    include(FetchContent)
    set(KEEL_FFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../fft/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_FFT)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_FFT "${CMAKE_CURRENT_SOURCE_DIR}/../fft" CACHE PATH "Local keel_fft checkout")
    endif()
    if(KEEL_AUDIO_NO_EXCEPTIONS_RTTI)
        set(KEEL_FFT_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel_fft
        GIT_REPOSITORY https://github.com/keelcpp/fft.git
        GIT_TAG        8f27754ba6e5671439d97d3847479bdbf7ccfee7
    )
    FetchContent_MakeAvailable(keel_fft)
endif()

add_library(keel_audio STATIC
    src/scales.cpp
    src/descriptors.cpp
    src/features.cpp
    src/cepstral.cpp
    src/pitch.cpp
)
add_library(keel::audio ALIAS keel_audio)

target_link_libraries(keel_audio
    PUBLIC  keel::keel
    PRIVATE keel::interp keel::fft
)

target_compile_features(keel_audio PUBLIC cxx_std_20)
set_target_properties(keel_audio PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(keel_audio PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(keel_audio PRIVATE /W4 /utf-8)
    if(KEEL_AUDIO_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_audio PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_audio PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_audio PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_AUDIO_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_audio PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

if(COMMAND keel_enforce_guidelines)
    keel_enforce_guidelines(keel_audio
        SCOPE ALL
        BAN_RAW_INTS ALL
        BAN_RAW_FLOATS ALL
        HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
        STANDALONE_HEADERS ${KEEL_AUDIO_BUILD_TESTS}
        TEST_NAME test_guidelines_headers
        WARNINGS OFF
    )
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_audio EXPORT keel_audioTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_audioTargets
    FILE keel_audioTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_audio
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_audioConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_audioConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_audio
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_audioConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_audioConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_audioConfigVersion.cmake"
    DESTINATION lib/cmake/keel_audio
)

if(KEEL_AUDIO_BUILD_TESTS)
    enable_testing()

    add_executable(test_audio tests/test_audio.cpp)
    target_link_libraries(test_audio PRIVATE keel::audio)
    if(MSVC)
        target_compile_options(test_audio PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_audio PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(COMMAND keel_enforce_guidelines)
        keel_enforce_guidelines(test_audio
            SCOPE ALL
            BAN_RAW_INTS ALL
            BAN_RAW_FLOATS ALL
            WARNINGS OFF
        )
    endif()
    add_test(NAME test_audio COMMAND test_audio)

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
