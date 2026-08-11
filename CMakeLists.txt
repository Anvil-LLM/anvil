cmake_minimum_required(VERSION 3.22)
project(anvil C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backends/llama-turbo)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(LLAMA_INSTALL_VERSION 0.0.1 CACHE STRING "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
if(NOT DEFINED GGML_METAL)
if(APPLE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
else()
set(GGML_METAL OFF CACHE BOOL "" FORCE)
endif()
endif()
if(NOT DEFINED GGML_VULKAN)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
endif()
if(NOT DEFINED GGML_CUDA)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama-build EXCLUDE_FROM_ALL)
include(FetchContent)
FetchContent_Declare(ftxui GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI GIT_TAG v7.0.1 GIT_SHALLOW TRUE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ftxui)

add_library(mdtty STATIC
    third_party/mdtty/src/mdtty.cpp
    third_party/mdtty/utf8proc/utf8proc.c
)
set_target_properties(mdtty PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    C_STANDARD 11
)
target_compile_definitions(mdtty PRIVATE UTF8PROC_STATIC)
target_include_directories(mdtty PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mdtty/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mdtty/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mdtty/utf8proc
)

if(MSVC)
    target_compile_options(mdtty PRIVATE /w)
else()
    target_compile_options(mdtty PRIVATE -w)
endif()

add_executable(anvil
    src/anvil.cpp
)
target_include_directories(anvil PRIVATE src ${LLAMA_DIR}/include ${LLAMA_DIR}/ggml/include ${LLAMA_DIR}/tools/mtmd)

target_include_directories(anvil PRIVATE SYSTEM ${LLAMA_DIR}/vendor)
add_subdirectory(${LLAMA_DIR}/tools/mtmd ${CMAKE_BINARY_DIR}/mtmd-build EXCLUDE_FROM_ALL)
target_link_libraries(anvil PRIVATE llama ggml ftxui::ftxui mdtty mtmd)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(anvil PRIVATE -Wall -Wextra -Wshadow -Wconversion -Wno-unused-parameter)
endif()
if(APPLE)

target_link_libraries(anvil PRIVATE "-framework Metal" "-framework Foundation" "-framework Accelerate" "-framework QuartzCore" "-framework CoreFoundation" "-framework IOKit" "-lobjc")
elseif(UNIX)
target_link_libraries(anvil PRIVATE pthread dl m)
endif()
if(WIN32)
target_link_libraries(anvil PRIVATE ws2_32)
endif()
install(TARGETS anvil RUNTIME DESTINATION bin)
