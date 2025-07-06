# 必须放在文件最顶部
cmake_minimum_required(VERSION 3.15)

# 项目声明必须紧跟其后
project(CubeDemo LANGUAGES C CXX)

# 设置 C 和 C++ 标准
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)  # 使用C++23
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 包含目录设置 - 使用 target_include_directories 更安全
# 注意：include_directories 已过时，但我们在这里保留以便简单
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/.3rdParty/include
    $ENV{PDT_I}
    $ENV{PDT_I}/imgui
    $ENV{PDT_I}/assimp
    $ENV{PDT_I}/stb
)

# 链接目录设置 - 同样，link_directories 已过时，但简单
link_directories(
    ${CMAKE_SOURCE_DIR}/lib
    $ENV{PDT_L}
    $ENV{MINGW64}/lib
)

# 添加子目录
add_subdirectory(src)
add_subdirectory(.3rdParty)