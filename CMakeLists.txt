# 项目根目录的CMakeLists.txt文件
cmake_minimum_required(VERSION 3.15)

project(CubeDemo LANGUAGES C CXX)

# 设置C/C++标准库版本
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)  # C++23
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# 检查PDT_I环境变量
if(NOT DEFINED ENV{PDT_I})
    message(FATAL_ERROR "未设置环境变量$PDT_I!")
endif()

# 添加glad的源文件
add_library(glad STATIC
    "$ENV{PDT_I}/glad/glad.c"
)

# 设置Glad包含目录
target_include_directories(glad PUBLIC
    $ENV{PDT_I}
)

# 添加imgui的源文件
add_library(imgui STATIC
    "$ENV{PDT_I}/imgui/imgui.cpp"
    "$ENV{PDT_I}/imgui/imgui_draw.cpp"
    "$ENV{PDT_I}/imgui/imgui_tables.cpp"
    "$ENV{PDT_I}/imgui/imgui_widgets.cpp"
    "$ENV{PDT_I}/imgui/backends/imgui_impl_glfw.cpp"
    "$ENV{PDT_I}/imgui/backends/imgui_impl_opengl3.cpp"
)

# 设置imgui包含目录
target_include_directories(imgui PUBLIC
    $ENV{PDT_I}
    $ENV{PDT_I}/imgui
    $ENV{PDT_I}/imgui/backends
)

# 添加子目录(用于添加项目目录的源文件)
add_subdirectory(src)

# 设置 CubeDemo 目标的属性
set_target_properties(CubeDemo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    LINKER_LANGUAGE CXX
)

# 设置包含目录
target_include_directories(CubeDemo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{PDT_I}
    $ENV{PDT_I}/assimp
    $ENV{PDT_I}/stb
)

# 将环境变量转换为CMake路径格式
file(TO_CMAKE_PATH "$ENV{PDT_L}" PDT_L_DIR)
# 确保路径以斜杠结尾
if(NOT PDT_L_DIR MATCHES "/$")
    set(PDT_L_DIR "${PDT_L_DIR}/")
endif()
# 设置链接目录 - 使用绝对路径
target_link_directories(CubeDemo PRIVATE
    ${CMAKE_BINARY_DIR}/lib
    "${PDT_L_DIR}"
)
# 链接依赖库 - 确保使用正确的库名
target_link_libraries(CubeDemo PRIVATE 
    glad
    imgui
    "${PDT_L_DIR}libglfw3.a"
    "${PDT_L_DIR}libassimp.dll.a"
)

# 添加必要的宏定义
target_compile_definitions(CubeDemo PRIVATE
)
