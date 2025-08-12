# 项目根目录的CMakeLists.txt文件

# 注意事项
#   1: optiX 9.0+中没有任何静态链接库，只有头文件. 因此这里不需要设置任何optiX的链接库
#   2: 此项目只能使用mingw/GCC工具链构建, 因此生成模式为mingw makefiles.
#   3: 需要设置环境变量PDT_I, PDT_L, CUDA_PATH, OPTI_X. PDT_I/L是设置第三方库的include和静态链接库目录, CUDA_PATH和OPTI_X是设置这两个库的根目录.

cmake_minimum_required(VERSION 3.15)

project(CubeDemo LANGUAGES C CXX)

# 设置标准库为c23的版本
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


# 检查环境变量
if(NOT DEFINED ENV{PDT_I})
    message(FATAL_ERROR "未设置环境变量$PDT_I!")
endif()
if(NOT DEFINED ENV{PDT_L})
    message(FATAL_ERROR "未设置环境变量$PDT_L!")
endif()
if(NOT DEFINED ENV{CUDA_PATH})
    message(FATAL_ERROR "未设置环境变量$CUDA_PATH!")
endif()
if(NOT DEFINED ENV{OPTI_X})
    message(FATAL_ERROR "未设置环境变量$OPTI_X!")
endif()

# 将路径转换为CMake路径格式
file(TO_CMAKE_PATH "$ENV{PDT_I}" PDT_I_DIR)
file(TO_CMAKE_PATH "$ENV{PDT_L}" PDT_L_DIR)
file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/lib/x64" CUDA_LIB_DIR)
file(TO_CMAKE_PATH "$ENV{OPTI_X}/lib" OPTIX_LIB_DIR)
# 确保路径以斜杠结尾
if(NOT PDT_I_DIR MATCHES "/$" )
    set(PDT_I_DIR "${PDT_I_DIR}/")
endif()
if(NOT PDT_L_DIR MATCHES "/$" )
    set(PDT_L_DIR "${PDT_L_DIR}/")
endif()
if(NOT CUDA_LIB_DIR MATCHES "/$" )
    set(CUDA_LIB_DIR "${CUDA_LIB_DIR}/")
endif()
if(NOT OPTIX_LIB_DIR MATCHES "/$" )
    set(OPTIX_LIB_DIR "${OPTIX_LIB_DIR}/")
endif()

# 设置cuda和optiX库的路径
set(CUDA_INC $ENV{CUDA_PATH}/include)
set(OPTIX_INC $ENV{OPTI_X}/include)

# 添加glad的源文件
add_library(glad STATIC
    "${PDT_I_DIR}glad/glad.c"
)
# 设置Glad包含目录
target_include_directories(glad PUBLIC
    ${PDT_I_DIR}
)
# 添加imgui的源文件
add_library(imgui STATIC
    "${PDT_I_DIR}imgui/imgui.cpp"
    "${PDT_I_DIR}imgui/imgui_draw.cpp"
    "${PDT_I_DIR}imgui/imgui_tables.cpp"
    "${PDT_I_DIR}imgui/imgui_widgets.cpp"
    "${PDT_I_DIR}imgui/backends/imgui_impl_glfw.cpp"
    "${PDT_I_DIR}imgui/backends/imgui_impl_opengl3.cpp"
)
# 设置imgui包含目录
target_include_directories(imgui PUBLIC
    ${PDT_I_DIR}
    ${PDT_I_DIR}/imgui
    ${PDT_I_DIR}/imgui/backends
)

# 添加子目录(用于添加项目目录的源文件)
add_subdirectory(src)

# 设置CubeDemo目标的属性
set_target_properties(CubeDemo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    LINKER_LANGUAGE CXX

    # 启用设备符号解析
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON

    CUDA_RUNTIME_LIBRARY Shared
)

# 设置项目的include目录
target_include_directories(CubeDemo PRIVATE
    ${CMAKE_SOURCE_DIR}/include   # 项目include目录
    ${PDT_I_DIR}                  # 第三方库目录
    ${PDT_I_DIR}/assimp           # assimp库目录
    ${PDT_I_DIR}/stb              # stb库目录
    ${CUDA_INC}                   # CUDA库目录
    ${OPTIX_INC}                  # OptiX库目录
)

# 设置链接目录
target_link_directories(CubeDemo PRIVATE
    ${CMAKE_BINARY_DIR}/lib  # 项目lib目录
    "${PDT_L_DIR}"           # 第三方库目录
    "${CUDA_LIB_DIR}"        # CUDA库目录
    "${OPTIX_LIB_DIR}"       # OptiX库目录
)

# 链接静态链接库
target_link_libraries(CubeDemo PRIVATE 
    glad imgui glfw3 assimp.dll  # 项目依赖的基本静态库
    cudart cuda                  # CUDA
    setupapi                     # 修复windows API链接问题
)

# 添加必要的宏定义
target_compile_definitions(CubeDemo PRIVATE
)
