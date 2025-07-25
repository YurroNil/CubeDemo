#src/CMakeLists.txt

# 收集所有源文件
file(GLOB_RECURSE SOURCES 
    "*.cpp" 
    "*.c"
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 添加预编译头文件
target_precompile_headers(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include/pch.h
)

# 抑制GCC警告
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(CubeDemo PRIVATE -Wno-psabi)
endif()
