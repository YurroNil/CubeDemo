# 收集所有源文件
file(GLOB_RECURSE SOURCES 
    "*.cpp" 
    "*.c"
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接第三方库
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        3rd_party
        glfw3
        opengl32
        assimp
)
