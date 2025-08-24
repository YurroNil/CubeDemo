// src/main/init.cpp
#include "pch.h"
#include "main/init_inc.h"
#include "main/global_variables.h"

namespace CubeDemo {

// Init函数
GLFWwindow* Init(int argc, char* argv[]) {

    /* 解析参数 */
    if(argc != 0) parsing_arguments(argc, argv);
    
    init_program_core();    // 程序核心
    WINDOW::Init(1920, 1080, "Cube Demo");
    Renderer::Init();       // 渲染器
    init_managers();        // 场景与预制体
    init_camera();          // 摄像机
    SCENE_MNG->Init();      // 场景管理器
    UIMng::Init();          // GUI

    /* ---------- 结束 ------------ */
    std::cout << "[INITER] 初始化阶段结束\n" << std::endl;

    // 返回窗口
    return WINDOW::GetWindow();
}

// 解析参数
void parsing_arguments(int argc, char* argv[]) {
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        
        if(arg == "-async") DEBUG_ASYNC_MODE = true;
        if(arg == "-debug1") DEBUG_INFO_LV = 1; // 调试信息详细等级: 1
        if(arg == "-debug2") DEBUG_INFO_LV = 2; // 调试信息详细等级: 2
    }
}
// 程序核心初始化(如GLFW, GLAD, ThreadModules, etc.)
void init_program_core() {

    // 分配线程数量. 同步模式=1, 异步模式=CPU核心数
    const unsigned int MAX_THREADS = DEBUG_ASYNC_MODE ? 1 : std::thread::hardware_concurrency();
    CubeDemo::Loaders::Resource::Init(MAX_THREADS);

/* ---------- OpenGL检查 ------------ */
    if (!glfwInit()) {
        std::cerr << "[INITER_ERROR] GLFW初始化失败" << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwSetErrorCallback([](int error, const char* what) {
        std::cerr << "[INITER_ERROR] GLFW错误 " << error << ": " << what << std::endl;
    });
}
// 初始化管理器
void init_managers() {
    // 创建场景和光源管理器
    SCENE_MNG = SceneMng::CreateInst();
    LIGHT_MNG = LightMng::CreateInst();
    MODEL_MNG = ModelMng::CreateInst();
}
// 初始化相机
void init_camera() {
    Camera* camera = new Camera(
        vec3(0.5f, 0.5f, 3.0f),
        vec3(0.0f, 1.0f, 0.0f),
        -90.0f, 0.0f
    );

    if (!camera) {
        glfwTerminate();
        glfwDestroyWindow(WINDOW::GetWindow());
        throw std::runtime_error("[INITER_ERROR] 窗口创建失败");
    }
    Camera::SaveCamera(camera);
}
}   // namespace CubeDemo
