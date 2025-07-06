// src/main/init.cpp
#include "pch.h"
#include "main/init_inc.h"
#include "main/global_variables.h"

namespace CubeDemo {

// Init函数
GLFWwindow* Init(int argc, char* argv[]) {

    /* 解析参数 */
    if(argc != 0) parsing_arguments(argc, argv);

    /* 程序核心初始化 */
    init_program_core();

    /* 基本模块初始化 */
    WINDOW::Init(1920, 1080, "Cube Demo");
    Renderer::Init();
    UIMng::Init();

    /* 场景与预制体初始化 */
    init_managers();

    /* 摄像机初始化 */
    init_camera();

    /* ---------- 模型初始化 ------------ */

    // 场景管理器初始化
    SCENE_MNG->Init();
    // 注意：场景管理器的初始化会根据当前场景状态，来自动调用内部场景实例的初始化.
    // 然后又因为内部场景实例的初始化，会进行一系列的场景、着色器、光源等资源初始化的加载，所以再场景实例初始化中会执行：MIL::InitModels();函数进行初始化模型

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

    // 线程初始化
    CubeDemo::Loaders::Resource::Init(1);

/* ---------- OpenGL初始化 ------------ */
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

    // 创建阴影
    SHADOW_MAP = ShadowMap::CreateShadow();
    // 创建阴影着色器
    SHADOW_MAP->CreateShader();
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
