// src/main/init.cpp

#include "main/init.h"
#include "loaders/modelIniter.h"

namespace CubeDemo {

// 全局变量
ModelPtrArray MODEL_POINTERS;
Shader* MODEL_SHADER;
Scene* SCENE_INST;
ShadowMap* SHADOW_MAP;

// 暂时采用同步模式, 以及不采用LOD系统
bool DEBUG_ASYNC_MODE = false, DEBUG_LOD_MODE = false;

// Init函数
GLFWwindow* Init() {

/* ---------- OpenGL初始化 ------------ */
    if (!glfwInit()) {
        std::cerr << "GLFW初始化失败" << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwSetErrorCallback([](int error, const char* what) {
        std::cerr << "GLFW错误 " << error << ": " << what << std::endl;
    });


/* ---------- 基本模块初始化 ------------ */
    Window::Init(1280, 720, "Cube Demo");
    Renderer::Init();
    UIMng::Init();

/* ---------- 场景与预制体初始化 ------------ */

    // 创建场景管理器
    SCENE_INST = Scene::CreateSceneInst();
    // 设置场景为默认场景
    SCENE_INST->Current = SceneID::NIGHT;
    // 创建阴影
    SHADOW_MAP = ShadowMap::CreateShadow();
    // 创建阴影着色器
    SHADOW_MAP->CreateShader();

/* ---------- 摄像机初始化 ------------ */
    Camera* camera = new Camera(
        vec3(0.5f, 0.5f, 3.0f),
        vec3(0.0f, 1.0f, 0.0f),
        -90.0f,
        0.0f
    );

    if (!camera) {
        glfwTerminate();
        glfwDestroyWindow(Window::GetWindow());
        throw std::runtime_error("[Error] 窗口创建失败");
    }
    Camera::SaveCamera(camera);
    Inputs::Init(camera);

/* ---------- 模型初始化 ------------ */

    Loaders::ModelIniter::InitModels();

/* ---------- 结束 ------------ */
    std::cout << "[INITER] 初始化阶段结束" << std::endl;
    return Window::GetWindow();
}

}   // namespace CubeDemo
