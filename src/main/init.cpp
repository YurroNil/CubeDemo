// src/main/init.cpp
#include "pch.h"
#include "main/init.h"
#include "loaders/model_initer.h"
#include "core/inputs.h"
// 管理器模块
#include "managers/lightMng.h"
#include "managers/modelMng.h"
#include "managers/uiMng.h"

namespace CubeDemo {

// 全局变量 (生命周期是到程序结束)

ModelPtrArray MODEL_POINTERS;

// 管理器
SceneMng* SCENE_MNG;
LightMng* LIGHT_MNG;
ModelMng* MODEL_MNG;
ShadowMap* SHADOW_MAP;


// 暂时采用同步模式
bool DEBUG_ASYNC_MODE = false;

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

    // 创建场景和光源管理器
    SCENE_MNG = SceneMng::CreateInst();
    LIGHT_MNG = LightMng::CreateInst();
    MODEL_MNG = ModelMng::CreateInst();

    // 设置场景为
    SCENE_MNG->Current = SceneID::NIGHT;
    // 创建阴影
    SHADOW_MAP = ShadowMap::CreateShadow();
    // 创建阴影着色器
    SHADOW_MAP->CreateShader();

/* ---------- 摄像机初始化 ------------ */
    Camera* camera = new Camera(
        vec3(0.5f, 0.5f, 3.0f),
        vec3(0.0f, 1.0f, 0.0f),
        -90.0f, 0.0f
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
