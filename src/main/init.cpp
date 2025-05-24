// src/main/init.cpp
#include "main/init.h"
#include "loaders/modelIniter.h"

namespace CubeDemo {

// 全局变量
std::vector<Model*> MODEL_POINTERS;
Shader* MODEL_SHADER;
bool DEBUG_ASYNC_MODE = false;  // 暂时采用同步模式
bool DEBUG_LOD_MODE = false;    // 暂时不采用LOD系统

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

/* ---------- 摄像机初始化 ------------ */
    Camera* camera = new Camera(
        vec3(0.5f, 0.5f, 3.0f),
        vec3(0.0f, 1.0f, 0.0f),
        -90.0f,
        0.0f
    );

    if (!camera) { glfwTerminate(); glfwDestroyWindow(Window::GetWindow()); throw std::runtime_error("[Error] 窗口创建失败"); }
    Camera::SaveCamera(camera); Inputs::Init(camera);

/* ---------- 模型初始化 ------------ */

    Loaders::ModelIniter::InitModels();

/* ---------- 结束 ------------ */
    std::cout << "[INITER] 初始化阶段结束" << std::endl;
    return Window::GetWindow();
}

}   // namespace CubeDemo
