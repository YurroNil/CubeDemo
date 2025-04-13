// src/init.cpp
#include "init.h"
#include "core/camera.h"

namespace CubeDemo {

std::vector<Model*> ModelPointers;


GLFWwindow* Init() {
    if (!glfwInit()) {
        std::cerr << "GLFW初始化失败" << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwSetErrorCallback([](int error, const char* what) {
        std::cerr << "GLFW错误 " << error << ": " << what << std::endl;
    });

    WindowMng::Init(1280, 720, "Cube Demo");
    Renderer::Init();
    UIMng::Init();

    Camera* camera = new Camera(
        vec3(0.5f, 3.0f, 3.0f),
        vec3(0.0f, 1.0f, 0.0f),
        -90.0f,
        0.0f
    );
    if (!camera) {
        glfwTerminate();
        glfwDestroyWindow(WindowMng::GetWindow());
        throw std::runtime_error("[Error] 窗口创建失败");
    }

    Camera::SaveCamera(camera);
    InputHandler::Init(camera);

    std::cout << "创建模型中..." << std::endl;
    ModelPointers.push_back(
        new Model("../res/models/sample/sample.obj")
    );
    std::cout << "模型创建任务结束" << std::endl;

    return WindowMng::GetWindow();
}

}