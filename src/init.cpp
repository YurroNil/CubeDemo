// src/init.cpp
#include "init.h"
#include "mainProgramInc.h"
#include "core/camera.h"

namespace CubeDemo {



GLFWwindow* Init() {
    if (!glfwInit()) {
        std::cerr << "GLFW初始化失败" << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwSetErrorCallback([](int error, const char* description) {
        std::cerr << "GLFW错误 " << error << ": " << description << std::endl;
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
    ModelMng::Register(
        "cube",
        "../res/models/primitives/cube.json",
        ShaderLoader::s_vshPath + "lit.glsl",
        ShaderLoader::s_fshPath + "lit.glsl"
    );

    return WindowMng::GetWindow();
}

}