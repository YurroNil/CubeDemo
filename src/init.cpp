// src/init.cpp

#include "init.h"

GLFWwindow* CubeDemo::Init() {
    if (!glfwInit()) {
        std::cerr << "GLFW初始化失败" << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwSetErrorCallback([](int error, const char* description) {
        std::cerr << "GLFW错误 " << error << ": " << description << std::endl;
    });

    WindowManager::Init(1280, 720, "Cube Demo");
    Renderer::Init();
    UIManager::Init();
    DebugInfoManager::Init();
    TextRenderer::Init();

    Camera* camera = new Camera(
        vec3(0.5f, 3.0f, 3.0f),
        vec3(0.0f, 1.0f, 0.0f),
        -90.0f,
        0.0f
    );
    Camera::SaveCamera(camera);

    InputHandler::Init(camera);
    
    DebugInfoManager::AddDebugInfo([&]{

        return "帧数FPS: " + std::to_string(TimeManager::FPS()) + 
                "  X: " + std::to_string(camera->Position.x) + 
                ", Y: " + std::to_string(camera->Position.y) + 
                ", Z: " + std::to_string(camera->Position.z);
    });


    ModelManager::Register(
        "cube",
        "../res/models/primitives/cube.json",
        ShaderLoader::s_vshPath + "lit.glsl",
        ShaderLoader::s_fshPath + "lit.glsl"
    );

    return WindowManager::GetWindow();
}
