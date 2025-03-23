// src/main.cpp
#include <iostream>

// 第三方库
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"
#include "glm/gtx/dual_quaternion.hpp"
#include "glad/glad.h"

// 项目包含的头文件
#include "core/inputHandler.h"
#include "rendering/modelLoader.h"
#include "renderer/main.h"
#include "ui/uiManager.h"
#include "core/timeManager.h"


int main() {

    if (!glfwInit()) {
        std::cerr << "GLFW初始化失败" << std::endl;
        return -1;
    }
    glfwSetErrorCallback([](int error, const char* description) {
        std::cerr << "GLFW错误 " << error << ": " << description << std::endl;
    });


    // 初始化(窗口, 渲染器, 界面, 文字)
    WindowManager::Init(1280, 720, "Cube Demo");
    Renderer::Init();
    UIManager::Init();
    TextRenderer::Init();

    // 相机信息初始化
    Camera camera(
        vec3(0.5f, 3.0f, 3.0f), // position
        vec3(0.0f, 1.0f, 0.0f), // up
        -90.0f,                       // yaw
        0.0f                         // pitch
    );

    InputHandler::Init(&camera);
    
    // 注册调试信息
    UIManager::AddDebugInfo([&]{
    return "FPS: " + std::to_string(TimeManager::FPS()) + "  X: " + std::to_string(camera.Position.x) + ", Y: " + std::to_string(camera.Position.y) + ", Z: " + std::to_string(camera.Position.z);
    });

     // 加载模型
    ModelData cubeData;
    ModelLoader modelLoader;
    try {
        cubeData = modelLoader.LoadFromJson("res/models/lit_cube.json");
        //cubeData = modelLoader.LoadFromJson("res/models/" + cubeData.name + ".json");

    } catch (const std::exception& e) {
        std::cerr << "模型加载失败：" << e.what() << std::endl;
        return -1;
    }

     // 创建着色器
    Shader shader(
        ("res/shaders/" + cubeData.shaders.vertexShader).c_str(),
        ("res/shaders/" + cubeData.shaders.fragmentShader).c_str()
    );

    Mesh cubeMesh(cubeData);  //创建网格
    GLFWwindow* window = WindowManager::GetWindow();

    // 主循环
    while (!WindowManager::ShouldClose()) {

        // 输入处理
        TimeManager::Update();
        InputHandler::ProcessKeyboard( window, TimeManager::DeltaTime() );
        Renderer::BeginFrame();  //开始帧

        int width, height;
        glfwGetWindowSize(window, &width, &height);  // 获取当前窗口尺寸

        Renderer::ApplyCamera(  //应用相机参数
            shader,
            camera,
            (height == 0) ? 1.0f :(static_cast<float>(width) / height)
        );

        WindowManager::FullscreenTrigger(window);  // F11全屏检测
        Renderer::SetLitParameter(shader, camera, cubeData);
        Renderer::Submit(shader, cubeMesh);  // 提交渲染对象
        UIManager::RenderUI();  // 渲染调试信息
        Renderer::EndFrame(window);  // 结束帧

        glfwPollEvents();
    }

    std::cout << "程序正常退出" << std::endl;
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}