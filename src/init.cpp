// src/init.cpp
#include "init.h"
#include "core/camera.h"

namespace CubeDemo {

std::vector<Model*> MODEL_POINTERS;
Shader* MODEL_SHADER;

GLFWwindow* Init() {
    if (!glfwInit()) {
        std::cerr << "GLFW初始化失败" << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwSetErrorCallback([](int error, const char* what) {
        std::cerr << "GLFW错误 " << error << ": " << what << std::endl;
    });

    Window::Init(1280, 720, "Cube Demo");
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
        glfwDestroyWindow(Window::GetWindow());
        throw std::runtime_error("[Error] 窗口创建失败");
    }

    Camera::SaveCamera(camera);
    Inputs::Init(camera);

    // 测试模型
    string sampleModelData[] = {
        MODEL_PATH + string("sample/sample.obj"),
        VSH_PATH + string("model.glsl"),
        FSH_PATH + string("model.glsl")
    };

std::cout << "创建模型中..." << std::endl;

    // 模型指针
    MODEL_POINTERS.push_back( new Model(sampleModelData[0]) );
    // 模型着色器指针
    MODEL_SHADER = new Shader(sampleModelData[1], sampleModelData[2]);

std::cout << "模型创建任务结束" << std::endl;

    return Window::GetWindow();
}

}