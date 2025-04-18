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

std::cout << "[INITER] 创建模型中...\n" << std::endl;
try {
    // 模型指针
    auto* sample_model = new Model(sampleModelData[0]);
    std::cout << "--------------------\n\n" << "[INITER] 模型顶点数: " << sample_model->m_meshes[0].Vertices.size() << ", ";
    
    MODEL_POINTERS.push_back(sample_model);
    // 模型着色器指针
    MODEL_SHADER = new Shader(sampleModelData[1], sampleModelData[2]);
    // 检查包围球有效性
    if (sample_model->bounds.Rad < 0.01f) {
        std::cerr << "[ERROR] 模型包围球计算异常，可能未正确加载顶点数据" << std::endl;
    }

    std::cout << "模型创建任务结束" << std::endl;

} catch (const std::exception& e) {
    std::cerr << "[FATAL] 初始化失败: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
}

    return Window::GetWindow();
}

}