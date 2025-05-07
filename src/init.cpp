// src/init.cpp
#include "init.h"

namespace CubeDemo {

// 乱七八糟的别名
using csclock = std::chrono::steady_clock;

// 全局变量
std::vector<Model*> MODEL_POINTERS;
Shader* MODEL_SHADER;
bool DEBUG_ASYNC_MODE = false;

// Init函数
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
    // 测试模型
    string sampleModelData[] = {
        MODEL_PATH + string("sample/sample.obj"),
        VSH_PATH + string("model.glsl"),
        FSH_PATH + string("model.glsl")
    };

std::cout << "\n[INITER] 创建模型中..." << std::endl;
try {
    
    std::atomic<bool> modelLoaded{false};
    Model* sample_model = new Model(sampleModelData[0]);
    
    if (DEBUG_ASYNC_MODE == true) sample_model->LoadAsync([&]{ modelLoaded.store(true); }); // 加载模型（异步模式）
    else sample_model->LoadSync([&]{ modelLoaded.store(true); }); // 加载模型（同步模式）

    while(!modelLoaded.load()) {
        int processed = 0; TaskQueue::ProcTasks(processed);
        
        // 超时机制
        static auto deadline = csclock::now() + std::chrono::seconds(3);

        if(csclock::now() > deadline) {
            throw std::runtime_error("模型加载超时");
        }

        // 事件驱动等待
        if (processed == 0) { std::this_thread::yield(); }
    }

    MODEL_POINTERS.push_back(sample_model);

    // 模型着色器指针
    MODEL_SHADER = new Shader(sampleModelData[1], sampleModelData[2]);
    // 检查包围球有效性
    if (sample_model->bounds.Rad < 0.01f) {
        std::cerr << "[ERROR] 模型包围球计算异常，可能未正确加载顶点数据×" << std::endl;
    }

std::cout << "[INITER] 模型创建任务结束" << std::endl;

} catch (const std::exception& e) {
    std::cerr << "[FATAL] 初始化失败: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
}
std::cout << "[INITER] 初始化阶段结束" << std::endl;
    return Window::GetWindow();
}

}