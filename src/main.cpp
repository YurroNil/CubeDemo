#include <iostream>
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"

#include "core/camera.h"
#include "core/inputHandler.h"
#include "rendering/shader.h"
#include "rendering/modelLoader.h"
#include "rendering/mesh.h"
#include "rendering/renderer.h"

const unsigned int& SCR_INIT_WIDTH = 1280;
const unsigned int& SCR_INIT_HEIGHT = 720;


int main() {
    // 初始化窗口
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_INIT_WIDTH, SCR_INIT_HEIGHT, "Cube Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // 初始化渲染器
    Renderer::Init();

    //相机信息初始化
    Camera camera(
        glm::vec3(0.0f, 0.0f, 3.0f), // position
        glm::vec3(0.0f, 1.0f, 0.0f), // up
        -90.0f,                       // yaw
        0.0f                         // pitch
    );

    // 初始化系统
    InputHandler::Initialize(&camera, window);

    //添加窗口大小回调来更新window
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });

     // 加载模型
    ModelData cubeData;
     try {
        cubeData = ModelLoader::LoadFromJson(
            "../res/models/cube.json"
        );

    } catch (const std::exception& e) {
        std::cerr << "模型加载失败：" << e.what() << std::endl;
        return -1;
    }

     // 创建着色器
    Shader shader(
        ("../res/shaders/" + cubeData.shaders.vertexShader).c_str(),
        ("../res/shaders/" + cubeData.shaders.fragmentShader).c_str()
    );
    // 创建网格
    Mesh cubeMesh(cubeData);

    while (!glfwWindowShouldClose(window)) {

        // 输入处理
        InputHandler::Handling(window);

        //开始帧
        Renderer::BeginFrame();

        // 获取当前窗口尺寸
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        float aspectRatio = (height == 0) ? 1.0f :(static_cast<float>(width) / height);

        //应用相机参数
        Renderer::ApplyCamera(shader, camera, aspectRatio);

        // 提交渲染对象
        Renderer::Submit(shader, cubeMesh);
        
        // 结束帧
        Renderer::EndFrame(window);

        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}