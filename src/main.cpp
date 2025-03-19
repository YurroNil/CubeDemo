#include <iostream>
#include "3rd-lib/glad/glad.h"
#include "3rd-lib/GLFW/glfw3.h"
#include "3rd-lib/glm/glm.hpp"
#include "3rd-lib/glm/gtc/matrix_transform.hpp"
#include "3rd-lib/glm/gtc/type_ptr.hpp"

#include "core/windowManager.h"
#include "core/camera.h"
#include "core/inputHandler.h"
#include "rendering/shader.h"
#include "rendering/modelLoader.h"
#include "rendering/mesh.h"

const unsigned int& SCR_INIT_WIDTH = 1280;
const unsigned int& SCR_INIT_HEIGHT = 720;


int main() {
    // 初始化窗口
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_INIT_WIDTH, SCR_INIT_HEIGHT, "Genshin Simulator", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);


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
        static float s_lastFrame = 0.0f;
        float currentFrame = glfwGetTime();
        float deltaTime = currentFrame - s_lastFrame;
        s_lastFrame = currentFrame;
        InputHandler::ProcessKeyboard(window, deltaTime);

        // 渲染
        glClearColor(0.8f, 0.94f, 0.98f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 获取当前窗口尺寸
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        float aspectRatio = (height == 0) ? 1.0f :(static_cast<float>(width) / height);

        // 更新渲染指令
        shader.Use();
        // 传入宽高比
        shader.ApplyCamera(
            camera,
            &aspectRatio//避免最小化窗口除以0的错误
        );
        
        shader.SetMat4("model", glm::mat4(1.0f));
        cubeMesh.Draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}