#include "tplib/glad/glad.h"
#include "tplib/GLFW/glfw3.h"
#include "tplib/glm/glm.hpp"
#include "tplib/glm/gtc/matrix_transform.hpp"
#include "tplib/glm/gtc/type_ptr.hpp"

#include "core/windowManager.h"
#include "core/camera.h"
#include "core/inputHandler.h"
#include "rendering/shader.h"
#include "rendering/mesh.h"

const unsigned int SCR_INIT_WIDTH = 1280;
const unsigned int SCR_INIT_HEIGHT = 720;

float CUBE_VERTICES[] = {
     //正面(front)
    -0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,

     0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f,  0.5f,

    //左面(right)
    0.5f, -0.5f, -0.5f,
    0.5f, -0.5f,  0.5f,
    0.5f,  0.5f,  0.5f,
    0.5f,  0.5f,  0.5f,
    0.5f,  0.5f, -0.5f,
    0.5f, -0.5f, -0.5f,
    
    //后面(back)
    -0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
    -0.5f,  0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,

    //右面(left)
    -0.5f, -0.5f, -0.5f,
    -0.5f,  0.5f, -0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f,
    
    //底面(bottom)
    -0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,
    -0.5f, -0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f,

    //顶面(top)
    -0.5f, 0.5f, -0.5f,
     0.5f, 0.5f, -0.5f,
     0.5f, 0.5f,  0.5f,
     0.5f, 0.5f,  0.5f,
    -0.5f, 0.5f,  0.5f,
    -0.5f, 0.5f, -0.5f,

};


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

    // 初始化系统
    Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
    InputHandler::Initialize(&camera, window);

    //添加窗口大小回调来更新window
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });

    Shader shader("../shaders/general.vsh", "../shaders/general.fsh");
    Mesh cubeMesh(CUBE_VERTICES, sizeof(CUBE_VERTICES));

    while (!glfwWindowShouldClose(window)) {
        // 输入处理
        static float lastFrame = 0.0f;
        float currentFrame = glfwGetTime();
        float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        InputHandler::ProcessKeyboard(window, deltaTime);

        // 渲染
        glClearColor(0.8f, 0.94f, 0.98f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 获取当前窗口尺寸
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        float aspectRatio = static_cast<float>(width) / height;

        // 更新渲染指令
        shader.Use();
        // 传入宽高比
        shader.ApplyCamera(
            camera,
            (height == 0) ? 1.0f : aspectRatio//避免最小化窗口除以0的错误
        );
        
        shader.SetMat4("model", glm::mat4(1.0f));
        cubeMesh.Draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}