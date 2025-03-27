// src/core/windowManager.cpp
#include "streams.h"
#include "core/inputHandler.h"


void WindowManager::Init(int width, int height, const char* title) {
    // 初始化窗口
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    s_Window = glfwCreateWindow(width, height, "Cube Demo", NULL, NULL);
    if (!s_Window) {
        glfwTerminate();
        throw std::runtime_error("窗口创建失败");
    }

    // 设置OpenGL上下文
    glfwMakeContextCurrent(s_Window);
    if (gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0) {
        throw std::runtime_error("GLAD初始化失败");
        glfwDestroyWindow(s_Window);
        glfwTerminate();
    }
    
    int winWidth, winHeight;
    glfwGetFramebufferSize(s_Window, &winWidth, &winHeight);

    s_InitMouseX = winWidth / 2.0f;
    s_InitMouseY = winHeight / 2.0f;

    // 设置GLFW回调（通过lambda转发到静态方法）
    glfwSetCursorPosCallback(s_Window, [](GLFWwindow* w, double x, double y) {
        InputHandler::MouseCallback(x, y);
    });
    glfwSetScrollCallback(s_Window, [](GLFWwindow* w, double x, double y) {
        InputHandler::ScrollCallback(y);
    });
    glfwSetInputMode(s_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // 添加窗口大小回调来更新窗口
    glfwSetFramebufferSizeCallback(s_Window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });
}


bool WindowManager::ShouldClose() {
    return glfwWindowShouldClose(s_Window);
}

void WindowManager::ToggleFullscreen(GLFWwindow* window) {
    if (!s_IsFullscreen) {
        // 保存窗口位置和尺寸
        glfwGetWindowPos(window, &s_WindowPosX, &s_WindowPosY);
        glfwGetWindowSize(window, &s_WindowWidth, &s_WindowHeight);

        // 切换到全屏
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    } else {
        // 恢复窗口
        glfwSetWindowMonitor(window, nullptr, s_WindowPosX, s_WindowPosY, s_WindowWidth, s_WindowHeight, GLFW_DONT_CARE);
    }
    s_IsFullscreen = !s_IsFullscreen;
}

void WindowManager::FullscreenTrigger(GLFWwindow* window) {
    static bool f11LastState = false;
    bool f11CurrentState = glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS;

    if (f11CurrentState && !f11LastState) { ToggleFullscreen(window); }
    f11LastState = f11CurrentState;
}