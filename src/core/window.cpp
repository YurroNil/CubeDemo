// src/core/Window.cpp

// 项目头文件
#include "core/window.h"
#include "core/inputs.h"
#include "threads/taskQueue.h"

namespace CubeDemo {


void Window::Init(int width, int height, const char* title) {

    // 在初始化时捕获主线程ID
    TaskQueue::s_MainThreadId = std::this_thread::get_id();
    // 初始化窗口
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    s_Window = glfwCreateWindow(width, height, "Cube Demo", NULL, NULL);
    if (!s_Window) { glfwTerminate(); throw std::runtime_error("[Error] 窗口创建失败"); }
    // 设置OpenGL上下文
    glfwMakeContextCurrent(s_Window);
    if (gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0) {
        throw std::runtime_error("[Error] GLAD初始化失败");
        glfwDestroyWindow(s_Window);
        glfwTerminate();
    }
    
    int winWidth, winHeight; glfwGetFramebufferSize(s_Window, &winWidth, &winHeight);
    s_InitMouseX = winWidth / 2.0f; s_InitMouseY = winHeight / 2.0f;
    // 设置GLFW回调（通过lambda转发到静态方法）
    glfwSetCursorPosCallback(s_Window, [](GLFWwindow* w, double x, double y) { Inputs::MouseCallback(x, y); });
    glfwSetScrollCallback(s_Window, [](GLFWwindow* w, double x, double y) { Inputs::ScrollCallback(y); });
    glfwSetInputMode(s_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // 添加窗口大小回调来更新窗口
    glfwSetFramebufferSizeCallback(s_Window, [](GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); }); 
}

bool Window::ShouldClose() { return glfwWindowShouldClose(s_Window); }

void Window::ToggleFullscreen(GLFWwindow* window) {
    if(!window) return; // 防止空指针
    if (!s_IsFullscreen) {
        // 保存窗口位置和尺寸
        UpdateWindowSize(window); UpdateWindowPos(window);
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

void Window::FullscreenTrigger(GLFWwindow* window) {
    static bool f11LastState = false;
    bool f11CurrentState = glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS;
    if (f11CurrentState && !f11LastState) { ToggleFullscreen(window); }
    f11LastState = f11CurrentState;
}

void Window::UpdateWindowSize(GLFWwindow* window) {
    // 更新成员的值
    glfwGetWindowSize(window, &s_WindowWidth, &s_WindowHeight);
}

void Window::UpdateWindowPos(GLFWwindow* window) {
    // 更新成员的值
    glfwGetWindowPos(window, &s_WindowPosX, &s_WindowPosY);
}

float Window::GetAspectRatio() {
    return (s_WindowHeight == 0) ? 1.0f : static_cast<float>(s_WindowWidth) / s_WindowHeight;
}

}