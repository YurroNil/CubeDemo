#include "core/windowManager.h"

void windowManager::ToggleFullscreen(GLFWwindow* window) {
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
