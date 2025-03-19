#include "core/inputHandler.h"

namespace {
    InputHandler* s_CurrentHandler = nullptr;
}

void InputHandler::Initialize(Camera* camera, GLFWwindow* window) {
    s_Camera = camera;
    s_Window = window;

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    s_LastX = width / 2.0f;
    s_LastY = height / 2.0f;

    // 设置 GLFW 回调（通过 lambda 转发到静态方法）
    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
        MouseCallback(w, x, y);
    });
    glfwSetScrollCallback(window, [](GLFWwindow* w, double x, double y) {
        ScrollCallback(w, x, y);
    });
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

// 静态回调方法
void InputHandler::MouseCallback(GLFWwindow* window, double& xpos, double& ypos) {
    if (s_FirstMouse) {
        s_LastX = xpos;
        s_LastY = ypos;
        s_FirstMouse = false;
    }

    float xoffset = xpos - s_LastX;
    float yoffset = s_LastY - ypos;
    s_LastX = xpos;
    s_LastY = ypos;

    s_Camera->ProcessMouseMovement(xoffset, yoffset);
}

void InputHandler::ScrollCallback(GLFWwindow* window, double& xoffset, double& yoffset) {
    float temp = static_cast<float>(yoffset);
    s_Camera->ProcessMouseScroll(temp);
}


void InputHandler::ProcessKeyboard(GLFWwindow* window, float& deltaTime) {
    // ESC 退出程序
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // WASD 移动
    float velocity = s_Camera->MovementSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        s_Camera->Position += s_Camera->Front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        s_Camera->Position -= s_Camera->Front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        s_Camera->Position -= s_Camera->Right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        s_Camera->Position += s_Camera->Right * velocity;

    // 空格上升
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        s_Camera->Position.y += velocity;
    // Shift下降
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        s_Camera->Position.y -= velocity;

    // Alt键呼出鼠标(参考原神)
    bool altPressed = glfwGetKey(window, GLFW_KEY_LEFT_ALT) || glfwGetKey(window, GLFW_KEY_RIGHT_ALT);
    if (altPressed != s_AltPressed) {
        glfwSetInputMode(window, GLFW_CURSOR, altPressed ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        s_FirstMouse = true; // 重置鼠标初始位置
        s_AltPressed = altPressed;
    }

    // F11 全屏切换（防抖处理）
    static bool f11LastState = false;
    bool f11CurrentState = glfwGetKey(window, GLFW_KEY_F11);
    if (f11CurrentState && !f11LastState) {
        windowManager::ToggleFullscreen(window);
    }
    f11LastState = f11CurrentState;
}

