// src/core/inputHandler.cpp

#include "core/inputHandler.h"
#include "core/timeManager.h"
#include <string>

namespace {
    InputHandler* s_CurrentHandler = nullptr;
}

void InputHandler::Init(Camera* camera) {
    s_Camera = camera;

}
// 静态回调方法
void InputHandler::MouseCallback(double xpos, double ypos) {
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

void InputHandler::ScrollCallback(double yoffset) {
    s_Camera->ProcessMouseScroll(static_cast<float>(yoffset));
}


void InputHandler::ProcessKeyboard(GLFWwindow* &window, float deltaTime) {
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

    // F3切换调试信息
    static bool f3LastState = false;
    bool f3CurrentState = glfwGetKey(window, GLFW_KEY_F3);
    if (f3CurrentState && !f3LastState) {
        showDebugInfo = !showDebugInfo;
    }
    f3LastState = f3CurrentState;
}

