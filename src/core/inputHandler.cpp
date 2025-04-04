// src/core/inputHandler.cpp

#include "core/inputHandler.h"
#include "core/timeManager.h"
#include <iostream>

namespace { InputHandler* s_CurrentHandler = nullptr; }
void InputHandler::Init(Camera* camera) {
    s_Camera = camera;
}
float InputHandler::lastEscPressTime = 0.0f;

// 静态回调方法
void InputHandler::MouseCallback(double xpos, double ypos) {

     if (isGamePaused || ImGui::GetIO().WantCaptureMouse) {
        s_FirstMouse = true;
        return;
    }

    if (s_FirstMouse) {
        // 调试输出验证
        s_LastX = xpos;
        s_LastY = ypos;
        s_FirstMouse = false;
    }

    float xoffset = xpos - s_LastX;
    float yoffset = s_LastY - ypos; // 注意Y轴方向
    s_LastX = xpos;
    s_LastY = ypos;

    if (s_Camera) { s_Camera->ProcessMouseMovement(xoffset, yoffset); }

}

void InputHandler::ScrollCallback(double yoffset) {

    s_Camera->ProcessMouseScroll(static_cast<float>(yoffset));
}

// 游戏暂停
void InputHandler::PauseTheGame(GLFWwindow* &window) {
    if (!glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) return;

/* --------若点击了ESC键则：----------*/
    // 变化状态
    isGamePaused = !isGamePaused;
    // 切换鼠标状态
    glfwSetInputMode(window, GLFW_CURSOR, isGamePaused ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);

    // 恢复游戏时更新鼠标位置
    double currentX, currentY;
    glfwGetCursorPos(window, &currentX, &currentY);
    s_LastX = currentX;
    s_LastY = currentY;
    s_FirstMouse = true; // 强制下次移动时重置初始位置

}

// 输入处理
void InputHandler::ProcessKeyboard(GLFWwindow* &window, float deltaTime) {
    
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

    // Alt键呼出鼠标
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


// 回到游戏(解除暂停状态)
void InputHandler::ResumeTheGame(GLFWwindow* &window) {
    isGamePaused = false;
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}
