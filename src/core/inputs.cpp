// src/core/inputs.cpp

#include "core/inputs.h"
#include "kits/imgui.h"

namespace CubeDemo {

namespace { Inputs* s_CurrentHandler = nullptr; }
bool Inputs::s_isDebugVisible = false;
float Inputs::lastEscPressTime = 0.0f;

void Inputs::Init(Camera* camera) { s_Camera = camera; }

// 静态回调方法
void Inputs::MouseCallback(double xpos, double ypos) {

    if (isGamePaused || ImGui::GetIO().WantCaptureMouse) { s_FirstMouse = true; return; }
    // 调试输出验证
    if (s_FirstMouse) {
        s_LastX = xpos;
        s_LastY = ypos;
        s_FirstMouse = false;
    }

    float xoffset = xpos - s_LastX; float yoffset = s_LastY - ypos; // 注意Y轴方向
    s_LastX = xpos; s_LastY = ypos;

    if (s_Camera) { s_Camera->ProcMouseMovement(xoffset, yoffset, true); }

}

void Inputs::ScrollCallback(double yoffset) {
    s_Camera->ProcMouseScroll(static_cast<float>(yoffset));
}

// 游戏暂停
void Inputs::PauseTheGame(GLFWwindow* &window) {
    if (!glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) return;

/* --------若点击了ESC键则：----------*/
    // 变化状态
    isGamePaused = !isGamePaused;
    // 切换鼠标状态
    glfwSetInputMode(window, GLFW_CURSOR, isGamePaused ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);

    // 恢复游戏时更新鼠标位置
    double current_x, current_y;
    glfwGetCursorPos(window, &current_x, &current_y);
    s_LastX = current_x;
    s_LastY = current_y;
    s_FirstMouse = true; // 强制下次移动时重置初始位置

}

// 输入处理
void Inputs::ProcKeyboard(GLFWwindow* &window, float deltaTime) {
    // WASD 移动
    float velocity = 2* s_Camera->attribute.movementSpeed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        s_Camera->Position += s_Camera->direction.front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        s_Camera->Position -= s_Camera->direction.front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        s_Camera->Position -= s_Camera->direction.right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        s_Camera->Position += s_Camera->direction.right * velocity;
    // 空格上升
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        s_Camera->Position.y += velocity;
    // Shift下降
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        s_Camera->Position.y -= velocity;

    // Alt键呼出鼠标
    bool alt_pressed = glfwGetKey(window, GLFW_KEY_LEFT_ALT) || glfwGetKey(window, GLFW_KEY_RIGHT_ALT);
    if (alt_pressed != s_AltPressed) {
        glfwSetInputMode(window, GLFW_CURSOR, alt_pressed ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        s_FirstMouse = true; // 重置鼠标初始位置
        s_AltPressed = alt_pressed;
    }

    // F3切换调试信息
    static bool f3_last_state = false;
    bool f3_current_state = glfwGetKey(window, GLFW_KEY_F3);
    if (f3_current_state && !f3_last_state) s_isDebugVisible = !s_isDebugVisible;
    f3_last_state = f3_current_state;
}

// 回到游戏(解除暂停状态)
void Inputs::ResumeTheGame(GLFWwindow* &window) {
    isGamePaused = false;
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

// 程序主循环中，ESC按键高频检测
void Inputs::isEscPressed(GLFWwindow* &window) {
    static float last_esc_press_time = 0.0f;
    const float current_time = glfwGetTime();
    
    if ((current_time - last_esc_press_time) > Inputs::escCoolDown) {
        last_esc_press_time = current_time;
        if (!Inputs::isGamePaused) Inputs::PauseTheGame(window);
        else if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) Inputs::ResumeTheGame(window);
    }
}

}