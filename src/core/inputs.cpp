// src/core/inputs.cpp
#include "pch.h"
#include "core/inputs.h"

namespace CubeDemo {

void Inputs::Init(Camera* camera) { m_Camera = camera; }

// 静态回调方法
void Inputs::MouseCallback(double xpos, double ypos) {

    if (s_isEditMode || isGamePaused || ImGui::GetIO().WantCaptureMouse) return;

    // 调试输出验证
    if (m_FirstMouse) {
        m_LastX = xpos;
        m_LastY = ypos;
        m_FirstMouse = false;
    }

    float xoffset = xpos - m_LastX; float yoffset = m_LastY - ypos; // 注意Y轴方向
    m_LastX = xpos; m_LastY = ypos;

    if (m_Camera) m_Camera->ProcMouseMovement(xoffset, yoffset, true);

}

bool Inputs::isCameraEnabled() {
    // 编辑模式、游戏暂停状态下，禁用鼠标控制摄像机移动

    if (s_isEditMode || isGamePaused || ImGui::GetIO().WantCaptureMouse || m_AltPressed || s_isPresetVisible) {
        m_FirstMouse = true;
        return true;
    } else return false;
}

void Inputs::ScrollCallback(double yoffset) {

    if (isCameraEnabled) return;

    m_Camera->ProcMouseScroll(static_cast<float>(yoffset));
}

// 游戏暂停
void Inputs::PauseTheGame(GLFWwindow* window) {
    if (!glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) return;

/* --------若点击了ESC键则：----------*/
    // 变化状态
    isGamePaused = !isGamePaused;
    // 切换鼠标状态
    glfwSetInputMode(window, GLFW_CURSOR, isGamePaused ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);

    // 恢复游戏时更新鼠标位置
    double current_x, current_y;
    glfwGetCursorPos(window, &current_x, &current_y);
    m_LastX = current_x;
    m_LastY = current_y;
    m_FirstMouse = true; // 强制下次移动时重置初始位置

}

// 输入处理
void Inputs::ProcKeyboard(GLFWwindow* window, float delta_time) {
    
    // T键切换编辑模式
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) ToggleEditMode(window);
    // C键切换预设库
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) TogglePresetlib(window);

    // 编辑模式下禁用相机控制
    if (s_isEditMode || isGamePaused || s_isPresetVisible) return;
    
    // WASD 移动
    float velocity = 2* m_Camera->attribute.movementSpeed * delta_time;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        m_Camera->Position += m_Camera->direction.front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        m_Camera->Position -= m_Camera->direction.front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        m_Camera->Position -= m_Camera->direction.right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        m_Camera->Position += m_Camera->direction.right * velocity;
    // 空格上升
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        m_Camera->Position.y += velocity;
    // Shift下降
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        m_Camera->Position.y -= velocity;

    // Alt键呼出鼠标
    bool alt_pressed = glfwGetKey(window, GLFW_KEY_LEFT_ALT) || glfwGetKey(window, GLFW_KEY_RIGHT_ALT);
    if (alt_pressed != m_AltPressed) {
        glfwSetInputMode(window, GLFW_CURSOR, alt_pressed ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        m_FirstMouse = true; // 重置鼠标初始位置
        m_AltPressed = alt_pressed;
    }

    // F3切换调试信息
    static bool f3_last_state = false;
    bool f3_current_state = glfwGetKey(window, GLFW_KEY_F3);
    if (f3_current_state && !f3_last_state) s_isDebugVisible = !s_isDebugVisible;
    f3_last_state = f3_current_state;
}

// 回到游戏(解除暂停状态)
void Inputs::ResumeTheGame(GLFWwindow* window) {
    isGamePaused = false;
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

// 程序主循环中，ESC按键高频检测
void Inputs::isEscPressed(GLFWwindow* window) {
    static float last_esc_press_time = 0.0f;
    const float current_time = glfwGetTime();
    
    if ((current_time - last_esc_press_time) < Inputs::s_EscCoolDown) return;

    last_esc_press_time = current_time;
    if (!Inputs::isGamePaused) Inputs::PauseTheGame(window);
    else if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) Inputs::ResumeTheGame(window);
}

// T键切换编辑面板
void Inputs::ToggleEditMode(GLFWwindow* window) {
    float currentTime = glfwGetTime();
    if (currentTime - m_LastToggleTime < m_ToggleCD) return;
    
    m_LastToggleTime = currentTime;
    s_isEditMode = !s_isEditMode;
    
    // 编辑模式下显示鼠标，禁用摄像机
    if (s_isEditMode) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        m_FirstMouse = true; // 重置鼠标初始位置
    } else if (!isGamePaused) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}
// C键切换编辑面板
void Inputs::TogglePresetlib(GLFWwindow* window) {
    float currentTime = glfwGetTime();
    if (currentTime - m_LastToggleTime < m_ToggleCD) return;
    
    m_LastToggleTime = currentTime;
    s_isPresetVisible = !s_isPresetVisible;
    
    // 预设库模式下显示鼠标，禁用摄像机
    if (s_isPresetVisible) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        m_FirstMouse = true; // 重置鼠标初始位置
    } else if (!isGamePaused) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}
}
