// src/core/inputs.cpp
#include "pch.h"
#include "core/inputs.h"

namespace CubeDemo {

// 面板切换按键映射表初始化
std::unordered_map<int, std::function<void()>> INPUTS::s_PanelkeyMap = {
    {GLFW_KEY_T, []() { s_isEditMode = !s_isEditMode; }},
    {GLFW_KEY_C, []() { s_isPresetVsble = !s_isPresetVsble; }},
    {GLFW_KEY_F3, []() { s_isDebugVsble = !s_isDebugVsble; }}
};

// 相机操控按键映射表初始化
std::unordered_map<int, std::function<void(Camera* camera, float velocity)>> INPUTS::s_CamerakeyMap = {
    {GLFW_KEY_W, [](Camera* camera, float velocity) { camera->Position += camera->direction.front * velocity; }},
    {GLFW_KEY_A, [](Camera* camera, float velocity) { camera->Position -= camera->direction.right * velocity; }},
    {GLFW_KEY_S, [](Camera* camera, float velocity) { camera->Position -= camera->direction.front * velocity; }},
    {GLFW_KEY_D, [](Camera* camera, float velocity) { camera->Position += camera->direction.right * velocity; }},
    {GLFW_KEY_SPACE, [](Camera* camera, float velocity) { camera->Position.y += velocity; }},
    {GLFW_KEY_LEFT_SHIFT, [](Camera* camera, float velocity) { camera->Position.y -= velocity; }}
};

// 按键状态跟踪
std::unordered_map<int, bool> INPUTS::s_KeyState;

bool INPUTS::isOpeningPanel() {
    return s_isEditMode || s_isGamePaused || s_isPresetVsble;
}

void INPUTS::UpdateCursorMode(GLFWwindow* window) {

    bool showCursor = isOpeningPanel() || m_AltPressed;

    glfwSetInputMode(window, GLFW_CURSOR, showCursor ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
    // 下次隐藏光标时需要重置首次移动标志
    if (showCursor) m_Mouse.firstMove = true;
}

void INPUTS::SetPaused(GLFWwindow* window, bool paused) {
    s_isGamePaused = paused;
    UpdateCursorMode(window);
}

void INPUTS::ProcPanelKeys(GLFWwindow* window) {
    // 处理面板切换按键
    for (auto& pair : s_PanelkeyMap) {
        int key = pair.first;
        int state = glfwGetKey(window, key);
        
        // 边缘检测：按键刚按下时触发
        if (state != GLFW_PRESS) {
            s_KeyState[key] = false;
            continue;
        }

        // 若之前已按下则
        if (s_KeyState[key]) continue;

        pair.second(); // 执行按键操作
        if (key != GLFW_KEY_F3) UpdateCursorMode(window);
        s_KeyState[key] = true;
    }

    // 单独处理ESC键
    static bool escPressed = false;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS) {
        escPressed = false;
        return;
    }

    if (!escPressed) {
        SetPaused(window, !s_isGamePaused);
        escPressed = true;
    }
}

void INPUTS::ProcCameraKeys(GLFWwindow* window, Camera* camera, float deltaTime) {
    if (isOpeningPanel()) return; // 面板打开时不处理相机控制
    
    float velocity = 2 * camera->attribute.movementSpeed * deltaTime;
    
    // 处理所有按下的相机控制键
    for (auto& pair : s_CamerakeyMap) {
        if (glfwGetKey(window, pair.first) == GLFW_PRESS) {
            pair.second(camera, velocity);
        }
    }

    // Alt键切换鼠标显示
    bool altNow = (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS);
    
    if (altNow != m_AltPressed) {
        m_AltPressed = altNow;
        UpdateCursorMode(window);
    }
}

// 回调函数
void INPUTS::MouseCallback(double xpos, double ypos) {
    // 获取当前相机
    Camera* camera = Camera::GetCamera();
    // 如果正在打开面板或者Alt键被按下，则返回
    if (isOpeningPanel() || m_AltPressed) return;
    
    // 如果是第一次移动鼠标，则记录当前位置
    if (m_Mouse.firstMove) {
        m_Mouse.lastX = xpos;
        m_Mouse.lastY = ypos;
        m_Mouse.firstMove = false;
    }

    // 计算鼠标移动的偏移量
    float xoffset = static_cast<float>(xpos - m_Mouse.lastX);
    float yoffset = static_cast<float>(m_Mouse.lastY - ypos); // Y轴反向
    // 更新鼠标当前位置
    m_Mouse.lastX = xpos;
    m_Mouse.lastY = ypos;

    // 如果相机存在，则处理鼠标移动
    if (camera != nullptr) {
        camera->ProcMouseMovement(xoffset, yoffset, true);
    }
}

void INPUTS::ScrollCallback(double yoffset) {
    Camera* camera = Camera::GetCamera();
    if (camera == nullptr || isOpeningPanel()) return;

    camera->ProcMouseScroll(static_cast<float>(yoffset));
}
} // namespace CubeDemo
