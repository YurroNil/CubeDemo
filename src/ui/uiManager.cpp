// src/ui/uiManager.cpp

#include "ui/uiManager.h"
#include "core/inputHandler.h"

void UIManager::Init() {

}

void UIManager::RenderUI() {
    if (!InputHandler::showDebugInfo) return;

    // 获取窗口尺寸
    GLFWwindow* window = glfwGetCurrentContext();
    int winWidth, winHeight;
    glfwGetWindowSize(window, &winWidth, &winHeight);
    
    // 动态计算坐标
    float x = 20.0f;
    float y = winHeight - 30.0f;

    // 收集所有调试信息
    std::stringstream ss;
    for (auto& callback : s_DebugCallbacks) {
        ss << callback() << "\n";
    }
    
    // 渲染
    TextRenderer::RenderText(ss.str(), x, y, 0.5f, glm::vec3(1.0f));
}

void UIManager::AddDebugInfo(std::function<std::string()> callback) {
    s_DebugCallbacks.push_back(callback);
}