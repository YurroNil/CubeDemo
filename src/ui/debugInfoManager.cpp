// src/ui/debugInfoManager.cpp

#include "ui/debugInfoManager.h"
#include "unit/stringConvertor.h"

// 初始化静态成员
SystemMonitor DebugInfoManager::m_SystemMonitor;  // 初始化系统监控实例


// 初始化
void DebugInfoManager::Init() {}


// 调试面板显示(渲染循环内)
void DebugInfoManager::DisplayDebugInfo(Camera& camera) {
    if (!InputHandler::showDebugInfo) return;

    // 获取窗口尺寸
    GLFWwindow* window = glfwGetCurrentContext();
    int winWidth, winHeight;
    glfwGetWindowSize(window, &winWidth, &winHeight);
    
    // 动态计算坐标
    float x = 20.0f; float y = winHeight - 100.0f;

    // 收集所有调试信息
    std::stringstream ss;

    // 添加系统监控信息
    ss << "设备：" << StringConvertor::WstringTo_U8(m_SystemMonitor.GetOSVersion());
    ss << "  内存用量: " << m_SystemMonitor.GetMemoryUsage() << "%  ";
    ss << " ";

    for (auto& callback : s_DebugCallbacks) {
        ss << callback() << "";
    }
    wstring wtext = StringConvertor::U8_to_Wstring(ss.str());

    // 渲染字体
    TextRenderer::RenderText(wtext, x, y, 0.7f, vec3(1.0f));
}


// 调试信息
void DebugInfoManager::AddDebugInfo(std::function<string()> callback) {
    s_DebugCallbacks.push_back(callback);
}
