// src/ui/uiManager.cpp

#include "ui/uiManager.h"
#include "core/inputHandler.h"
#include "ui/systemMonitor.h"
#include <codecvt> // 用于 std::wstring_convert
#include <locale>

SystemMonitor UIManager::m_SystemMonitor;  // 初始化系统监控实例

void UIManager::Init() {
    
}

// 添加UTF-8到wstring的转换函数（Windows需要定义_WIN32_WINNT）
std::wstring UTF8ToWideString(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
}
 
// 添加wstring到UTF-8的转换函数
std::string WideStringToUTF8(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}


void UIManager::RenderUI() {
    if (!InputHandler::showDebugInfo) return;

    // 获取窗口尺寸
    GLFWwindow* window = glfwGetCurrentContext();
    int winWidth, winHeight;
    glfwGetWindowSize(window, &winWidth, &winHeight);
    
    // 动态计算坐标
    float x = 20.0f;
    float y = winHeight - 100.0f;

    // 收集所有调试信息
    std::stringstream ss;

    // 添加系统监控信息
    ss << "系统信息：" << WideStringToUTF8(m_SystemMonitor.GetOSVersion());
    ss << "内存用量: " << m_SystemMonitor.GetMemoryUsage() << "%";
    ss << " ";

    for (auto& callback : s_DebugCallbacks) {
        ss << callback() << "";
    }
    std::wstring wtext = UTF8ToWideString(ss.str());

    // 渲染
    TextRenderer::RenderText(wtext, x, y, 0.7f, glm::vec3(1.0f));
}

void UIManager::AddDebugInfo(std::function<std::string()> callback) {
    s_DebugCallbacks.push_back(callback);
}