#pragma once
#include <functional>
#include <vector>
#include <sstream>
#include "renderer/textRenderer.h"
#include "ui/systemMonitor.h"

using string = std::string;

class UIManager {
public:
    static void Init();
    static void RenderUI();

    // 注册调试信息回调
    static void AddDebugInfo(std::function<string()> callback);
    
private:
    static SystemMonitor m_SystemMonitor;
    inline static std::vector<std::function<std::string()>> s_DebugCallbacks = {};
};
