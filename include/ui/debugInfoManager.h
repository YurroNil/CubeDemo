// include/ui/debugInfoManager.h

#pragma once

#include "ui/uiManager.h"
#include "ui/systemMonitor.h"
#include "core/inputHandler.h"



class DebugInfoManager {
public:
    static void Init();

    static void DisplayDebugInfo(Camera& camera);   // 显示调试信息面板
    static void AddDebugInfo(std::function<string()> callback); // 注册调试信息回调

private:

    static SystemMonitor m_SystemMonitor;
    inline static std::vector<std::function<string()>> s_DebugCallbacks = {};


};