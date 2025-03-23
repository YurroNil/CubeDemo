#pragma once
#include <functional>
#include <vector>
#include <sstream>
#include "renderer/text.h"

using string = std::string;

class UIManager {
public:
    static void Init();
    static void RenderUI();

    // 注册调试信息回调
    static void AddDebugInfo(std::function<string()> callback);

private:
    inline static std::vector<std::function<string()>> s_DebugCallbacks;
};
