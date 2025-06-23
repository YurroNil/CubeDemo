// include/ui/panels/presetlib.h
#pragma once

namespace CubeDemo::UI {

class PresetlibPanel {

    
    inline static bool s_initialized = false;

    // 内部绘制方法
    static void DrawMenuBar();
    
    // 功能方法
    static void UpdateSelector();

public:
    static void Init();
    static void Render(Camera* camera);
};

} // namespace CubeDemo::UI
