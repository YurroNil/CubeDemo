// include/ui/presetlib/panel.h
#pragma once

namespace CubeDemo::UI {

class PresetlibPanel {
    #define s_v static void
    
    inline static bool m_inited = false;
    s_v DrawMenuBar();
    s_v UpdateSelector();

public:
    s_v Init();
    s_v Render(Camera* camera);
};
} // namespace CubeDemo::UI
