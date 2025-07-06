// include/ui/presetlib/paralist_area.h
#pragma once

namespace CubeDemo::UI {

class ParalistArea {
    #define s_v static void
// private
    s_v PanelHeader();
    s_v TransformSection();
    s_v ColorSection();
    s_v ShaderSection();
    s_v MaterialSection();
    s_v ActionButtons();
    s_v AdvancedSettings();
public:
    s_v Render();
};
}   // namespace CubeDemo::UI
