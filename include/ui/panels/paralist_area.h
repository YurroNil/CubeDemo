// include/ui/panels/paralist_area.h
#pragma once

namespace CubeDemo::UI {

class ParalistArea {
// private
    static void PanelHeader();
    static void TransformSection();
    static void ColorSection();
    static void ShaderSection();
    static void MaterialSection();
    static void ActionButtons();
    static void AdvancedSettings();
public:
    static void Render();

};
}   // namespace CubeDemo::UI
