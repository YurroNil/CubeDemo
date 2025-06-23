// include/ui/panels/presetlist_area.h
#pragma once

namespace CubeDemo::UI {

class PresetlistArea {
// private
    inline static string m_CurrSelector = "None";
    static void DrawPresetGrid();

public:
    inline static int s_SelectedPreset = -1;

    static void Render();
    static void UpdateSelector();

};
}   // namespace CubeDemo::UI
