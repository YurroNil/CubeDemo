// include/ui/presetlib/presetlist_area.h
#pragma once

namespace CubeDemo::UI {

class PresetlistArea {
    #define s_v static void
// private
    inline static string m_CurrSelector = "None";
    s_v DrawPresetGrid();
    s_v DrawPresetCard(const char *name, int id, float width, float height);

public:

    inline static int s_SelectedPreset = -1;
    s_v Render();
    s_v UpdateSelector();

};
}   // namespace CubeDemo::UI
