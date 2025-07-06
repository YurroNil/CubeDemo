// include/ui/settings/panel.h
#pragma once

namespace CubeDemo::UI {
class PausePanel;

struct PausePanelBridge {
    ImVec2 MenuSize;
    int& CurrentTab;
    ImDrawList* draw_list;

};

class SettingPanel {
    friend class PausePanel;
    static void Render(PausePanelBridge& bridge);
};
} // namespace CubeDemo::UI
