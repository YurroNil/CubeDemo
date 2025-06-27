// include/ui/settings/content_area.h
#pragma once

namespace CubeDemo::UI {
class PausePanel;

struct PausePanelBridge {
    ImVec2 MenuSize;
    int& CurrentTab;
    ImDrawList* draw_list;

};

class ContentArea {
    friend class PausePanel;
    static void Render(PausePanelBridge& bridge);
};
} // namespace CubeDemo::UI
