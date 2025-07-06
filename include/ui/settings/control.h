// include/ui/settings/control.h
#pragma once

namespace CubeDemo::UI {
class SettingPanel;

class CtrlSettings {
    friend class SettingPanel;
    static void Render();
};
} // namespace CubeDemo::UI
