// include/ui/settings/game.h
#pragma once

namespace CubeDemo::UI {
class SettingPanel;

class GameSettings {
    friend class SettingPanel;
    static void Render();
};
} // namespace CubeDemo::UI
