// include/ui/settings/audio.h
#pragma once

namespace CubeDemo::UI {
class SettingPanel;
class AudioSettings {
    friend class SettingPanel;
    static void Render();
};
} // namespace CubeDemo::UI
