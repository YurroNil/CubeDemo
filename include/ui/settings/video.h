// include/ui/settings/video.h
#pragma once

namespace CubeDemo::UI {
class SettingPanel;

class VideoSettings {
    friend class SettingPanel;
    static void Render();
};
} // namespace CubeDemo::UI
