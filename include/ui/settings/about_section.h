// include/ui/settings/about_section.h
#pragma once

namespace CubeDemo::UI {
class SettingPanel;

class AboutSection {
    friend class SettingPanel;
    static void Render();
};
} // namespace CubeDemo::UI
