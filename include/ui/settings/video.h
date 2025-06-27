// include/ui/settings/video.h
#pragma once

namespace CubeDemo::UI {
class ContentArea;

class VideoSettings {
    friend class ContentArea;
    static void Render();
};
} // namespace CubeDemo::UI
