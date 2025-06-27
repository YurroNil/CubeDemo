// include/ui/settings/game.h
#pragma once

namespace CubeDemo::UI {
class ContentArea;

class GameSettings {
    friend class ContentArea;
    static void Render();
};
} // namespace CubeDemo::UI
