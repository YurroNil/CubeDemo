// include/ui/main_menu/scene_selection.h
#pragma once
#include "ui/main_menu/base.h"

namespace CubeDemo::UI::MainMenu {

class SceneSelection : public MainMenuBase {
// private

    static void SceneCard(const SceneInfo& scene, int id, float width, float height);

public:
    static void Render();
};
} // namespace CubeDemo::UI
