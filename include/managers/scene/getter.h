// include/managers/scene/getter.h
#pragma once
#include "scenes/default.h"
#include "scenes/night.h"

namespace CubeDemo::Managers {

class SceneGetter {
    const SceneMng* m_owner = nullptr;
public:
    SceneGetter(const SceneMng* inst);
    string Name() const;
    string ID() const;
    std::vector<string> ModelNames() const;
};
}
