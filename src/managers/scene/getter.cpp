// src/scenes/night.cpp
#include "pch.h"
#include "managers/sceneMng.h"
#include "resources/model.h"

// 外部变量声明
namespace CubeDemo {
    extern Shader* MODEL_SHADER;
    extern std::vector<Model*> MODEL_POINTERS;
    extern Managers::SceneMng* SCENE_MNG;
}

namespace CubeDemo::Managers {

SceneGetter::SceneGetter(const SceneMng* inst)
    : m_owner(inst) {}

string SceneGetter::Name() const {
    switch (SCENE_MNG->Current) {
        case SceneID::NIGHT:
            return "夜晚场景";

        case SceneID::DEFAULT:
        default:
            return "默认场景";
    }
}

string SceneGetter::ID() const {
    switch (SCENE_MNG->Current) {
        case SceneID::NIGHT:
            return "night";

        case SceneID::DEFAULT:
        default:
            return "default";
    }
    return "null";
}

std::vector<string> SceneGetter::ModelNames() const {
    std::vector<string> arr;

    for(auto& model : MODEL_POINTERS) {
        arr.push_back(model->GetName());
    }
    return arr;
}
}
