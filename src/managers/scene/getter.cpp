// src/managers/scene/getter.cpp
#include "pch.h"
#include "managers/sceneMng.h"
#include "resources/model.h"

// 外部变量声明
namespace CubeDemo {
    extern std::vector<Model*> MODEL_POINTERS;
}

namespace CubeDemo::Managers {

// 别名
namespace Internal = SceneInternal;

// 获取当前场景名称
string SceneGetter::Name() const {
    switch (m_owner->Current) {
        case SceneID::NIGHT:
            return m_owner->Night.GetName();

        case SceneID::DEFAULT:
        default:
            return m_owner->Default.GetName();
    }
    return "null";
}

SceneGetter::SceneGetter(const SceneMng* inst)
    : m_owner(inst) {}

string SceneGetter::ID() const {
    switch (m_owner->Current) {
        case SceneID::NIGHT:
            return m_owner->Night.GetID();

        case SceneID::DEFAULT:
        default:
            return m_owner->Default.GetID();
    }
    return "null";
}

std::vector<string> SceneGetter::ModelNames() const {
    std::vector<string> arr;
    if(MODEL_POINTERS.empty()) return arr;

    for(auto& model : MODEL_POINTERS) {
        arr.push_back(model->GetName());
    }
    return arr;
}
}
