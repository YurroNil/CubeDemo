// include/managers/scene/utils.inl
#pragma once

namespace CubeDemo::Managers {
namespace SceneInternal {

// 调用场景类方法
template <typename SceneT, typename... Args>
void CallSceneMethod(SceneT& scene, void (SceneT::*method)(Args...), Args&&... args) {
    if constexpr (std::is_member_function_pointer_v<decltype(method)>) {
        (scene.*method)(std::forward<Args>(args)...);
    }
}

// 定义场景注册表的列表
struct SceneRegistryEntry {
    std::string_view name;
    Scenes::SceneBase& instance;
    
    explicit SceneRegistryEntry(
        std::string_view name_, 
        Scenes::SceneBase& scene
    ) : name(name_), instance(scene) {}
};

inline auto& GetSceneRegistry(SceneMng* mng) {
    static std::unordered_map<SceneMng::SceneID, SceneRegistryEntry> registry;
    if (registry.empty()) {
        registry.try_emplace(
            SceneID::DEFAULT,
            mng->Default.GetName(),
            mng->Default
        );
        registry.try_emplace(
            SceneID::NIGHT,
            mng->Night.GetName(),
            mng->Night
        );
    }
    return registry;
}
}// namespace SceneInternal
} // namespace CubeDemo::Managers
