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
// 获取场景实例
template <SceneMng::SceneID id>
auto& GetSceneInstance(SceneMng* mng) {
    if constexpr (id == SceneMng::SceneID::DEFAULT) return mng->Default;
    else if constexpr (id == SceneMng::SceneID::NIGHT) return mng->Night;
}

// 定义场景注册表的列表
struct SceneRegistryEntry {
    std::string_view name;
    std::reference_wrapper<Scenes::SceneBase> instance;
    
    explicit SceneRegistryEntry(
        std::string_view name_, 
        Scenes::SceneBase& scene
    ) : name(name_), instance(scene) {}
};

inline auto& GetSceneRegistry(SceneMng* mng) {
    static std::unordered_map<SceneMng::SceneID, SceneRegistryEntry> registry;
    if (registry.empty()) {
        registry.emplace(
            SceneMng::SceneID::DEFAULT,
            SceneRegistryEntry("Default", mng->Default)
        );
        registry.emplace(
            SceneMng::SceneID::NIGHT,
            SceneRegistryEntry("Night", mng->Night)
        );
    }
    return registry;
}
}// namespace SceneInternal

// 清理场景模板
template <SceneMng::SceneID id>
void SceneMng::CleanScene() {

    auto& scene = SceneInternal::GetSceneInstance<id>(this);
    if (scene.s_isInited && !scene.s_isCleanup) {
        scene.Cleanup();
        scene.s_isCleanup = true; scene.s_isInited = false;
    }
}
} // namespace CubeDemo::Managers
