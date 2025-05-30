// include/scenes/sceneMng.inl

namespace CubeDemo::Scenes {
// 清理场景模板
template <SceneMng::SceneID id>
void SceneMng::CleanScene(Light& light) {

    auto& scene = Internal::GetSceneInstance<id>(this);
    if (scene.s_isInited && !scene.s_isCleanup) {
        scene.Cleanup(light);
        scene.s_isCleanup = true; scene.s_isInited = false;
    }
}

}   // namespace CubeDemo::Scenes
