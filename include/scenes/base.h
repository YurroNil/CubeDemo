// include/scenes/base.h
#pragma once
#include "prefabs/lights_fwd.h"
#include "prefabs/volum_beam.h"

namespace CubeDemo::Scenes {

struct SceneInfo {
    string id, name, description,
    author, icon, previewImage,
    resourcePath;
    
    struct PrefabRef {
        string type, path;
    };
    std::vector<PrefabRef> prefabs;
    
    static SceneInfo FromJson(const json& j);
};

class SceneBase {
public:
    explicit SceneBase(const SceneInfo& info) : m_info(info) {}
    virtual ~SceneBase() = default;
    
    virtual void Init() = 0;
    virtual void Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) = 0;
    virtual void Cleanup() = 0;
    
    const SceneInfo& GetSceneInfo() const { return m_info; }
    const string& GetID() const { return m_info.id; }
    const string& GetName() const { return m_info.name; }
    
    // 成员变量
    SceneInfo m_info;
};

} // namespace CubeDemo::Scenes
