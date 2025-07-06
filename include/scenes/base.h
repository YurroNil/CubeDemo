// include/scenes/base.h
#pragma once
#include "prefabs/lights/fwd.h"
#include "prefabs/lights/volum_beam.h"

namespace CubeDemo::Scenes {

struct SceneInfo {
    string id, name, description,
    author, icon,
    resourcePath; // 添加资源路径
    
    struct PrefabRef {
        string type, path;
    };
    std::vector<PrefabRef> prefabs;
    
    static SceneInfo FromJson(const json& j);
};

class SceneBase {
public:
    virtual ~SceneBase() = default;
    
    virtual void Init() = 0;
    virtual void Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) = 0;
    virtual void Cleanup() = 0;
    
    const string& GetID() const { return m_id; }
    const string& GetName() const { return m_name; }
    
    // 成员变量
    string m_id, m_name;
};

} // namespace CubeDemo::Scenes
