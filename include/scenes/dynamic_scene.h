// include/scenes/dynamic_scene.h
#pragma once
#include "scenes/base.h"

namespace CubeDemo::Scenes {

class DynamicScene : public SceneBase {
public:
    // 构造函数，接受一个SceneInfo参数
    DynamicScene(const SceneInfo& info);
    
     // 实现纯虚函数
    // 初始化场景
    void Init() override {}
    // 渲染场景
    void Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) override {}
    // 清理场景
    void Cleanup() override {}
    
    // 获取场景信息
    const SceneInfo& GetSceneInfo() const { return m_info; }
    // 获取方向光
    const std::vector<Prefabs::DirLight*>& GetDirLights() const { return m_dirLights; }
    // 获取点光源
    const std::vector<Prefabs::PointLight*>& GetPointLights() const { return m_pointLights; }
    // 获取聚光灯
    const std::vector<Prefabs::SpotLight*>& GetSpotLights() const { return m_spotLights; }
    // 获取天空光
    const std::vector<Prefabs::SkyLight*>& GetSkyLights() const { return m_skyLights; }
    // 获取体积光
    const std::vector<Prefabs::VolumBeam*>& GetVolumBeams() const { return m_volumBeams; }
    // 获取阴影贴图
    const std::vector<Prefabs::ShadowMap*>& GetShadowMaps() const { return m_shadowMaps; }
    
    // 添加模型
    void AddModel(::CubeDemo::Model* model) { m_models.push_back(model); }
    // 添加方向光
    void AddDirLight(Prefabs::DirLight* light) { m_dirLights.push_back(light); }
    // 添加点光源
    void AddPointLight(Prefabs::PointLight* light) { m_pointLights.push_back(light); }
    // 添加聚光灯
    void AddSpotLight(Prefabs::SpotLight* light) { m_spotLights.push_back(light); }
    // 添加天空光
    void AddSkyLight(Prefabs::SkyLight* light) { m_skyLights.push_back(light); }
    // 添加体积光
    void AddVolumBeam(Prefabs::VolumBeam* beam) { m_volumBeams.push_back(beam); }
    // 添加阴影贴图
    void AddShadowMap(Prefabs::ShadowMap* shadowMap) { m_shadowMaps.push_back(shadowMap); }
    
private:
    // 场景信息
    SceneInfo m_info;
    std::vector<::CubeDemo::Model*> m_models;
    std::vector<Prefabs::DirLight*> m_dirLights;
    std::vector<Prefabs::PointLight*> m_pointLights;
    std::vector<Prefabs::SpotLight*> m_spotLights;
    std::vector<Prefabs::SkyLight*> m_skyLights;
    std::vector<Prefabs::VolumBeam*> m_volumBeams;
    std::vector<Prefabs::ShadowMap*> m_shadowMaps;
};

} // namespace CubeDemo::Scenes
