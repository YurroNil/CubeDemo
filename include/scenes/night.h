// include/scenes/night.h
#pragma once
#include "scenes/base.h"
#include "prefabs/lights/base.h"

using SL = CubeDemo::Prefabs::SpotLight;

namespace CubeDemo::Scenes {
class NightScene : public SceneBase {
public:
    NightScene() = default;
    ~NightScene();
    
    void Init(SceneMng& scene_inst, Light& light) override;
    void Render(GLFWwindow* window, Camera* camera, const Light& light, ShadowMap* shadow_map) override;
    void Cleanup(Light& light) override;
    
    void UpdateBeam(float deltaTime);
    
private:
    SL* m_SpotLight = nullptr;
    DL* m_MoonLight = nullptr;
    Shader* m_VolumetricShader = nullptr;
    Mesh* m_LightVolume = nullptr;
};
}
