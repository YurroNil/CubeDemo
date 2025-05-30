// include/scenes/night.h
#pragma once
#include "scenes/base.h"

using SL = CubeDemo::Prefabs::SpotLight;

namespace CubeDemo::Scenes {
class NightScene : public SceneBase {
public:
    NightScene();
    ~NightScene();
    
    void Init(SceneMng* scene_inst, Light& light) override;
    void Render(GLFWwindow* window, Camera* camera, const Light& light, ShadowMap* shadow_map) override;
    void Cleanup(Light& light) override;
    void SetLightsData(const string& config_path, SL* spot_light, DL* moon_light);

    // Getters
    SL* GetSpotLight() { return m_SpotLight; }
    DL* GetMoonLight() { return m_MoonLight; }

private:
    SL* m_SpotLight = nullptr;
    DL* m_MoonLight = nullptr;
    Shader* m_VolumetricShader = nullptr;
    Mesh* m_LightVolume = nullptr;

    void RenderVolumetricBeam(Camera* camera);
    mat4 CalcBeamTransform();
    Mesh* CreateLightCone(float radius, float height);

};
}   // CubeDemo::Scenes
