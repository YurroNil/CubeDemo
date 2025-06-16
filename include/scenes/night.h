// include/scenes/night.h
#pragma once
#include "scenes/base.h"
#include "graphics/mesh.h"
#include "prefabs/lights/volum_beam.h"

// 别名
using SL = CubeDemo::Prefabs::SpotLight;
using VolumBeam = CubeDemo::Prefabs::VolumBeam;

namespace CubeDemo::Scenes {
class NightScene : public SceneBase {
public:
    NightScene();
    ~NightScene();
    
    void Init() override;
    void Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) override;
    void Cleanup() override;
    // Getters
    SL* GetSpotLight() { return m_SpotLight; }
    DL* GetMoonLight() { return m_MoonLight; }

    // 光束实例
    VolumBeam Beam;

private:
    SL* m_SpotLight = nullptr;
    DL* m_MoonLight = nullptr;

};
}   // CubeDemo::Scenes
