// include/prefabs/lights/volum_beam.h
#pragma once
#include "prefabs/lights/data.h"

namespace CubeDemo::Prefabs {

using BeamEffects = CubeDemo::Prefabs::BeamEffects;

class VolumBeam {
private:
    // 私有方法
    void SetFx(Camera* camera, SL* spot_light);
    void SetTextureArgs(const string& path);
    mat4 CalcTransform(SL* spot_light);

public:
    // 着色器与光束模型的指针存储
    Shader* VolumShader = nullptr;
    Mesh* LightVolume = nullptr;

    // 光束效果配置
    BeamEffects Effects;
    TexturePtr NoiseTexture = nullptr;

    VolumBeam();
    void Render(Camera* camera, SL* spot_light);

    // 配置光束效果
    void Configure(const BeamEffects& effects);
    void LoadNoiseTexture(const string& path);

    // Creaters
    void CreateVolumShader();
    void CreateLightCone(float radius, float height);
};
}