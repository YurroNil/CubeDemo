// include/prefabs/lights/volum_beam.h
#pragma once
#include "prefabs/lights/data.h"

namespace CubeDemo::Prefabs {

using BeamEffects = CubeDemo::Prefabs::BeamEffects;

class VolumBeam {
private:
    // 着色器与光束模型的指针存储
    Shader* m_VolumShader = nullptr;
    Mesh* m_LightVolume = nullptr;

    // 光束效果配置
    BeamEffects m_effects;
    TexturePtr m_noiseTexture = nullptr;

    // 私有方法
    void SetFx(Camera* camera, SL* spot_light);
    void SetTextureArgs(const string& path);
    mat4 CalcTransform(SL* spot_light);

public:
    VolumBeam();
    void Render(Camera* camera, SL* spot_light);

    // 配置光束效果
    void Configure(const BeamEffects& effects);
    void LoadNoiseTexture(const string& path);

    // Creaters
    void CreateVolumShader();
    void CreateLightCone(float radius, float height);

    // Getters
    Shader* GetVolumShader() const;
    Mesh* GetLightVolume() const;
    const BeamEffects& GetEffects() const;

    // 注意：该光束清理器的方法在：`managers/light/cleanner.cpp`中实现
    // 而不是`prefabs/lights/volum_beam.cpp`!
    class Remover {
        VolumBeam* m_owner;
        public:
        explicit Remover(VolumBeam* owner);

        Remover& VolumShader();
        Remover& LightCone();
        Remover& NoiseTexture();
    };

    Remover Remove;
};
}