// resources/shaders/fragment/core/volumetric.glsl
#version 450 core
in vec3 WorldPos;
in vec3 Normal;
in vec3 EmitColor;
in float DistToAxis;

out vec4 FragColor;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform float lightIntensity;
uniform float lightCutOff;
uniform float lightOuterCutOff;
uniform float scatterPower;
uniform vec2 attenuationFactors; // x: 距离衰减, y: 径向衰减
uniform float alphaMultiplier;
uniform float density;
uniform float scatterAnisotropy;

void main() {
    // 计算基本光照参数
    vec3 toLight = lightPos - WorldPos;
    float distToLight = length(toLight);
    vec3 lightVec = normalize(toLight);
    
    // 聚光灯锥体检测
    float cosTheta = dot(lightVec, normalize(-lightDir));
    float cosInner = lightCutOff;
    float cosOuter = lightOuterCutOff;
    float falloff = clamp((cosTheta - cosOuter) / (cosInner - cosOuter), 0.7, 1.0);
    
    // 距离衰减 (指数衰减)
    float distAtten = 1 - exp(-attenuationFactors.x * distToLight);
    
    // 径向衰减 (高斯衰减)
    float radialAtten = exp(-attenuationFactors.y * DistToAxis * DistToAxis);
    
    // 视角相关散射 (Henyey-Greenstein相位函数)
    vec3 viewDir = normalize(viewPos - WorldPos);
    vec3 scatterDir = normalize(lightVec + viewDir);
    float cosScatter = dot(normalize(lightVec), scatterDir);
    float phase = (1.0 - scatterAnisotropy*scatterAnisotropy) / pow(1.0 + scatterAnisotropy*scatterAnisotropy - 2*scatterAnisotropy*cosScatter, 1.5);
    
    // 光束密度效果
    float densityFactor = 1.0 - exp(-density * distToLight);
    
    // 组合所有效果
    float total = lightIntensity * falloff * distAtten * radialAtten * phase * pow(densityFactor, scatterPower);
    
    // 白光手电筒特性 (5500K色温)
    vec3 coolWhite = vec3(0.95, 0.98, 1.0);
    vec3 beamColor = mix(vec3(1.0, 0.95, 0.9), coolWhite, 0.7) * lightColor;
    
    float alpha = total * alphaMultiplier;
    vec3 result = beamColor * total;
    
    // 边缘柔化
    float edgeSoftness = 1.0 - smoothstep(0.7, 0.95, cosTheta);
    result *= mix(1.0, 3.0, edgeSoftness);
    
    FragColor = vec4(result, alpha);
}
