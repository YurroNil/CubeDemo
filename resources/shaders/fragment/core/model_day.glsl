// resources/shaders/fragment/core/model_day.glsl
#version 450 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
out vec4 FragColor;

// 纹理与光照结构
uniform sampler2D texture_diffuse1;
uniform sampler2D noiseTex; // 云层噪声纹理
uniform sampler2D aoMap;   // 环境光遮蔽贴图

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float sourceRadius;
    float sourceSoftness;
    vec3 skyColor;
    float atmosphereThickness;
};
uniform DirLight dir_light;

uniform vec3 viewPos;
uniform float time; // 时间变量驱动动态效果

// 物理光照增强函数
vec3 calculateAtmosphericScattering(vec3 fragPos, vec3 viewDir) {
    // 瑞利散射模拟（波长相关）
    const vec3 rayleighCoeff = vec3(5.8e-6, 1.35e-5, 3.31e-5); // 蓝光散射最强
    float scatterIntensity = 1.0 - clamp(dot(normalize(-dir_light.direction), viewDir), 0.0, 1.0);
    vec3 rayleigh = rayleighCoeff * scatterIntensity * dir_light.atmosphereThickness;

    // 米氏散射（日光边缘柔化）
    float distToLight = length(fragPos);
    float mieFactor = smoothstep(
        dir_light.sourceRadius, 
        dir_light.sourceRadius + dir_light.sourceSoftness, 
        distToLight
    );
    return (1.0 - mieFactor) * rayleigh * dir_light.skyColor;
}

void main() {
    // 基础纹理采样（增加细节层）
    vec3 baseColor = texture(texture_diffuse1, TexCoords).rgb;
    vec3 aoValue = texture(aoMap, TexCoords).rrr; // AO贴图采样
    
    // 法线与视线计算
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-dir_light.direction);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfDir = normalize(lightDir + viewDir);

    // PBR光照计算
    float NdotL = max(dot(norm, lightDir), 0.0);
    float NdotV = max(dot(norm, viewDir), 0.01);
    
    // 微表面高光（GGX模型）
    float roughness = 0.3;
    float NdotH = max(dot(norm, halfDir), 0.0);
    float alpha = roughness * roughness;
    float denom = (NdotH * alpha - NdotH) * NdotH + 1.0;
    float specular = alpha / (3.14159 * denom * denom);
    
    // 能量守恒合成
    vec3 F0 = vec3(0.04); // 基础反射率
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
    vec3 kD = (vec3(1.0) - F) * (1.0 - specular);
    
    vec3 diffuse = dir_light.diffuse * NdotL * baseColor * kD;
    vec3 spec = dir_light.specular * specular * F;
    vec3 ambient = dir_light.ambient * baseColor * aoValue; // AO影响环境光

    vec3 lighting = ambient + diffuse + spec;

    // 多层次天空渲染 ----------------------------------------
    // 基础渐变层
    float heightFactor = clamp(FragPos.y / 100.0, 0.0, 1.0);
    vec3 skyBase = mix(dir_light.skyColor * 0.6, dir_light.skyColor, heightFactor);
    
    // 云层动态效果
    vec2 cloudUV = FragPos.xz / 200.0 + time * 0.01;
    float cloudNoise = texture(noiseTex, cloudUV * 2.0).r;
    float cloudMask = smoothstep(0.3, 0.8, cloudNoise) * dir_light.cloudOpacity;
    
    // 地平线光晕（太阳方位相关）
    vec3 sunDir = -dir_light.direction;
    float horizonGlow = pow(1.0 - abs(sunDir.y), 8.0) * 2.0;
    vec3 glowColor = mix(vec3(1.0, 0.7, 0.4), dir_light.skyColor, 0.5);
    
    // 体积光束（需在CPU端计算光源空间矩阵）
    vec3 lightSpacePos = (lightViewMatrix * vec4(FragPos, 1.0)).xyz;
    float beamAtten = smoothstep(0.8, 0.2, length(lightSpacePos.xy));
    vec3 godRays = beamAtten * glowColor * horizonGlow * 0.3;

    // 大气散射合成
    vec3 scattering = calculateAtmosphericScattering(FragPos, viewDir);
    vec3 finalSky = skyBase * (1.0 - cloudMask) + dir_light.cloudColor * cloudMask + godRays + scattering;

    // 最终合成 ----------------------------------------------
    float depthFactor = smoothstep(200.0, 500.0, length(FragPos - viewPos));
    vec3 finalColor = mix(lighting, finalSky, depthFactor);
    
    // 应用ACES色调映射（电影级色彩空间）
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    finalColor = clamp((finalColor * (a * finalColor + b)) / (finalColor * (c * finalColor + d) + e), 0.0, 1.0);
    
    FragColor = vec4(pow(finalColor, vec3(1.0/2.2)), 1.0); // Gamma校正
}
