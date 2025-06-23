// resources/shaders/fragment/core/model_default.glsl
#version 450 core

in vec3 FragPos;        // 片段世界坐标
in vec3 Normal;         // 法线向量
in vec2 TexCoords;      // 纹理坐标
out vec4 FragColor;     // 输出颜色

// 纹理和平行光结构
uniform sampler2D texture_diffuse1;

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

// 相机位置（用于镜面反射）
uniform vec3 viewPos;

void main() {
    // 采样纹理获取基础颜色
    vec3 baseColor = texture(texture_diffuse1, TexCoords).rgb;

    // 标准化法线和光线方向
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-dir_light.direction); // 平行光方向取反

    // 环境光照（使用DirLight中的环境光参数）
    vec3 ambient = dir_light.ambient * baseColor;

    // 漫反射光照
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = dir_light.diffuse * (diff * baseColor);

    // 镜面反射（高光）
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0); // 32为反光度
    vec3 specular = dir_light.specular * spec;

    // 组合光照结果
    vec3 lighting = ambient + diffuse + specular;

    // 天空渐变效果（基于Y轴高度）
    float heightFactor = clamp(FragPos.y / 100.0, 0.0, 1.0);
    vec3 skyColor = mix(dir_light.skyColor * 0.6, dir_light.skyColor, heightFactor);

    // 光源柔和边缘（模拟大气散射）
    float edgeSoftness = 1.0 - smoothstep(
        dir_light.sourceRadius, 
        dir_light.sourceRadius + dir_light.sourceSoftness, 
        length(FragPos)
    );
    vec3 atmospheric = edgeSoftness * dir_light.skyColor * dir_light.atmosphereThickness;

    // 最终合成（物体光照 + 大气散射 + 天空背景）
    vec3 finalColor = lighting + atmospheric;

    FragColor = vec4(finalColor, 1.0);
}
