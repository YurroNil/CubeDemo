// resources/shaders/fragment/core/model_day.glsl
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
in vec4 FragPosLightSpace;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular;
uniform sampler2D texture_normal;
uniform sampler2D texture_ao;

uniform sampler2D shadowMap;
uniform vec3 viewPos;
uniform mat4 model;

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};
uniform DirLight dir_light;

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    
    float constant;
    float linear;
    float quadratic;
};
uniform SpotLight spot_light;

struct SkyLight {
    vec3 color;
    float intensity;
    float horizonBlend;
};
uniform SkyLight sky_light;

// 计算影子 - 白天阴影更柔和
float CalcShadow(vec4 fragPosLightSpace) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    float bias = max(0.02 * (1.0 - dot(Normal, -dir_light.direction)), 0.005);
    
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    
    // 更柔和的阴影采样
    for(int x = -2; x <= 2; ++x) {
        for(int y = -2; y <= 2; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x,y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 0.2 : 0.0;
        }
    }
    return shadow / 25.0; // 25个采样点
}

// 计算方向光照 - 白天更强
vec3 CalcDirLight(DirLight light, vec3 normal) {
    // 环境光
    vec3 ambient = 2.0 * light.ambient * texture(texture_diffuse1, TexCoords).rgb;
    
    // 漫反射
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * texture(texture_diffuse1, TexCoords).rgb;
    
    // 镜面反射 - 白天更明显
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 128.0); // 更高光泽度
    vec3 specular = light.specular * spec * texture(texture_specular, TexCoords).rgb;

    return (ambient + diffuse + specular);
}

// 计算天光散射
vec3 CalcSkyLight(vec3 normal) {
    // 基于法线方向混合天空颜色
    float horizonFactor = dot(normal, vec3(0, 1, 0)) * 0.5 + 0.5;
    vec3 skyColor = mix(vec3(0.5, 0.6, 0.8), sky_light.color, horizonFactor);
    
    // 强度根据法线与天空方向的夹角变化
    float skyIntensity = max(dot(normal, vec3(0, 1, 0)), 0.0) * sky_light.intensity;
    
    return skyColor * skyIntensity;
}

// 计算聚光 - 白天较弱
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos) {
    // 距离衰减
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    
    // 聚光强度计算
    vec3 lightDir = normalize(light.position - fragPos);
    float theta = dot(lightDir, normalize(-light.direction)); 
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    
    // 漫反射
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * texture(texture_diffuse1, TexCoords).rgb;
    
    // 镜面反射
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = light.specular * spec;
    
    return (diffuse + specular) * intensity * attenuation;
}

void main() {
    vec3 result = vec3(0);
    vec3 norm = normalize(Normal);
    float shadow = CalcShadow(FragPosLightSpace);

    // 方向光（阳光）
    vec3 dirResult = CalcDirLight(dir_light, norm) * (1.0 - shadow);

    // 聚光（补光）
    vec3 spotResult = CalcSpotLight(spot_light, norm, FragPos);
    
    // 天光散射
    vec3 skyResult = CalcSkyLight(norm);
    
    // 环境光基础
    vec3 ambientBase = 0.1 * texture(texture_diffuse1, TexCoords).rgb;
    
    // 最终光照组合
    result = ambientBase + skyResult + dirResult + spotResult;
    
    // 轻微雾效 - 模拟大气透视
    float distance = length(viewPos - FragPos);
    float fogFactor = exp(-distance * 0.01);
    vec3 fogColor = vec3(0.8, 0.85, 0.9); // 天蓝色雾
    result = mix(fogColor, result, fogFactor);
    
    // 饱和度增强
    vec3 saturated = mix(result, result * vec3(1.1, 1.05, 1.0), 0.3);
    
    // 片段颜色
    FragColor = vec4(saturated, 1.0);
    
    // 透明度测试
    if(texture(texture_diffuse1, TexCoords).a < 0.1) discard;
}
