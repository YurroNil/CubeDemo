#version 450 core
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

// 早晨场景的光源设置
struct DirLight {
    vec3 direction;     // 早晨阳光方向 (偏东方向)
    vec3 ambient;       // 环境光 (柔和的蓝色调)
    vec3 diffuse;       // 漫反射 (温暖的橙黄色)
    vec3 specular;      // 高光 (明亮的白色)
};
uniform DirLight dir_light;

// 点光源 (用于模拟路灯、窗户等)
struct PointLight {
    vec3 position;
    vec3 color;
    float intensity;
    float constant;
    float linear;
    float quadratic;
    float radius;
};
#define MAX_POINT_LIGHTS 4
uniform PointLight point_lights[MAX_POINT_LIGHTS];
uniform int num_point_lights;

// 聚光灯 (用于特殊区域)
struct SpotLight {
    vec3 position;
    vec3 direction;
    vec3 color;
    float cutOff;
    float outerCutOff;
    float intensity;
};
#define MAX_SPOT_LIGHTS 2
uniform SpotLight spot_lights[MAX_SPOT_LIGHTS];
uniform int num_spot_lights;

// 计算影子 (优化版本)
float CalcShadow(vec4 fragPosLightSpace) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    if(projCoords.z > 1.0) return 0.0; // 避免超出阴影贴图范围
    
    float currentDepth = projCoords.z;
    float bias = max(0.0005 * (1.0 - dot(Normal, dir_light.direction)), 0.0001);
    
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 0.5 : 0.0;
        }
    }
    return shadow / 9.0;
}

// 计算方向光照 (早晨阳光)
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
    // 环境光 - 柔和的蓝色调
    vec3 ambient = light.ambient * texture(texture_diffuse1, TexCoords).rgb;
    
    // 漫反射 - 温暖的阳光
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * texture(texture_diffuse1, TexCoords).rgb;
    
    // 镜面反射 - 明亮的反光
    vec3 halfwayDir = normalize(lightDir + viewDir); // Blinn-Phong优化
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 specular = light.specular * spec * texture(texture_specular, TexCoords).rgb;
    
    // 早晨特有的柔和阴影
    float shadow = CalcShadow(FragPosLightSpace) * 0.7; // 早晨阴影较淡
    
    return ambient + (diffuse + specular) * (1.0 - shadow);
}

// 计算点光源 (路灯、窗户等)
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    // 距离衰减
    float distance = length(light.position - fragPos);
    if(distance > light.radius) return vec3(0.0);
    
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                             light.quadratic * (distance * distance));
    
    // 漫反射
    vec3 lightDir = normalize(light.position - fragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff * texture(texture_diffuse1, TexCoords).rgb;
    
    // 镜面反射
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = light.color * spec * texture(texture_specular, TexCoords).rgb;
    
    // 应用AO贴图
    vec3 ao = texture(texture_ao, TexCoords).rgb;
    
    return (diffuse + specular) * attenuation * light.intensity * ao;
}

// 计算聚光 (特殊区域照明)
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // 聚光角度衰减
    float theta = dot(lightDir, normalize(-light.direction)); 
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    
    // 漫反射
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff * texture(texture_diffuse1, TexCoords).rgb;
    
    // 镜面反射
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = light.color * spec * texture(texture_specular, TexCoords).rgb;
    
    // 距离衰减
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * (distance * distance));
    
    // 应用AO贴图
    vec3 ao = texture(texture_ao, TexCoords).rgb;
    
    return (diffuse + specular) * intensity * attenuation * light.intensity * ao;
}

void main() {
    // 丢弃透明片段
    if(texture(texture_diffuse1, TexCoords).a < 0.1) discard;
    
    // 基础属性
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // 早晨主光照 - 方向光
    vec3 result = CalcDirLight(dir_light, norm, viewDir);
    
    // 点光源 - 路灯、窗户等
    for(int i = 0; i < num_point_lights; i++) {
        result += CalcPointLight(point_lights[i], norm, FragPos, viewDir);
    }
    
    // 聚光灯 - 特殊区域
    for(int i = 0; i < num_spot_lights; i++) {
        result += CalcSpotLight(spot_lights[i], norm, FragPos, viewDir);
    }
    
    // 早晨雾效 - 柔和的蓝色调
    float fogDensity = 0.02;
    float fogDistance = length(viewPos - FragPos);
    float fogFactor = exp(-fogDensity * fogDensity * fogDistance * fogDistance);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    
    vec3 fogColor = vec3(0.7, 0.8, 0.9); // 早晨天空的淡蓝色
    result = mix(fogColor, result, fogFactor);
    
    // 最终颜色
    FragColor = vec4(result, 1.0);
}