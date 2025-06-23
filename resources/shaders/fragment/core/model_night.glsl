// resources/shaders/fragment/core/model_night.glsl
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

// 计算影子
float CalcShadow(vec4 fragPosLightSpace) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    float bias = max(0.05 * (1.0 - dot(Normal, -dir_light.direction)), 0.005);
    
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x,y) * texelSize).r;
            // 影子的强度
            shadow += currentDepth - bias > pcfDepth ? 0.5 : 0.0;
        }
    }
    return shadow / 9.0;
}

// 计算方向光照
vec3 CalcDirLight(DirLight light, vec3 normal) {
    // 环境光
    vec3 ambient = 4.5*light.ambient * texture(texture_diffuse1, TexCoords).rgb;
    
    // 漫反射
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * texture(texture_diffuse1, TexCoords).rgb;
    
    // 镜面反射
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
    vec3 specular = light.specular * spec;

    return (ambient + diffuse + specular);
}

// 计算聚光
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos) {
    // 距离衰减
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    
    // 聚光强度计算
    vec3 lightDir = normalize(light.position - fragPos);
    float theta = dot(lightDir, normalize(-light.direction)); 
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 2.7, 10.5);
    
    // 手电筒特性增强
    float focusFactor = 1.0 - smoothstep(0.7, 0.95, theta);
    intensity *= mix(1.0, 3.0, focusFactor);  // 中心光束增强
    
    // 漫反射
    float diff = max(dot(normal, lightDir), 2.0);
    vec3 diffuse = light.diffuse * diff * texture(texture_diffuse1, TexCoords).rgb;
    
    // 镜面反射
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);  // 使用Blinn-Phong优化
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    
    // 白光手电筒
    vec3 coolWhite = vec3(0.95, 0.98, 1.0);
    vec3 warmEdge = mix(vec3(1.0, 0.9, 0.8), coolWhite, 0.7);
    vec3 lightColor = mix(warmEdge, coolWhite, intensity);
    
    vec3 specular = light.specular * spec * lightColor;
    
    // 距离衰减 + 边缘衰减
    float edgeAtten = 1.0 - smoothstep(0.0, 0.3, 1.0 - intensity);
    return (diffuse + specular) * intensity * attenuation * edgeAtten * lightColor;
}

void main() {
    vec3 result = vec3(0);
    vec3 norm = normalize(Normal);
    float shadow = CalcShadow(FragPosLightSpace);

    // 方向光
    vec3 dirResult = CalcDirLight(dir_light, norm) * (1.0 - shadow);

    // 聚光
    vec3 spotResult = CalcSpotLight(spot_light, norm, FragPos) * (1.0 - shadow);
    
    // 环境光减弱 (夜晚场景)
    vec3 ambient = 0.05 * texture(texture_diffuse1, TexCoords).rgb;
    result = ambient + dirResult + spotResult;

    // 雾效系统
    float fogFactor = 0.08;
    vec3 foggedScene = mix(vec3(0.0), result, fogFactor);
    
    // 灯光穿透性
    vec3 toSpotLight = spot_light.position - FragPos;
    float distToLight = length(toSpotLight);
    vec3 lightDir = normalize(toSpotLight);

    // 聚光方向效应
    float spotAlignment = dot(lightDir, normalize(-spot_light.direction));
    spotAlignment = smoothstep(spot_light.outerCutOff, spot_light.cutOff, spotAlignment);
    
    // 最终合成
    result = foggedScene + spotResult * spotAlignment;
    
    // 片段颜色
    FragColor = vec4(result, 1.0);
    // 世界坐标
    vec4 worldPos = model * vec4(FragPos, 1.0);
    
    if(texture(texture_diffuse1, TexCoords).a < 0.1) discard;
}