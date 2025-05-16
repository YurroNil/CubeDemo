// resources/shaders/fragment/core/light.glsl

#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
in vec4 FragPosLightSpace;

uniform sampler2D texture_diffuse;
uniform sampler2D shadowMap;
uniform vec3 viewPos;

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};
uniform DirLight dirLight;

float ShadowCalculation(vec4 fragPosLightSpace) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    float bias = max(0.05 * (1.0 - dot(Normal, -dirLight.direction)), 0.005);
    
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

// 光照计算
vec3 CalcDirLight(DirLight light, vec3 normal) {
    // 环境光
    vec3 ambient = 4.5*light.ambient * texture(texture_diffuse, TexCoords).rgb;
    
    // 漫反射
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * texture(texture_diffuse, TexCoords).rgb;
    
    // 镜面反射
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = light.specular * spec;

    return (ambient + diffuse + specular);
}

void main() {
    vec3 norm = normalize(Normal);
    vec3 result = CalcDirLight(dirLight, norm);
    result *= (1.0 - ShadowCalculation(FragPosLightSpace));
    
    FragColor = vec4(result, 1.0);
    if(texture(texture_diffuse, TexCoords).a < 0.1) discard;
}