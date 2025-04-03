//res/shaders/core/lit.fsh
#version 330 core
out vec4 FragColor;
 
// Blinn-Phong近似
// Blinn-Phong近似指: 出镜面反射计算使用的是改进的Blinn-Phong方法（比传统Phong更高效）

// 材质结构体
struct Material {
    vec3 ambient;     // 环境光反射系数（RGB）
    vec3 diffuse;     // 漫反射系数（RGB）
    vec3 specular;    // 镜面反射系数（RGB）
    float shininess;  // 镜面高光指数（控制高光大小）
};
 
// 光源结构体
struct Light {
    vec3 position;    // 光源位置（世界空间）
    vec3 color;       // 光源颜色（RGB）
    float ambientStrength;  // 环境光强度系数
    float specularStrength; // 镜面反射强度系数
    // 衰减参数（用于计算距离衰减）
    float constant;   // 常数项
    float linear;     // 线性项
    float quadratic;  // 二次项
};
 
// 从顶点着色器传递的输入变量
in vec3 Normal;       // 法线向量（世界空间）
in vec3 FragPos;      // 片段位置（世界空间）
 
// 统一变量（从外部传入）
uniform Material material;
uniform Light light;
uniform vec3 viewPos; // 观察者位置（世界空间）
 
void main() {
    // ========== 环境光计算 ==========
    // 环境光 = 光源环境强度 * 材质环境反射系数
    vec3 ambient = light.ambientStrength * material.ambient;
    

    // ========== 漫反射计算 ==========
    // 1. 标准化法线向量
    vec3 norm = normalize(Normal);
    // 2. 计算光源到片段方向（世界空间）
    vec3 lightDir = normalize(light.position - FragPos);
    // 3. 计算漫反射强度（Lambert余弦定理）
    float diff = max(dot(norm, lightDir), 0.0);
    // 4. 漫反射颜色 = 漫反射强度 * 光源颜色 * 材质漫反射系数
    vec3 diffuse = diff * light.color * material.diffuse;
    

    // ========== 镜面反射计算 ==========
    // 1. 计算观察方向（世界空间）
    vec3 viewDir = normalize(viewPos - FragPos);
    // 2. 计算反射方向（使用入射光的反方向）
    vec3 reflectDir = reflect(-lightDir, norm);
    // 3. 计算镜面反射强度（Blinn-Phong近似）
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    // 4. 镜面反射颜色 = 镜面强度 * 光源颜色 * 材质镜面系数
    vec3 specular = light.specularStrength * spec * light.color * material.specular;

    // ========== 衰减计算 ==========
    // 计算光源到片段的距离
    float distance = length(light.position - FragPos);
    // 计算衰减因子（使用二次衰减公式）
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    

    // 应用衰减到各光照成分
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    
    
    // 最终颜色 = 环境光 + 漫反射 + 镜面反射
    FragColor = vec4(ambient + diffuse + specular, 1.0);
}