// resources/shaders/fragment/model/preview.glsl
#version 460 core
in vec2 TexCoords;
out vec4 FragColor;

// 只需要基础纹理
uniform sampler2D texture_diffuse1;

// 简化光照
uniform vec3 lightDir = normalize(vec3(0.5, 1.0, 0.7));

void main() {
    // 采样纹理
    vec4 texColor = texture(texture_diffuse1, TexCoords);
    if (texColor.a < 0.1) discard;
    
    // 简单光照计算
    vec3 ambient = vec3(0.2);
    float diff = max(dot(vec3(0.0, 1.0, 0.0), lightDir), 0.0);
    vec3 diffuse = diff * vec3(0.8);
    
    // 最终颜色
    vec3 result = (ambient + diffuse) * texColor.rgb;
    FragColor = vec4(result, texColor.a);
}