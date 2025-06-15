// src/prefabs/lights/volum_beam.cpp
#include "pch.h"
#include "prefabs/lights/volum_beam_inc.h"

namespace CubeDemo::Prefabs {
using namespace glm;

// 别名
using TLS = CubeDemo::Texture::LoadState;

// 设置动态效果+应用着色器
void VolumBeam::SetFx(Camera* camera, SL* spot_light) {
    // 别名
    const auto& s = m_VolumShader;
    const auto& fx = m_effects;

    // 计算闪烁效果
    float flicker_factor = 1.0f;
    if (m_effects.flicker.enable) {
        flicker_factor = mix(
            m_effects.flicker.min, 
            m_effects.flicker.max, 
            0.5f + 0.5f * sin(glfwGetTime() * m_effects.flicker.speed)
        );
    }

    // 计算光束变换矩阵
    mat4 model = CalcTransform(spot_light);

    // 设置着色器
    s->SetMat4("model", model);
    s->SetMat4("view", camera->GetViewMat());
    s->SetMat4(
        "projection", 
        perspective(
            radians(camera->attribute.zoom),
            Window::GetAspectRatio(),
            camera->frustumPlane.near_plane,
            camera->frustumPlane.far_plane
        )
    );
    s->SetVec3("viewPos", camera->Position);


    s->SetVec3("lightPos", spot_light->position);
    s->SetVec3("lightDir", spot_light->direction);
    s->SetVec3("lightColor", spot_light->diffuse);
    s->SetFloat("lightIntensity", fx.intensity * flicker_factor);
    s->SetFloat("lightCutOff", cos(radians(spot_light->cutOff)));
    s->SetFloat("lightOuterCutOff", cos(radians(spot_light->outerCutOff)));
    s->SetFloat("scatterPower", fx.scatterPower);
    s->SetVec2("attenuationFactors", fx.attenuationFactors);
    s->SetFloat("alphaMultiplier", fx.alphaMultiplier);
    s->SetFloat("time", glfwGetTime());
    s->SetFloat("density", m_effects.density);
    s->SetFloat("scatterAnisotropy", m_effects.scatterAnisotropy);
}

// 渲染体积光束
void VolumBeam::Render(Camera* camera, SL* spot_light) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);


    // 绑定噪声纹理（如果可用且已加载完成）
    if (m_noiseTexture && m_noiseTexture->State.load() == TLS::Ready) {
        // 确保在主线程绑定纹理
        TaskQueue::PushTaskSync([this] {
            GLuint texId = m_noiseTexture->ID.load();
            if (texId != 0) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, texId);
                m_VolumShader->SetInt("noiseTex", 0);
            }
        });
    } else {
        // 没有可用纹理时使用默认值
        m_VolumShader->SetInt("noiseTex", 0);
    }

    m_VolumShader->Use();
    
    // 设置效果
    SetFx(camera, spot_light);

    // 绘制体积光束
    m_LightVolume->Draw(m_VolumShader);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

/* 光束变换矩阵
    最终矩阵 = 平移矩阵 * 旋转矩阵 * 缩放矩阵
            = 先缩放 → 再旋转 → 最后平移
*/
mat4 VolumBeam::CalcTransform(SL* spot_light) {
    vec3 targetDir = normalize(spot_light->direction);
    vec3 defaultDir = vec3(0.0f, -1.0f, 0.0f); // 圆锥默认方向
    
    // 计算旋转轴和角度
    float dotVal = dot(defaultDir, targetDir);
    vec3 axis = (abs(dotVal) > 0.9999f) ? 
                vec3(1.0f, 0.0f, 0.0f) : // 避免叉积为零
                normalize(cross(defaultDir, targetDir));
    
    float angle = acos(clamp(dotVal, -1.0f, 1.0f));
    
    // 构建变换矩阵
    mat4 transform = translate(mat4(1.0f), spot_light->position);
    transform = rotate(transform, angle, axis);
    transform = scale(transform, vec3(1, 15.0f, 1));
    
    return transform;
}

/* 光锥几何体的创建
    底面圆周顶点参数方程: x = rad*cosθ; y = 0; z = rad*sin θ;
        θ= 2π*index / segments;
        index = 0, 1, 2, ..., segments - 1

    每个侧面三角形由底面两个相邻点和圆锥顶点构成: 
        indices={ i, (i+1) mod segments, segments }
*/
// 创建光锥几何体
void VolumBeam::CreateLightCone(float radius, float height) {
      std::vector<Vertex> vertices;
    std::vector<unsigned> indices;
    
    const int segments = 128; // 圆周分段数（值越高锥面越平滑）
    const float centerY = height; // 圆锥顶点Y坐标（底部在Y=0处）
    
    // ============== 1. 生成底面圆周顶点 ==============
    for(int i = 0; i < segments; ++i) {
        // 计算圆周角度（0~360度）
        float angle = glm::radians(360.0f * i / segments);
        
        // 计算底面顶点坐标（XZ平面）
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        
        // 计算法线（垂直于圆锥侧面）
        vec3 normal = normalize(vec3(x, 0, z));
        
        // 计算自发光颜色梯度（核心过渡->边缘）
        float distToCenter = distance(vec3(x, 0, z), vec3(0)); // 当前点到底面中心距离
        vec3 emitColor = mix(
            vec3(0.0f),            // 中心色（白）
            vec3(1.0f),            // 边缘色（白）
            distToCenter / radius  // 颜色混合比例（0=中心，1=边缘）
        );
        
        // 添加到顶点数组（底部顶点）
        vertices.push_back({
            vec3(x, 0, z),  // 位置
            normal,          // 法线
            vec2(0),         // UV坐标（未使用）
            emitColor        // 自发光颜色
        });
    }
    
    // 添加圆锥顶点
    vec3 topNormal = normalize(vec3(0, height, 0)); // 顶点法线（垂直向上）
    vertices.push_back({
        vec3(0, centerY, 0), // 顶点位置（圆锥顶部）
        topNormal,            // 法线
        vec2(0),              // UV坐标
        vec3(1.0f) // 自发光（白色）
    });
    
    // 生成侧面三角形
    for(int i = 0; i < segments; ++i) {
        int next = (i + 1) % segments;
        
        // 每个侧面由两个底面点+顶点构成
        indices.push_back(i);          // 当前底面点
        indices.push_back(next);       // 下一个底面点
        indices.push_back(segments);    // 圆锥顶点
    }
    
    // 生成底面三角形
    // for(int i = 2; i < segments; ++i) {
    //     indices.push_back(0);      // 第一个顶点（固定）
    //     indices.push_back(i-1);    // 前一个点
    //     indices.push_back(i);      // 当前点
    // }
    
    // 创建网格对象
    m_LightVolume = new Mesh(vertices, indices, {});
}
// 创建光锥着色器
void VolumBeam::CreateVolumShader() {
    m_VolumShader = new Shader(
        VSH_PATH + string("volumetric.glsl"),
        FSH_PATH + string("volumetric.glsl")
    );
}

void VolumBeam::SetTextureArgs(const string& path) {
    const string& full_path = TEX_PATH + path;
try {
    // 同步加载纹理
    m_noiseTexture = TL::LoadSync(full_path, "noise");
    
    // 设置纹理参数（在纹理创建后立即设置）
    TaskQueue::AddTasks([this] {
        if (m_noiseTexture && m_noiseTexture->State == TLS::Ready) {
            GLuint texId = m_noiseTexture->ID.load();
            glBindTexture(GL_TEXTURE_2D, texId);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }, false);
} 
catch (const std::exception& e) {
    std::cerr << "加载噪声纹理失败(" << full_path << "): " << e.what() << std::endl;
    m_noiseTexture = nullptr;
}}

void VolumBeam::Configure(const BeamEffects& effects) {
    m_effects = effects;
    // 加载噪声纹理
    if (!effects.noiseTexture.empty()) SetTextureArgs(effects.noiseTexture);
}

void VolumBeam::LoadNoiseTexture(const string& path) {
    // 释放旧纹理引用
    if (m_noiseTexture) m_noiseTexture = nullptr;
    // 加载噪声纹理
    SetTextureArgs(path);
}

// Getters
Shader* VolumBeam::GetVolumShader() const { return m_VolumShader; }
Mesh* VolumBeam::GetLightVolume() const { return m_LightVolume; }
const BeamEffects& VolumBeam::GetEffects() const { return m_effects; }
}
