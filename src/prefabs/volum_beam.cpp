// src/prefabs/volum_beam.cpp
#include "pch.h"
#include "prefabs/volum_beam_inc.h"

namespace CubeDemo::Prefabs {
using namespace glm;

// 别名
using TLS = CubeDemo::Texture::LoadState;

VolumBeam::VolumBeam() {
}
VolumBeam::~VolumBeam() {
    if (coneVAO) glDeleteVertexArrays(1, &coneVAO);
    if (coneVBO) glDeleteBuffers(1, &coneVBO);
    if (VolumShader) delete VolumShader;
}

void VolumBeam::Init() {
    try {
        // 创建体积光着色器
        CreateVolumShader();
        
        // 创建光锥几何体
        const float radius = (Effects.radius > 0) ? Effects.radius : 1.0f;
        const float height = (Effects.height > 0) ? Effects.height : 5.0f;
        CreateLightCone(radius, height);
        
        // 验证资源创建
        if (!VolumShader) {
            throw std::runtime_error("体积光着色器创建失败");
        }
        
        if (coneVAO == 0) {
            throw std::runtime_error("光锥几何体创建失败");
        }
        
        std::cout << "体积光初始化成功 (半径: " << radius << ", 高度: " << height << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "体积光初始化失败: " << e.what() << std::endl;
        // 清理部分创建的资源
        if (coneVAO) glDeleteVertexArrays(1, &coneVAO);
        if (coneVBO) glDeleteBuffers(1, &coneVBO);
        if (VolumShader) delete VolumShader;
        coneVAO = coneVBO = 0;
        VolumShader = nullptr;
        throw; // 重新抛出异常
    }
}

// 设置动态效果+应用着色器
void VolumBeam::SetFx(Camera* camera, SL* spot_light) {

    if(spot_light == nullptr) return;

    // 别名
    const auto& s = VolumShader;
    const auto& fx = Effects;

    // 计算闪烁效果
    float flicker_factor = 1.0f;
    if (Effects.flicker.enable) {
        flicker_factor = mix(
            Effects.flicker.min, 
            Effects.flicker.max, 
            0.5f + 0.5f * sin(glfwGetTime() * Effects.flicker.speed)
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
            WINDOW::GetAspectRatio(),
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
    s->SetFloat("density", Effects.density);
    s->SetFloat("scatterAnisotropy", Effects.scatterAnisotropy);
}

// 渲染体积光束
void VolumBeam::Render(Camera* camera, SL* spot_light) {

    if (spot_light == nullptr) {
        std::cerr << "体积光渲染失败: 聚光灯为空" << std::endl;
        return;
    }
    
    if (VolumShader == nullptr) {
        std::cerr << "体积光渲染失败: 着色器未初始化" << std::endl;
        return;
    }
    
    if (coneVAO == 0) {
        std::cerr << "体积光渲染失败: 几何体未初始化" << std::endl;
        return;
    }

    // 设置动态效果（变换矩阵、着色器参数等）
    SetFx(camera, spot_light);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);

    // 绑定噪声纹理（如果可用且已加载完成）
    if (NoiseTexture && NoiseTexture->State.load() == TLS::Ready) {
        // 确保在主线程绑定纹理
        TaskQueue::PushTaskSync([this] {
            GLuint texId = NoiseTexture->ID.load();
            if (texId != 0) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, texId);
                VolumShader->SetInt("noiseTex", 0);
            }
        });
    } else {
        // 没有可用纹理时使用默认值
        VolumShader->SetInt("noiseTex", 0);
    }

    VolumShader->Use();
    
    // 根据距离动态调整分段数 (LOD)
    VolumShader->SetInt("segments", 64);
    
    // 绘制体积光束 (使用几何着色器)
    glBindVertexArray(coneVAO);
    glDrawArrays(GL_POINTS, 0, 1); // 仅绘制一个点
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

/* 光束变换矩阵
    最终矩阵 = 平移矩阵 * 旋转矩阵 * 缩放矩阵
            = 先缩放 → 再旋转 → 最后平移
*/
mat4 VolumBeam::CalcTransform(SL* spot_light) {

    if(spot_light == nullptr) return mat4(0);

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
    // 定义顶点结构
    struct ConeVertex {
        vec3 tip;
        vec3 base;
        float radius;
    };
    
    // 单个顶点包含所有必要信息
    ConeVertex vertex = {
        vec3(0.0f, height, 0.0f), // 圆锥顶点
        vec3(0.0f, 0.0f, 0.0f),    // 底面中心
        radius
    };
    
    // 创建VAO和VBO
    glGenVertexArrays(1, &coneVAO);
    glGenBuffers(1, &coneVBO);
    
    // 检查OpenGL错误
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        throw std::runtime_error("创建OpenGL缓冲区失败: " + std::to_string(error));
    }
    
    if (coneVAO == 0 || coneVBO == 0) {
        throw std::runtime_error("光锥几何体创建失败 (VAO/VBO为0)");
    }

    
    glBindVertexArray(coneVAO);
    glBindBuffer(GL_ARRAY_BUFFER, coneVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(ConeVertex), &vertex, GL_STATIC_DRAW);
    
    // 设置顶点属性指针
    // 位置0：圆锥顶点
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ConeVertex), (void*)0);
    glEnableVertexAttribArray(0);
    
    // 位置1：底面中心
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ConeVertex), (void*)offsetof(ConeVertex, base));
    glEnableVertexAttribArray(1);
    
    // 位置2：半径
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(ConeVertex), (void*)offsetof(ConeVertex, radius));
    glEnableVertexAttribArray(2);
    
    glBindVertexArray(0);
}

// 创建体积光着色器 (添加几何着色器)
void VolumBeam::CreateVolumShader() {
    VolumShader = new Shader(
        VSH_PATH + string("volumetric.glsl"),
        FSH_PATH + string("volumetric.glsl")
    );
}

void VolumBeam::SetTextureArgs(const string& path) {
    const string& full_path = TEX_PATH + path;
try {
    // 同步加载纹理
    NoiseTexture = TL::LoadSync(full_path, "noise");
    
    // 设置纹理参数（在纹理创建后立即设置）
    TaskQueue::AddTasks([this] {
        if (NoiseTexture && NoiseTexture->State == TLS::Ready) {
            GLuint texId = NoiseTexture->ID.load();
            glBindTexture(GL_TEXTURE_2D, texId);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }, false);
} 
catch (const std::exception& e) {
    std::cerr << "加载噪声纹理失败(" << full_path << "): " << e.what() << std::endl;
    NoiseTexture = nullptr;
}}

void VolumBeam::Configure(const BeamEffects& effects) {
    Effects = effects;
    // 加载噪声纹理
    if (!effects.noiseTexture.empty()) SetTextureArgs(effects.noiseTexture);
}

void VolumBeam::LoadNoiseTexture(const string& path) {
    // 释放旧纹理引用
    if (NoiseTexture) NoiseTexture = nullptr;
    // 加载噪声纹理
    SetTextureArgs(path);
}
void VolumBeam::CleanUp() {
    if (coneVAO) {
        glDeleteVertexArrays(1, &coneVAO);
        coneVAO = 0;
    }
    if (coneVBO) {
        glDeleteBuffers(1, &coneVBO);
        coneVBO = 0;
    }
    if (VolumShader) {
        delete VolumShader;
        VolumShader = nullptr;
    }
    NoiseTexture = nullptr; // 纹理由纹理管理器管理，不需要手动删除
}
}   // namespace CubeDemo::Prefabs
