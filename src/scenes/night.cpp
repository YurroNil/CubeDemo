// src/scenes/night.cpp

#include "scenes/sceneMng.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>
#include "utils/defines.h"
#include "utils/jsonConfig.h"

// 外部变量声明
namespace CubeDemo {
    extern Shader* MODEL_SHADER;
    extern std::vector<Model*> MODEL_POINTERS;
    extern bool DEBUG_LOD_MODE;
}

namespace CubeDemo::Scenes {

// 场景初始化
void NightScene::Init(SceneMng* scene_inst, Light& light) {
    if(s_isInited) return;
    
    // 创建月光（方向光）
    m_MoonLight = light.Create.DirLight();
    light.Get.SetDirLight(m_MoonLight);

    // 创建聚光灯（光束）
    m_SpotLight = light.Create.SpotLight();
    light.Get.SetSpotLight(m_SpotLight);

    // 使用配置文件的数据来设置参数
    SetLightsData(SCENES_CONFIG + string("night.json"), m_SpotLight, m_MoonLight);

    // 加载体积光着色器
    m_VolumetricShader = new Shader(
        VSH_PATH + string("volumetric.glsl"),
        FSH_PATH + string("volumetric.glsl")
    );

    // 创建光束几何体（圆锥）
    m_LightVolume = CreateLightCone(5.0f, 15.0f);

    scene_inst->Current = SceneID::NIGHT;
}
// 场景清理
void NightScene::Cleanup(Light& light) {
    if(!s_isInited || s_isCleanup) return;

    delete m_VolumetricShader; m_VolumetricShader = nullptr;
    delete m_LightVolume; m_LightVolume = nullptr;

    light.Remove.DirLight();
    light.Remove.SpotLight();
}

// 渲染场景
void NightScene::Render(GLFWwindow* window,
    Camera* camera,
    const Light& light,
    ShadowMap* shadow_map)
{
    glViewport(0, 0, Window::GetWidth(), Window::GetHeight());
    // 主着色器配置
    MODEL_SHADER->Use();

    shadow_map->BindForReading(GL_TEXTURE1);

    // 摄像机参数传递
    MODEL_SHADER->ApplyCamera(*camera, Window::GetAspectRatio());

    // 模型绘制循环
    for (auto* model : MODEL_POINTERS) {
        if (!model->IsReady()) {
            std::cout << "[Render] 模型未就绪: " << model << std::endl;
            continue;
        }

        // 视椎体裁剪判断
        if (model->IsReady() &&
            camera->isSphereVisible(model->bounds.Center, model->bounds.Rad)
        ) {
            model->DrawCall(DEBUG_LOD_MODE, *MODEL_SHADER, camera->Position);
        }
    }

    // 设置月光
    MODEL_SHADER->SetDirLight("dirLight", m_MoonLight);
    MODEL_SHADER->SetViewPos(camera->Position);

    // 设置聚光灯
    MODEL_SHADER->SetSpotLight("spotLight", *m_SpotLight);
    
    // 体积光渲染
    RenderVolumetricBeam(camera);
}

/* 私有成员 */

// 渲染体积光束
void NightScene::RenderVolumetricBeam(Camera* camera) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);

    m_VolumetricShader->Use();
    
    // 计算光束变换矩阵
    mat4 model = CalcBeamTransform();
    m_VolumetricShader->SetMat4("model", model);
    m_VolumetricShader->SetMat4("view", camera->GetViewMat());
    m_VolumetricShader->SetMat4(
        "projection", 
        glm::perspective(glm::radians(camera->attribute.zoom),
        Window::GetAspectRatio(),
        camera->frustumPlane.near,
        camera->frustumPlane.far
    ));
    m_VolumetricShader->SetVec3("lightColor", m_SpotLight->diffuse);

    m_LightVolume->Draw(*m_VolumetricShader);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

/* 光束变换矩阵
    最终矩阵 = 平移矩阵 * 旋转矩阵 * 缩放矩阵
            = 先缩放 → 再旋转 → 最后平移
*/
mat4 NightScene::CalcBeamTransform() {
    vec3 direction = normalize(m_SpotLight->direction);
    float length = 15.0f; // 光束长度
    
    mat4 transform = glm::translate(mat4(1.0f), m_SpotLight->position);
    transform *= glm::orientation(direction, vec3(0, -1, 0)); // 对齐方向
    transform = glm::scale(transform, vec3(1, length, 1));
    
    return transform;
}

/* 光锥几何体的创建
    底面圆周顶点参数方程: x = rad*cosθ; y = 0; z = rad*sin θ;
        θ= 2π*index / segments;
        index = 0, 1, 2, ..., segments - 1

    每个侧面三角形由底面两个相邻点和圆锥顶点构成: 
        indices={ i, (i+1) mod segments, segments }
*/

Mesh* NightScene::CreateLightCone(float radius, float height) {
    std::vector<Vertex> vertices;
    std::vector<unsigned> indices;
    
    // 圆锥底面
    const int segments = 32;
    for(int i = 0; i < segments; ++i) {
        float angle = glm::radians(360.0f * i / segments);
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        
        vertices.push_back({vec3(x, 0, z), vec3(0), vec2(0), vec3(0)});
    }
    
    // 圆锥顶点
    vertices.push_back({vec3(0, height, 0), vec3(0), vec2(0), vec3(0)});
    
    // 生成索引
    for(int i = 0; i < segments; ++i) {
        indices.push_back(i);
        indices.push_back((i+1)%segments);
        indices.push_back(segments);
    }
    
    return new Mesh(vertices, indices, {});
}

void NightScene::SetLightsData(const string& config_path, SL* spot_light, DL* moon_light) {

    json config = Utils::JsonConfig::GetFileData(config_path);
    
    const auto& spot = config["LightArgs"]["SpotLight"];
    const auto& moon = config["LightArgs"]["MoonLight"];

    // 位置
    spot_light->position.x = spot["position"][0].get<float>();
    spot_light->position.y = spot["position"][1].get<float>();
    spot_light->position.z = spot["position"][2].get<float>();
    
    moon_light->position.x = moon["position"][0].get<float>();
    moon_light->position.y = moon["position"][1].get<float>();
    moon_light->position.z = moon["position"][2].get<float>();
 
    // 方向向量
    spot_light->direction.x = spot["direction"][0].get<float>();
    spot_light->direction.y = spot["direction"][1].get<float>();
    spot_light->direction.z = spot["direction"][2].get<float>();
    
    moon_light->direction.x = moon["direction"][0].get<float>();
    moon_light->direction.y = moon["direction"][1].get<float>();
    moon_light->direction.z = moon["direction"][2].get<float>();
 
    // 颜色分量
    spot_light->ambient.x = spot["ambient"][0].get<float>();
    spot_light->ambient.y = spot["ambient"][1].get<float>();
    spot_light->ambient.z = spot["ambient"][2].get<float>();

    moon_light->ambient.x = moon["ambient"][0].get<float>();
    moon_light->ambient.y = moon["ambient"][1].get<float>();
    moon_light->ambient.z = moon["ambient"][2].get<float>();

    // 漫反射
    spot_light->diffuse.x = spot["diffuse"][0].get<float>();
    spot_light->diffuse.y = spot["diffuse"][1].get<float>();
    spot_light->diffuse.z = spot["diffuse"][2].get<float>();
    
    moon_light->diffuse.x = moon["diffuse"][0].get<float>();
    moon_light->diffuse.y = moon["diffuse"][1].get<float>();
    moon_light->diffuse.z = moon["diffuse"][2].get<float>();

    // 镜面反射
    spot_light->specular.x = spot["specular"][0].get<float>();
    spot_light->specular.y = spot["specular"][1].get<float>();
    spot_light->specular.z = spot["specular"][2].get<float>();
    
    moon_light->specular.x = moon["specular"][0].get<float>();
    moon_light->specular.y = moon["specular"][1].get<float>();
    moon_light->specular.z = moon["specular"][2].get<float>();

    spot_light->constant = spot["constant"].get<float>();
    spot_light->linear = spot["linear"].get<float>();
    spot_light->quadratic = spot["quadratic"].get<float>();
    spot_light->cutOff = spot["cutOff"].get<float>();
    spot_light->outerCutOff = spot["outerCutOff"].get<float>();
}

NightScene::NightScene() {}
NightScene::~NightScene() {}

} // namespace
