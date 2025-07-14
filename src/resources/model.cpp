// src/resources/model.cpp
#include "pch.h"
#include "resources/model.h"
#include "loaders/model.h"

using ML = CubeDemo::Loaders::Model;

namespace CubeDemo {

extern std::vector<::CubeDemo::Model*> MODEL_POINTERS;
extern unsigned int DEBUG_INFO_LV;

// Model构造函数传递路径到Loaders:Model类
Model::Model(const string& path)
    : m_Path(path), Managers::ModelGetter(this) {}

void Model::Init() {
}

// 绘制函数. 判断是否符合条件，若符合条件则调用NormalDraw来更进一步地绘制网格
void Model::DrawCall(Camera* camera, bool is_mainloop_draw) {
    // 模型绘制循环
    if (!IsReady()) {
        std::cout << "[Render] 模型未就绪: " << this << std::endl;
        return;
    }
    // 视椎体裁剪判断
    // if (IsReady() && camera->isSphereVsble(bounds.Center, bounds.Rad)) { ... }
    NormalDraw(is_mainloop_draw);
    // 使用着色器

}
// 普通模式绘制模型
void Model::NormalDraw(bool is_mainloop_draw) {

    if(is_mainloop_draw) ModelShader->SetMat4("model", m_ModelMatrix);

    if(m_isLoading.load() || m_Meshes.empty() ) return; // 加载中不绘制
    // 绘制该模型中的所有网格
    for (const Mesh& mesh : m_Meshes) mesh.Draw(ModelShader);
}

void Model::UseShaders(
    Camera* camera,
    DL* dir_light, SL* spot_light,
    PL* point_light, SkL* sky_light)
{
    // 使用着色器
    ModelShader->Use();
    // 摄像机参数传递
    ModelShader->ApplyCamera(camera, WINDOW::GetAspectRatio());
    // 设置视点
    ModelShader->SetViewPos(camera->Position);

    // 传递模型的着色器指针，给光源setter来获取光源信息
    if(dir_light != nullptr) dir_light->SetShader(*ModelShader);
    if(spot_light != nullptr) spot_light->SetShader(*ModelShader);
    if(point_light != nullptr) point_light->SetShader(*ModelShader);
    if(sky_light != nullptr) sky_light->SetShader(*ModelShader);

    // 设置模型变换
    ModelShader->SetMat4("model", GetModelMatrix());
}
// 更新模型矩阵
const void Model::UpdateModelMatrix() {
    m_ModelMatrix = mat4(1.0f);
    m_ModelMatrix = translate(m_ModelMatrix, m_Position);
    
    m_ModelMatrix = rotate(m_ModelMatrix, m_Rotation.x, vec3(1.0f, 0.0f, 0.0f));
    m_ModelMatrix = rotate(m_ModelMatrix, m_Rotation.y, vec3(0.0f, 1.0f, 0.0f));
    m_ModelMatrix = rotate(m_ModelMatrix, m_Rotation.z, vec3(0.0f, 0.0f, 1.0f));
    
    m_ModelMatrix = scale(m_ModelMatrix, m_Scale);
}

void Model::Delete() {
    if(m_isLoading.load()) return;

    // 找到当前对象在容器中的迭代器
    auto it = std::find(MODEL_POINTERS.begin(), MODEL_POINTERS.end(), this);

    if (it == MODEL_POINTERS.end()) return; // 如果找不到，则直接返回

    // 交换当前迭代器与末尾元素
    std::iter_swap(it, MODEL_POINTERS.end() - 1);

    // 从容器移除末尾指针
    MODEL_POINTERS.pop_back();

    if(DEBUG_INFO_LV > 0) std::cout << "[DELETER] 删除模型: " << this << " (" << this->GetID() << ")" << std::endl;

    delete this;
}

Model::~Model() {
    // 删除着色器
    if(DEBUG_INFO_LV > 1) std::cout << "  删除" << m_ID << "的着色器: " << ModelShader << std::endl;
    delete ModelShader;
    ModelShader = nullptr;
}
}   // namespace CubeDemo
