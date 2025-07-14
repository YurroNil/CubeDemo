// src/main/rendering.cpp
#include "pch.h"
#include "main/rendering.h"
#include "graphics/ray_tracing.h"

namespace CubeDemo {

extern bool RAY_TRACING_ENABLED, RT_DEBUG;
extern SceneMng* SCENE_MNG; extern LightMng* LIGHT_MNG;

/* <------------ 渲  染  循  环 ------------> */
void render_scene(GLFWwindow* window, Camera* camera)
{
    // 光线追踪路径
    if(RAY_TRACING_ENABLED && !RT_DEBUG) Renderer::s_RayTracing->Render(camera);
    // 光追调试路径
    else if(RT_DEBUG) Renderer::s_RayTracing->RenderDebug(camera);
    // 传统渲染路径
    else if(auto* scene = SCENE_MNG->GetCurrentScene()) scene->Render(window, camera, nullptr);
}

// 模型变换(如旋转)
void update_models() {}

}   // namespace CubeDemo
