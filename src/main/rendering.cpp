// src/main/rendering.cpp
#include "pch.h"
#include "main/rendering.h"

namespace CubeDemo {

extern SceneMng* SCENE_MNG;
extern LightMng* LIGHT_MNG;

/* <------------ 渲  染  循  环 ------------> */
void render_scene(
    GLFWwindow* window,
    Camera* camera,
    ShadowMap* shadow_map)
{

    // 阴影渲染阶段(暂不使用)
    // shadow_map->RenderShadow(camera);

    // 主渲染阶段
    if (auto* scene = SCENE_MNG->GetCurrentScene()) {
        scene->Render(window, camera, shadow_map);
    }
}

// 模型变换(如旋转)
void update_models() {}

}   // namespace CubeDemo
