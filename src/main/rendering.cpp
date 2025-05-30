// src/main/rendering.cpp

#include "main/rendering.h"

namespace CubeDemo {

/* <------------ 渲  染  循  环 ------------> */
void render_scene(
    GLFWwindow* window,
    Camera* camera,
    Scene* scene_inst,
    const Light& light,
    ShadowMap* shadow_map)
{
    // 阴影渲染阶段
    shadow_map->RenderShadow(camera, light);

    // 主渲染阶段
    scene_inst->Rendering(scene_inst->Current, window, camera, light, shadow_map);
}

// 模型变换(如旋转)
void update_models() {}

}   // namespace CubeDemo
