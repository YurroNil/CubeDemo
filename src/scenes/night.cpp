// include/scenes/night.cpp
#include "scenes/night.h"

namespace CubeDemo::Scenes {

// 初始化
void NightScene::Init(SceneMng& scene_inst, Light& light) {
    if(s_isInited) return;
    /* 未来再慢慢拓展 */
    s_isInited = true;
}

// 渲染
void NightScene::Render(GLFWwindow* window,
    Camera* camera,
    const Light& light,
    ShadowMap* shadow_map)
{
    /* 未来再慢慢拓展 */
}

// 清理
void NightScene::Cleanup(Light& light) {
    if(!s_isInited || s_isCleanup) return;

    light.Remove.All();
    
    s_isCleanup = true;
}

// 光束控制
void NightScene::UpdateBeam(float deltaTime) {
    /* 未来再慢慢拓展 */
}

NightScene::~NightScene() {}

}