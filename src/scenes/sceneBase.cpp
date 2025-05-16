#include "scenes/sceneBase.h"
#include "utils/defines.h"

namespace CubeDemo::Scenes {

Shader* SceneBase::Creater::ShadowShader() {
    // 阴影深度着色器
    Shader* shadow_depth = new Shader(
        VSH_PATH + string("shadow_depth.glsl"),
        FSH_PATH + string("shadow_depth.glsl")
    );
    return shadow_depth;
}

GSM* SceneBase::Creater::ShadowMap() {
    // 静态保持阴影贴图
    GSM* shadow_map = new GSM(2048, 2048);
    return shadow_map;
}

DL* SceneBase::Creater::DirLight() {
    // 创建太阳
    DL* sun = new DL {
        .direction = vec3(-0.5f, -1.0f, -0.3f),
        .ambient = vec3(0.2f),
        .diffuse = vec3(0.5f),
        .specular = vec3(0.4f)
    };
    return sun;
}

}