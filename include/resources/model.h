// include/resources/model.h

#pragma once
#include "loaders/model.h"

namespace CubeDemo {

class Model : public Loaders::Model {
public:
    Model(const string& path);  // 初始化
    void NormalDraw(Shader& shader);
    void LodDraw(Shader& shader, const vec3& camera_pos);
    void DrawCall(bool mode, Shader& shader, const vec3& camera_pos);

    void DrawSimple() const;
};

}   // namespace CubeDemo
