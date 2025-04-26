// src/resources/model.cpp

#include "resources/model.h"

namespace CubeDemo {
    // Model构造函数传递路径到中间层ModelLoader类
    Model::Model(const string& path) : ModelLoader(path) {}


}   // namespace CubeDemo