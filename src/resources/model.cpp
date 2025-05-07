// src/resources/model.cpp

#include "resources/model.h"

namespace CubeDemo {
    // Model构造函数传递路径到Loaders:Model类
    Model::Model(const string& path) : Loaders::Model(path) {}
}   // namespace CubeDemo