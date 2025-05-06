// include/resources/model.h

#pragma once
#include "threads/loaders.h"
#include "loaders/model.h"

namespace CubeDemo {

class Model : public Loaders::Model {
public:
    Model(const string& path);  // 初始化
};

}   // namespace CubeDemo