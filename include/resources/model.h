// include/resources/model.h

#pragma once
#include "threads/modelLoader.h"

namespace CubeDemo {

class ModelLoader;
class Model : public ModelLoader {
public:
    Model(const string& path);  // 初始化
};

}   // namespace CubeDemo