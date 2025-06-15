// include/managers/model/cleanner.h
#pragma once
namespace CubeDemo::Managers {

class ModelCleanner {
public:
    static void Delete(Model** model);
    static void DeleteAll(std::vector<Model*> &models);
    static void DeleteShader(Shader** shader);
    static void DeleteAllShader(std::vector<Shader*> &shaders);
};
}
