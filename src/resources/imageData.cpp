// src/resources/imageData.cpp

#include "resources/imageData.h" 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "glad/glad.h"

namespace CubeDemo {

ImageDataPtr ImageData::Load(const string& path) {
    auto data = std::make_shared<ImageData>();

    stbi_set_flip_vertically_on_load(true);
    data->data.reset(stbi_load(path.c_str(), 
        &data->width, &data->height, &data->channels, 0));
    if(!data->data) {
        throw std::runtime_error(stbi_failure_reason());
    }
    return data;
}


}   // namespace CubeDemo