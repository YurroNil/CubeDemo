
// include/resources/texture.h
#pragma once
#include <unordered_map>
#include "utils/stringsKits.h"
#include <memory>

namespace CubeDemo {

class Texture {
public:
    static std::shared_ptr<Texture> Create(const string& path, const string& type);
    
    unsigned int ID;
    string Type;
    string Path;

    void Bind(unsigned int slot = 0) const;

    // 删除拷贝构造函数和赋值运算符
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

private:
    Texture(const string& path, const string& type); // 构造函数私有化
    static std::unordered_map<string, std::weak_ptr<Texture>> s_TexturePool; // 使用weak_ptr防止内存泄漏
};

}

