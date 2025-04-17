
// include/resources/texture.h
#pragma once

// 标准库
#include <unordered_map>
#include "utils/stringsKits.h"
#include <memory>
#include <thread>


namespace CubeDemo {
// 乱七八糟的前置声明
class Texture;using TexPtrHashMap = std::unordered_map<string, std::weak_ptr<Texture>>; using TexturePtr = std::shared_ptr<Texture>;

// 声明Texture类
class Texture {
public:

    unsigned int ID;
    string Type;
    string Path;
    static std::atomic<size_t> s_TextureAliveCount;
    static TexPtrHashMap s_TexturePool;

    ~Texture();
    static TexturePtr Create(const string& path, const string& type);
    static TexturePtr CreateSync(const string& path, const string& type);
    void Bind(unsigned int slot = 0) const;

    // 删除拷贝构造函数和赋值运算符
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

private:
    Texture(const string& path, const string& type); // 构造函数私有化

};

}

