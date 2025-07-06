// include/loaders/async_tex.h
#pragma once
#include "loaders/fwd.h"
#include "resources/texture.h"

namespace CubeDemo {

using TexPtrArray = std::vector<TexturePtr>;
// 乱七八糟的别名
using ImagePtr = std::shared_ptr<Loaders::Image>;
using TexLoadCallback = std::function<void(TexturePtr)>; // 纹理加载回调
using TL = Loaders::Texture;

class Loaders::AsyncTexture {
public:
    struct Context {
        TexPtrArray* output;
        std::mutex mutex;
        std::vector<TexturePtr> loadedTextures;
        std::atomic<int> pendingCount{0};
        std::promise<void> completionPromise;
    };

    static void OnTexLoaded(const std::shared_ptr<Context>& ctx, const string& path, TexturePtr tex);

private:
    static void HandleTexState(TexturePtr tex, const string& path);
};

}