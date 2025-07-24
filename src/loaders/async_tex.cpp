// include/loaders/async_tex.cpp
#include "pch.h"
#include "loaders/async_tex.h"

using ATL = CubeDemo::Loaders::AsyncTexture;
using TLS = CubeDemo::Texture::LoadState;
namespace CubeDemo {

extern unsigned int DEBUG_INFO_LV;

void ATL::OnTexLoaded(const std::shared_ptr<Context>& ctx, const string& path, TexturePtr tex) {

    std::lock_guard<std::mutex> lock(ctx->mutex);
    HandleTexState(tex, path); // 提取状态处理逻辑
    ctx->loadedTextures.push_back(tex);

    if (ctx->pendingCount.fetch_sub(1) == 1) {
        *ctx->output = ctx->loadedTextures;
        ctx->completionPromise.set_value();
        std::cout << "[TEXTURE] 全部纹理加载完成\n";
    }
}

void ATL::HandleTexState(TexturePtr tex, const string& path) {
    if (!tex) return;
    switch(tex->State.load()) {
        case TLS::Ready:
            std::cout << "[Success] 加载完成: " << path << std::endl; break;
        case TLS::Placeholder:
            std::cout << "[Warning] 使用占位纹理: " << path << std::endl; break;
        case TLS::Failed:
            std::cerr << "[Error] 最终加载失败: " << path << std::endl; break;

        default: case TLS::Uninited: std::cout << "[Warning] 状态异常, 未能成功初始化: " << path << std::endl; break;
    }
}
}   // namespace CubeDemo
