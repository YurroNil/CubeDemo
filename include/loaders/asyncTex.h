// include/loaders/asyncTex.h

#include "loaders/texture.h"

namespace CubeDemo {
using TexPtrArray = std::vector<TexturePtr>;

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