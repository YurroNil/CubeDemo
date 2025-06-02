#pragma once

namespace std {
using TLS = CubeDemo::Texture::LoadState;

template<> 
struct formatter<TLS> : formatter<string_view> {
    auto format(TLS state, format_context& ctx) {
        string_view name = "Unknown";
        switch(state) {
            case TLS::Init: name = "Init"; break;
            case TLS::Placeholder: name = "Placeholder"; break;
            case TLS::Loading: name = "Loading"; break;
            case TLS::Ready: name = "Ready"; break;
            case TLS::Failed: name = "Failed"; break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};

}   // namespace std