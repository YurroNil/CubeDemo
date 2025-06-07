// include/utils/string_conv.h
#pragma once

using wstring = std::wstring;

namespace CubeDemo {

class StringConvertor {
public:
    static wstring U8_to_Wstring(const string& str);
    static string WstringTo_U8(const wstring& wstr);
};
}
