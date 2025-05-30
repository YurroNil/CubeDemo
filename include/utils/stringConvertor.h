// include/utils/stringConvert.h

#pragma once
#include "kits/strings.h"
#include <memory>

using wstring = std::wstring;

namespace CubeDemo {

class StringConvertor {
public:
    static wstring U8_to_Wstring(const string& str);
    static string WstringTo_U8(const wstring& wstr);
};
}
