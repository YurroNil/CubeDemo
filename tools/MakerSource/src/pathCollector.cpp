// src/pathCollector.cpp

#include "pathCollector.h"

void CopyPath(vector<fs::path> &s, fs::directory_entry entry) {

    if ( entry.is_regular_file() &&
        ( entry.path().extension() == ".cpp" || entry.path().extension() == ".c" )
    ) {
        s.push_back(entry.path());
    }
}

// 获取所有.cpp文件路径
vector<fs::path> GetAllFiles() {

    vector<fs::path> s;
    vector<string> folder_paths = JsonParser::s_PathsSourceCode;

    // 遍历所有文件夹路径
    for (const auto& folder_path : folder_paths) {
        try {
            if (!fs::exists(folder_path)) {
                cerr << "警告：路径不存在 - " << folder_path << endl;
                continue;
            }
            for (const auto& entry : fs::directory_iterator(folder_path))
            {
                CopyPath(s, entry);
            }
        }
        catch (const fs::filesystem_error& e) {
            std::cerr << "访问目录错误：" << folder_path << "\n错误信息：" << e.what() << std::endl;
        }
    }
    return s;
}


// 字符串拼接 (文件路径拼接部分)
string PathSpliSection(vector<fs::path> &cpp_files) {
    string str = "";
    for (size_t i = 0; i < cpp_files.size(); ++i) {

        str += "\"" + cpp_files[i].string() + "\"";
        if (i != cpp_files.size() - 1) { str += " "; }
    }
    return str;
}
