// src/main.cpp

#include "main.h"

void buildPowershell(string& str) {
    cout << str << "\n\n\nCompiling..." << endl;
    // 运行命令
    system(str.c_str());

    cout << "Execute Finished." << endl;
    cin.get();
}


int main() {
    JsonParser::LoadFromJson("./maker_config.json");    // 读取与程序相同路径的json

    vector<fs::path> cpp_files;

    try { cpp_files = GetAllFiles(); }  // 获取所有文件
    catch (const fs::filesystem_error& e) {
        cerr << "目录访问错误：" << e.what() << endl;
        return 1;
    }

    if (!cpp_files.empty()) {
        string c{GeneratedFullCommands(cpp_files)}; // 生成完整的命令
        buildPowershell(c); // 然后使用powershell运行
    }
    else { cout << "未找到任何文件." << endl; }

    return 0;
}