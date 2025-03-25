#define CMD_OUTPUT " -o ./bin/Demo.exe"
#define CMD_INCLUDE {\
    "./include",\
    "D:/Personal-Development-Toolkit/include",\
    "D:/Personal-Development-Toolkit/mingw64/include",\
    "D:/Personal-Development-Toolkit/include/freetype2",\
    "D:/WindowsSDK/Include/10.0.26100.0"\
}
#define CMD_LIB {\
    "./lib",\
    "D:/Personal-Development-Toolkit/lib",\
    "D:/WindowsSDK/Lib/10.0.26100.0"\
}
#define CMD_LIB_SURFFIX {\
    "-lglfw3",\
    "-lopengl32",\
    "-lgdi32",\
    "-lfreetype"\
}
#define CMD_PATH {\
    "./src",\
    "./src/core",\
    "./src/rendering",\
    "./src/renderer",\
    "./src/ui"\
}
#define ARGS_COUNT 3

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>
namespace fs = std::filesystem;
using namespace std;

// 获取所有.cpp文件路径
vector<fs::path> GetAllFiles() {

    vector<fs::path> s;
    string folder_paths[] = CMD_PATH;

    // 遍历所有文件夹路径
    for (const auto& folder_path : folder_paths) {
        try {
            if (!fs::exists(folder_path)) {
                cerr << "警告：路径不存在 - " << folder_path << endl;
                continue;
            }
            for (const auto& entry : fs::directory_iterator(folder_path)) {
                if (entry.is_regular_file() &&
                    (entry.path().extension() == ".cpp" or entry.path().extension() == ".c")
                ) {
                    s.push_back(entry.path());
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "访问目录错误：" << folder_path 
                      << "\n错误信息：" << e.what() <<
            std::endl;
        }
    }
    return s;
}

// g++命令、include库和静态链接库链接的 字符串拼接部分
string LinkingSpliSection() {
    string include[] = CMD_INCLUDE, lib[] = CMD_LIB, lib_surffix[] = CMD_LIB_SURFFIX, buffer[ARGS_COUNT] = {"", "", " "};

    for(string s : include) { buffer[0] += (" -I\"" + s + "\""); }
    for(string s : lib) { buffer[1] += (" -L\"" + s + "\""); }
    for(string s : lib_surffix) { buffer[2] += (s + " "); }


    return buffer[0] + buffer[1] + buffer[2];
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

// 构建PowerShell
string GeneratedFullCommands(vector<fs::path> &cpp_files) {

    // GCC命令
    string prefix{"powershell -Command '"}, buffer{""}, command{""};
    string gcc_command[4] = {
        "g++.exe -fexec-charset=utf-8 -g ",
        PathSpliSection(cpp_files),
        CMD_OUTPUT,
        LinkingSpliSection()
    };
    // 字符串拼接
    for(string buffer : gcc_command) { command += buffer; }

    return command;
}

void buildPowershell(string str) {
    cout << str << "\n\n\nCompiling..." << endl;
    // 运行命令
    system(str.c_str());

    cout << "Execute Finished." << endl;
    cin.get();
}


int main() {
    vector<fs::path> cpp_files;

    try { cpp_files = GetAllFiles(); }
    catch (const fs::filesystem_error& e) {
        cerr << "目录访问错误：" << e.what() << endl;
        return 1;
    }

    if (!cpp_files.empty()) {
        string c{GeneratedFullCommands(cpp_files)};
        buildPowershell(c);
    }
    else { cout << "未找到任何文件." << endl; }

    return 0;
}