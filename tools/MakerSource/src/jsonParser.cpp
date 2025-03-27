// src/jsonParser.cpp

#include "jsonParser.h"


// 静态成员初始化
    vector<string> JsonParser::s_PathsInclude = {};
    vector<string> JsonParser::s_PathsLib = {};
    vector<string> JsonParser::s_PathsSourceCode = {};
    vector<string> JsonParser::s_argsLibSuffix = {};

    string JsonParser::s_argsOutput = "";
    int JsonParser::s_argsCount = 3;



// 验证json配置文件是否有缺失的键值
void ValidateJson(const n_json& j) {

    if (!j.contains("paths") || !j["paths"].is_object()) {
        throw runtime_error("JSON无效：缺少paths部分");
    }
    
    if (!j.contains("arguments") || !j["arguments"].is_object()) {
        throw runtime_error("JSON无效：缺少arguments部分");
    }
    
    if (!j.contains("args_count") || !j["args_count"].is_number_integer()) {
        throw runtime_error("JSON无效：缺少args_count的值");
    }
    
    // 验证paths结构
    const auto& paths = j["paths"];
    if (!paths.contains("include") || !paths["include"].is_array()) {
        throw runtime_error("JSON无效：缺少include路径");
    }
    
    if (!paths.contains("lib") || !paths["lib"].is_array()) {
        throw runtime_error("JSON无效：缺少lib路径");
    }
    if (!paths.contains("source_code") || !paths["source_code"].is_array()) {
        throw runtime_error("JSON无效：缺少source_code路径");
    }
    
    // 验证arguments结构
    const auto& arguments = j["arguments"];
    if (!arguments.contains("output") || !arguments["output"].is_string()) {
        throw runtime_error("JSON无效：缺少output参数");
    }
    
    if (!arguments.contains("lib_suffix") || !arguments["lib_suffix"].is_array()) {
        throw runtime_error("JSON无效：缺少lib_suffix参数");
    }
}

// 解析元数据 
void JsonParser::Parsing(n_json& data) {

    // paths
    const auto& paths = data["paths"];
    s_PathsInclude = paths["include"].get<vector<string>>();

    s_PathsLib = paths["lib"].get<vector<string>>();
    s_PathsSourceCode = paths["source_code"].get<vector<string>>();

    // arguments
    s_argsOutput = data["arguments"].value("output", " -o Program.exe");
    const auto& arguments = data["arguments"];
    s_argsLibSuffix = arguments["lib_suffix"].get<vector<string>>();

    // args_count
    s_argsCount = data["args_count"].get<int>();
}

// 读取json路径并导入数据
void JsonParser::LoadFromJson(const string& filePath) {
    n_json data;

    // 验证是否能打开文件
    ifstream file(filePath);
    if (!file.is_open()) { throw runtime_error("打开配置文件失败：" + filePath); }
    // 验证JSON结构
    try { file >> data;  ValidateJson(data);}
    catch (const n_json::exception& e) { throw runtime_error("JSON粘贴错误：" + string(e.what())); }

    //解析JSON元数据
    Parsing(data);

}
