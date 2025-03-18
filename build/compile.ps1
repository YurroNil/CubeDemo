cd ../
g++.exe -fexec-charset=gbk -g src/*.cpp src/core/*.cpp src/rendering/*.cpp -o ./bin/Demo.exe -I"./include" -I"./include/3rd-lib" -L"./lib" -lglfw3 -lopengl32 -lgdi32
