cd ../
g++.exe -fexec-charset=utf-8 -g src/*.cpp src/core/*.cpp src/rendering/*.cpp -o ./bin/Demo.exe -I"./include" -I"D:/_devKits/include" -L"D:/_devKits/lib" -lglfw3 -lopengl32 -lgdi32
