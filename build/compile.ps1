cd ../
g++.exe -fexec-charset=gbk -g src/*.cpp src/core/*.cpp src/rendering/*.cpp -o ./bin/Demo.exe -I"./include" -I"D:/_devKits/include" -L"D:/_devKits/lib" -lglfw3 -lopengl32 -lgdi32
