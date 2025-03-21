cd ../
echo Compiling...
g++.exe -fexec-charset=utf-8 -g src/*.cpp src/core/*.cpp src/rendering/*.cpp src/ui/*.cpp -o ./bin/Demo.exe -I"./include" -I"D:/MSYS2/mingw64/include" -I"D:/MSYS2/mingw64/include/freetype2" -L"D:/MSYS2/mingw64/lib" -lglfw3 -lopengl32 -lgdi32 -lfreetype
echo CompileFinished.
pause
