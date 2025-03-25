cd ../
echo Compiling...

g++.exe -fexec-charset=utf-8 -g src/*.cpp src/core/*.cpp src/rendering/*.cpp src/ui/*.cpp src/renderer/*.cpp -o ./bin/Demo.exe -I"./include" -I"D:/Personal-Development-Toolkit/include" -I"D:/mingw64/include" -I"D:/Personal-Development-Toolkit/include/freetype2" -I"D:/WindowsSDK/Include/10.0.26100.0/um" -L"./lib" -L"D:/Personal-Development-Toolkit/lib" -L"D/WindowsSDK/Lib/10.0.26100.0/um/x64" -lglfw3 -lopengl32 -lgdi32 -lfreetype

echo CompileFinished.
pause
