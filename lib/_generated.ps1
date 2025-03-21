cd ../

gcc -I"D:/MSYS2/mingw64/include" -I"D:/MSYS2/mingw64/include/freetype2" -Iinclude -L"D:/MSYS2/mingw64/lib" -Llib -lglfw3 -lopengl32 -lgdi32 -lfreetype -c src/include.c -o ./lib/include.o

ar -rcs ./lib/libCubeDemo.a ./lib/*.o
cd ./lib
del -Recurse -Force *.o
echo Finished.
pause
