Credit: https://www.kdab.com/clang-tidy-part-1-modernize-source-code-using-c11c14/ and Max Ahnen (Positrigo).
See https://github.com/UCL/STIR/issues/827

Note that using the CMake settings did no twork for KT, possibly because not using clang then.

0. Best to make a build using clang. I had to disable OpenMP and therefore parallelproj
   ```
   CC=clang CXX=clang++ ccmake -DCMAKE_PREFIX_PATH=~/devel/build/SIRF-SuperBuild/INSTALL/ -DDISABLE_parallelproj=ON -B build -S sources/STIR
   ```
1. run CMake to output `compile_commands.json` in the build folder.
   ```
   cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   ```

2. Run `clang-tidy` like this:
   ```
   run-clang-tidy -p build/ -header-filter=.* -checks='-*,modernize-use-override' -fix
   ```

3. check and commit
   ```
   git commit -a --author="stir_maintenance <noreply@github.com>"
   ```
