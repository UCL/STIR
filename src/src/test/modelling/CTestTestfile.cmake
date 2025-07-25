# CMake generated Testfile for 
# Source directory: /root/devel/buildConda/sources/STIR/src/test/modelling
# Build directory: /root/devel/buildConda/sources/STIR/src/src/test/modelling
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_modelling "/root/devel/buildConda/sources/STIR/src/src/test/modelling/test_modelling" "/root/devel/buildConda/sources/STIR/src/test/modelling/input")
set_tests_properties(test_modelling PROPERTIES  _BACKTRACE_TRIPLES "/root/devel/buildConda/sources/STIR/src/test/modelling/CMakeLists.txt;25;ADD_TEST;/root/devel/buildConda/sources/STIR/src/test/modelling/CMakeLists.txt;0;")
add_test(test_ParametricDiscretisedDensity "/root/devel/buildConda/sources/STIR/src/src/test/modelling/test_ParametricDiscretisedDensity")
set_tests_properties(test_ParametricDiscretisedDensity PROPERTIES  ENVIRONMENT "STIR_CONFIG_DIR=/root/devel/buildConda/sources/STIR/src/config" _BACKTRACE_TRIPLES "/root/devel/buildConda/sources/STIR/src/cmake/stir_test_exe_targets.cmake;73;ADD_TEST;/root/devel/buildConda/sources/STIR/src/cmake/stir_test_exe_targets.cmake;95;create_stir_test;/root/devel/buildConda/sources/STIR/src/cmake/stir_test_exe_targets.cmake;0;;/root/devel/buildConda/sources/STIR/src/test/modelling/CMakeLists.txt;29;include;/root/devel/buildConda/sources/STIR/src/test/modelling/CMakeLists.txt;0;")
