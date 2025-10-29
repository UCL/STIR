#
#
set(dir examples)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	demo1.cxx demo2.cxx demo3.cxx demo4_obj_fun.cxx demo5_line_search.cxx
)


include(stir_exe_targets)
