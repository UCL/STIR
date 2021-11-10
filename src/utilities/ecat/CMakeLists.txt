#
#
set(dir utilities/ecat)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
  is_ecat7_file.cxx
  copy_ecat7_header.cxx
  ifheaders_for_ecat7.cxx
  conv_to_ecat7 print_ecat_singles_values.cxx
  convecat6_if.cxx
  conv_to_ecat6.cxx
  ecat_swap_corners.cxx
)


include(stir_exe_targets)
