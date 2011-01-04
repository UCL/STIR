#
# $Id$
#
set(dir utilities/ecat)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
   is_ecat7_file copy_ecat7_header 
	ifheaders_for_ecat7 conv_to_ecat7 print_ecat_singles_values 
	convecat6_if 
        conv_to_ecat6 
	ecat_swap_corners 
)


include(../../stir_exe_targets.cmake)
