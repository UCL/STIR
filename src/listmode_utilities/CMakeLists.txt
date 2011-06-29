#
# $Id$
#
set(dir listmode_utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
  lm_to_projdata
  lm_to_projdata_bootstrap
  lm_fansums
  list_lm_events
)

if (HAVE_ECAT)
  # yes, the LLN files seem to be there, so we can compile more
  list(APPEND $(dir)_SOURCES
	scan_sgl_file 
	print_sgl_values 
	rebin_sgl_file 
	add_ecat7_header_to_sgl
  )
endif()



include(../stir_exe_targets.cmake)
