#
# $Id$
#
set(dir IO)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  OutputFileFormat 
  OutputFileFormat_default 
  InterfileOutputFileFormat 
  interfile InterfileHeader 
  InputFileFormatRegistry 
  InterfileDynamicDiscretisedDensityOutputFileFormat 
  InterfileParametricDensityOutputFileFormat
)
if (LLN_FOUND)
 list(APPEND $(dir)_LIB_SOURCES
    ECAT7OutputFileFormat stir_ecat7 
    ECAT6OutputFileFormat 
    stir_ecat_common 
    ECAT7ParametricDensityOutputFileFormat 
    ECAT7DynamicDiscretisedDensityOutputFileFormat 
    stir_ecat6 ecat6_utils
 )
endif()

if(AVW_FOUND)
  list(APPEND $(dir)_LIB_SOURCES 
    stir_AVW
  )
endif()

add_library(${dir} ${${dir_LIB_SOURCES}}    )

if (LLN_FOUND)
  target_link_libraries(IO ${LLN_LIBRARIES})
endif()

if (AVW_FOUND)
  target_link_libraries(IO ${AVW_LIBRARIES})
endif()
