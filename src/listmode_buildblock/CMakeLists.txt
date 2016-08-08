#
#
set(dir listmode_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	CListEvent 
	CListModeData 
	LmToProjData 
        LmToProjDataBootstrap
        CListModeDataECAT8_32bit
        CListRecordECAT8_32bit
        CListModeDataGESigna
#        CListRecordGESigna
)

if (HAVE_ECAT)
  list(APPEND ${dir_LIB_SOURCES}
	CListModeDataECAT 
	CListRecordECAT962 
	CListRecordECAT966 
  )
endif()

#$(dir)_REGISTRY_SOURCES:= $(dir)_registries


include(stir_lib_target)

target_link_libraries(listmode_buildblock IO data_buildblock )
