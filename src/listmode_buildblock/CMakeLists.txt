#
#
set(dir listmode_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
        ListModeData
        ListEvent
        CListEvent
        LmToProjDataAbstract
	LmToProjData 
        LmToProjDataBootstrap
	LmToProjDataWithRandomRejection
        CListModeDataECAT8_32bit
        CListRecordECAT8_32bit
	CListModeDataSAFIR
	DetectorCoordinateMapFromFile
)

if (HAVE_ECAT)
  list(APPEND ${dir_LIB_SOURCES}
	CListModeDataECAT 
	CListRecordECAT962 
	CListRecordECAT966 
  )
endif()

if (HAVE_CERN_ROOT)
	list(APPEND ${dir_LIB_SOURCES}
		CListModeDataROOT
		CListRecordROOT
	)
endif()

if (STIR_WITH_NiftyPET_PROJECTOR)
  list(APPEND ${dir_LIB_SOURCES}
    NiftyPET_listmode/LmToProjDataNiftyPET
  )
endif()


#$(dir)_REGISTRY_SOURCES:= $(dir)_registries


include(stir_lib_target)

target_link_libraries(listmode_buildblock IO data_buildblock )
