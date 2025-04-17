#
#
set(dir listmode_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
        ListModeData.cxx
        ListEvent.cxx
        CListEvent.cxx
        LmToProjDataAbstract.cxx
	LmToProjData.cxx
        LmToProjDataBootstrap.cxx
	LmToProjDataWithRandomRejection.cxx
        CListModeDataECAT8_32bit.cxx
        CListRecordECAT8_32bit.cxx
	CListModeDataSAFIR.cxx
)

if (HAVE_HDF5)
list(APPEND ${dir_LIB_SOURCES}
        CListModeDataGEHDF5.cxx
        # CListRecordGEHDF5.cxx
)
endif()

if (HAVE_ECAT)
  list(APPEND ${dir_LIB_SOURCES}
	CListModeDataECAT.cxx
	CListRecordECAT962.cxx
	CListRecordECAT966.cxx
  )
endif()

if (HAVE_UPENN)
	list(APPEND ${dir_LIB_SOURCES}
	CListModeDataPENN.cxx
	CListRecordPENN.cxx)
endif()

if (HAVE_CERN_ROOT)
	list(APPEND ${dir_LIB_SOURCES}
		CListModeDataROOT.cxx
		CListRecordROOT.cxx
	)
endif()

if (STIR_WITH_NiftyPET_PROJECTOR)
  list(APPEND ${dir_LIB_SOURCES}
    NiftyPET_listmode/LmToProjDataNiftyPET.cxx
  )
endif()


#$(dir)_REGISTRY_SOURCES:= $(dir)_registries


include(stir_lib_target)

target_link_libraries(listmode_buildblock PUBLIC data_buildblock PRIVATE fmt)

if (HAVE_HDF5)
  # for GEHDF5, TODO remove once IO dependency added or GEHDF5Wrapper no longer includes H5Cpp.h
  target_include_directories(listmode_buildblock PRIVATE ${HDF5_INCLUDE_DIRS})
endif()
