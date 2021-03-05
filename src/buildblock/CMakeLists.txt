set(dir buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  Array  
  IndexRange 
  PatientPosition
  ExamInfo
  ExamData
  findSTIRConfig
  ProjData 
  ProjDataInfo 
  ProjDataInfoCylindrical 
  ProjDataInfoCylindricalArcCorr 
  ProjDataInfoCylindricalNoArcCorr 
  ArcCorrection 
  ProjDataFromStream  
  ProjDataGEAdvance
  ProjDataInMemory 
  ProjDataInterfile 
  Scanner 
  SegmentBySinogram 
  Segment 
  SegmentByView 
  Viewgram
  Verbosity
  Sinogram 
  RelatedViewgrams 
  scale_sinograms 
  interpolate_projdata 
  extend_projdata 
  DiscretisedDensity 
  VoxelsOnCartesianGrid 
  ParseDiscretisedDensityParameters
  utilities 
  interfile_keyword_functions 
  zoom 
  NumericType ByteOrder 
  KeyParser  
  recon_array_functions 
  RadionuclideDBProcessor
  linear_regression overlap_interpolate 
  error warning
  TextWriter
  DataSymmetriesForViewSegmentNumbers 
  TimeFrameDefinitions 
  ParsingObject 
	ArrayFilter1DUsingConvolutionSymmetricKernel 
	ArrayFilterUsingRealDFTWithPadding 
	SeparableArrayFunctionObject 
	SeparableMetzArrayFilter 
        SeparableGaussianImageFilter
        SeparableGaussianArrayFilter
	MedianArrayFilter3D 
	MedianImageFilter3D 
	MinimalArrayFilter3D 
	MinimalImageFilter3D 
	SeparableCartesianMetzImageFilter 
	TruncateToCylindricalFOVImageProcessor 
	ThresholdMinToSmallPositiveValueDataProcessor 
	ChainedDataProcessor 
	ArrayFilter1DUsingConvolution 
	SeparableConvolutionImageFilter 
	NonseparableConvolutionUsingRealDFTImageFilter 
	SSRB 
	inverse_SSRB 
	centre_of_gravity 
	DynamicDiscretisedDensity 
	DynamicProjData 
	MultipleProjData 
  MultipleDataSetHeader
	GatedProjData 
	ArrayFilter2DUsingConvolution 
	ArrayFilter3DUsingConvolution 
	find_fwhm_in_image
        GatedDiscretisedDensity
        MaximalArrayFilter3D
        MaximalImageFilter3D
        TimeGateDefinitions
	ML_norm
        num_threads
        GeneralisedPoissonNoiseGenerator
        FilePath
        date_time_functions
)
if (HAVE_HDF5)
 list(APPEND ${dir_LIB_SOURCES}
    ProjDataGEHDF5
    )
endif()

if (NOT HAVE_SYSTEM_GETOPT)
  # add our own version of getopt to buildblock
  list(APPEND ${dir_LIB_SOURCES} getopt)
endif()

if (HAVE_JSON)
  list(APPEND ${dir_LIB_SOURCES} HUToMuImageProcessor)
endif()

include(stir_lib_target)


if (HAVE_JSON)
  # Add the header-only nlohman_json header-only library
  # Unfortunately, the simple line below exports the dependency while this is really not
  # necessary.
  #
  # target_link_libraries(buildblock PRIVATE "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>")

  # So, we currently use an ugly work-around from
  # https://gitlab.kitware.com/cmake/cmake/-/issues/15415#note_334852
  # Warning: this will fail once nlohman_json stops being header-only!
  # In that case, we will need to add it in STIRConfig.cmake.in
  #
  get_target_property(TMP nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(buildblock PRIVATE "${TMP}")
endif()

# TODO Remove but currently needed for ProjData.cxx, DynamicDisc*cxx, TimeFrameDef
if (LLN_FOUND)
  target_link_libraries(buildblock ${LLN_LIBRARIES})
endif()

if (RDF_FOUND)
  # TODO cannot do this as it creates circular dependencies
  # target_link_libraries(buildblock local_IO_GE)
endif()

# TODO currently needed as filters need fourier
#target_link_libraries(buildblock numerics_buildblock)

if (STIR_OPENMP)
  target_link_libraries(buildblock ${OpenMP_EXE_LINKER_FLAGS})
endif()
