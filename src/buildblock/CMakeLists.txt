set(dir buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  Array  
  IndexRange 
  PatientPosition
  ProjData 
  ProjDataInfo 
  ProjDataInfoCylindrical 
  ProjDataInfoCylindricalArcCorr 
  ProjDataInfoCylindricalNoArcCorr 
  ArcCorrection 
  ProjDataFromHDF5
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
  utilities 
  interfile_keyword_functions 
  zoom 
  NumericType ByteOrder 
  KeyParser  
  recon_array_functions 
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
)

if (NOT HAVE_SYSTEM_GETOPT)
  # add our own version of getopt to buildblock
  list(APPEND ${dir_LIB_SOURCES} getopt)
endif()

include(stir_lib_target)

# TODO Remove but currently needed for ProjData.cxx, DynamicDisc*cxx, TimeFrameDef
if (LLN_FOUND)
  target_link_libraries(buildblock ${LLN_LIBRARIES})
endif()

if (HDF5_FOUND)
 list(APPEND ${dir_LIB_SOURCES}
    ProjDataFromHDF5
 )
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
