set(dir buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  Array.cxx
  IndexRange.cxx
  PatientPosition.cxx
  ExamInfo.cxx
  ExamData.cxx
  RadionuclideDB.cxx
  Radionuclide.cxx
  find_STIR_config.cxx
  ProjData.cxx
  ProjDataInfo.cxx
  ProjDataInfoCylindrical.cxx
  ProjDataInfoCylindricalArcCorr.cxx
  ProjDataInfoCylindricalNoArcCorr.cxx
  ProjDataInfoSubsetByView.cxx
  ArcCorrection.cxx
  ProjDataFromStream.cxx
  ProjDataGEAdvance.cxx
  ProjDataInMemory.cxx
  ProjDataInterfile.cxx
  Scanner.cxx
  SegmentBySinogram.cxx
  Segment.cxx
  SegmentByView.cxx
  Viewgram.cxx
  Verbosity.cxx
  Sinogram.cxx
  RelatedViewgrams.cxx
  scale_sinograms.cxx
  interpolate_axial_position.cxx
  interpolate_projdata.cxx
  extend_projdata.cxx
  multiply_crystal_factors.cxx
  DiscretisedDensity.cxx
  VoxelsOnCartesianGrid.cxx
  ParseDiscretisedDensityParameters.cxx
  utilities.cxx
  interfile_keyword_functions.cxx
  zoom.cxx
  NumericType.cxx
  ByteOrder.cxx
  KeyParser.cxx
  recon_array_functions.cxx
  linear_regression.cxx
  overlap_interpolate.cxx
  error.cxx
  warning.cxx
  TextWriter.cxx
  DataSymmetriesForViewSegmentNumbers.cxx
  TimeFrameDefinitions.cxx
  ParsingObject.cxx
	ArrayFilter1DUsingConvolutionSymmetricKernel.cxx
	ArrayFilterUsingRealDFTWithPadding.cxx
	SeparableArrayFunctionObject.cxx
	SeparableMetzArrayFilter.cxx
        SeparableGaussianImageFilter.cxx
        SeparableGaussianArrayFilter.cxx
	MedianArrayFilter3D.cxx
	MedianImageFilter3D.cxx
	MinimalArrayFilter3D.cxx
	MinimalImageFilter3D.cxx
	SeparableCartesianMetzImageFilter.cxx
	TruncateToCylindricalFOVImageProcessor.cxx
	ThresholdMinToSmallPositiveValueDataProcessor.cxx
	ChainedDataProcessor.cxx
	ArrayFilter1DUsingConvolution.cxx
	SeparableConvolutionImageFilter.cxx
	NonseparableConvolutionUsingRealDFTImageFilter.cxx
	SSRB.cxx
	inverse_SSRB.cxx
	centre_of_gravity.cxx
	DynamicDiscretisedDensity.cxx
	DynamicProjData.cxx
	MultipleProjData.cxx
  MultipleDataSetHeader.cxx
	GatedProjData.cxx
	ArrayFilter2DUsingConvolution.cxx
	ArrayFilter3DUsingConvolution.cxx
	find_fwhm_in_image.cxx
        GatedDiscretisedDensity.cxx
        MaximalArrayFilter3D.cxx
        MaximalImageFilter3D.cxx
        TimeGateDefinitions.cxx
	ML_norm.cxx
        num_threads.cxx
        GeneralisedPoissonNoiseGenerator.cxx
        FilePath.cxx
        date_time_functions.cxx
        DetectorCoordinateMap.cxx
       GeometryBlocksOnCylindrical.cxx
        ProjDataInfoBlocksOnCylindricalNoArcCorr.cxx
        ProjDataInfoGeneric.cxx
        ProjDataInfoGenericNoArcCorr.cxx
        date_time_functions.cxx
)
if (HAVE_HDF5)
 list(APPEND ${dir_LIB_SOURCES}
    ProjDataGEHDF5.cxx
    )
endif()

if (NOT HAVE_SYSTEM_GETOPT)
  # add our own version of getopt to buildblock
  list(APPEND ${dir_LIB_SOURCES} getopt.c)
endif()

#if (HAVE_JSON)
  list(APPEND ${dir_LIB_SOURCES} HUToMuImageProcessor.cxx)
#endif()

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
  target_include_directories(buildblock PUBLIC "${TMP}")
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
