set(dir buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  find_STIR_config.cxx
  error.cxx
  warning.cxx
  TextWriter.cxx
  Verbosity.cxx
  NumericType.cxx
  ByteOrder.cxx
  KeyParser.cxx
  interfile_keyword_functions.cxx
  ParsingObject.cxx
  num_threads.cxx
  Array.cxx
  IndexRange.cxx
  PatientPosition.cxx
  TimeFrameDefinitions.cxx
  ExamInfo.cxx
  ExamData.cxx
  RadionuclideDB.cxx
  Radionuclide.cxx
  ProjData.cxx
  ProjDataInfo.cxx
  ProjDataInfoCylindrical.cxx
  ProjDataInfoCylindricalArcCorr.cxx
  ProjDataInfoCylindricalNoArcCorr.cxx
  ProjDataInfoSubsetByView.cxx
  ProjDataInfoBlocksOnCylindricalNoArcCorr.cxx
  ProjDataInfoGeneric.cxx
  ProjDataInfoGenericNoArcCorr.cxx
  DetectorCoordinateMap.cxx
  GeometryBlocksOnCylindrical.cxx
  DiscretisedDensity.cxx
  VoxelsOnCartesianGrid.cxx
  DynamicDiscretisedDensity.cxx
  ProjDataFromStream.cxx
  ProjDataInMemory.cxx
  ProjDataInterfile.cxx
  Scanner.cxx
  SegmentBySinogram.cxx
  Segment.cxx
  SegmentByView.cxx
  Viewgram.cxx
  Sinogram.cxx
  RelatedViewgrams.cxx
  zoom.cxx
  DataSymmetriesForViewSegmentNumbers.cxx
  recon_array_functions.cxx
  utilities.cxx
  date_time_functions.cxx
  )

if (NOT MINI_STIR)
  list(APPEND ${dir_LIB_SOURCES}
    scale_sinograms.cxx
    interpolate_projdata.cxx
    extend_projdata.cxx
    multiply_crystal_factors.cxx
    ParseDiscretisedDensityParameters.cxx
    linear_regression.cxx
    overlap_interpolate.cxx
    ArrayFilter1DUsingConvolutionSymmetricKernel.cxx
    ArrayFilterUsingRealDFTWithPadding.cxx
    SeparableArrayFunctionObject.cxx
    SeparableMetzArrayFilter.cxx
    SeparableGaussianImageFilter.cxx
    SeparableGaussianArrayFilter.cxx
    MedianArrayFilter3D.cxx
    MedianImageFilter3D.cxx
    WienerArrayFilter2D.cxx
    WienerImageFilter2D.cxx
    GammaArrayFilter2D.cxx
    GammaImageFilter2D.cxx
    MinimalArrayFilter3D.cxx
    MinimalImageFilter3D.cxx
    SeparableCartesianMetzImageFilter.cxx
    TruncateToCylindricalFOVImageProcessor.cxx
    ThresholdMinToSmallPositiveValueDataProcessor.cxx
    HUToMuImageProcessor.cxx
    ChainedDataProcessor.cxx
    ArrayFilter1DUsingConvolution.cxx
    ArrayFilter2DUsingConvolution.cxx
    ArrayFilter3DUsingConvolution.cxx
    MaximalArrayFilter3D.cxx
    MaximalImageFilter3D.cxx
    FilePath.cxx
    SeparableConvolutionImageFilter.cxx
    NonseparableConvolutionUsingRealDFTImageFilter.cxx
    ArcCorrection.cxx
    SSRB.cxx
    inverse_SSRB.cxx
    centre_of_gravity.cxx
    DynamicProjData.cxx
    MultipleProjData.cxx
    MultipleDataSetHeader.cxx
    GatedProjData.cxx
    find_fwhm_in_image.cxx
    GatedDiscretisedDensity.cxx
    TimeGateDefinitions.cxx
    ML_norm.cxx
    GeneralisedPoissonNoiseGenerator.cxx
    )
  if (HAVE_HDF5)
    list(APPEND ${dir_LIB_SOURCES}
      ProjDataGEHDF5.cxx
      )
endif()

endif() # MINI_STIR

if (NOT HAVE_SYSTEM_GETOPT)
  # add our own version of getopt to buildblock
  list(APPEND ${dir_LIB_SOURCES} getopt.c)
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
  target_include_directories(buildblock PUBLIC "${TMP}")
endif()

target_link_libraries(buildblock PRIVATE fmt)

# TODO Remove but currently needed for ProjData.cxx, DynamicDisc*cxx, TimeFrameDef
if (LLN_FOUND)
  target_link_libraries(buildblock PUBLIC ${LLN_LIBRARIES})
endif()

if (HAVE_HDF5)
  # for GEHDF5, TODO remove once IO dependency added or GEHDF5Wrapper no longer includes H5Cpp.h
  target_include_directories(buildblock PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

# TODO currently needed as filters need fourier
#target_link_libraries(buildblock PUBLIC numerics_buildblock)

if (STIR_OPENMP)
  target_link_libraries(buildblock PUBLIC ${OpenMP_EXE_LINKER_FLAGS})
endif()
