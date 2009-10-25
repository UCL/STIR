#
# $Id$
#
dir:=buildblock

$(dir)_LIB_SOURCES := \
  Array.cxx  \
  IndexRange.cxx \
  ProjData.cxx \
  ProjDataInfo.cxx \
  ProjDataInfoCylindrical.cxx \
  ProjDataInfoCylindricalArcCorr.cxx \
  ProjDataInfoCylindricalNoArcCorr.cxx \
  ArcCorrection.cxx \
  ProjDataFromStream.cxx \
  ProjDataGEAdvance.cxx \
  ProjDataInMemory.cxx \
  ProjDataInterfile.cxx \
  Scanner.cxx \
  SegmentBySinogram.cxx \
  Segment.cxx \
  SegmentByView.cxx \
  Viewgram.cxx \
  Sinogram.cxx \
  RelatedViewgrams.cxx \
  scale_sinograms.cxx \
  interpolate_projdata.cxx \
  extend_projdata.cxx \
  DiscretisedDensity.cxx \
  VoxelsOnCartesianGrid.cxx \
  utilities.cxx \
  interfile_keyword_functions.cxx \
  zoom.cxx \
  NumericType.cxx ByteOrder.cxx \
  KeyParser.cxx  \
  recon_array_functions.cxx \
  linear_regression.cxx overlap_interpolate.cxx \
  error.cxx warning.cxx  \
  DataSymmetriesForViewSegmentNumbers.cxx \
  TimeFrameDefinitions.cxx \
  ParsingObject.cxx \
	ArrayFilter1DUsingConvolutionSymmetricKernel.cxx \
	ArrayFilterUsingRealDFTWithPadding.cxx \
	SeparableArrayFunctionObject.cxx \
	SeparableMetzArrayFilter.cxx \
	MedianArrayFilter3D.cxx \
	MedianImageFilter3D.cxx \
	MinimalArrayFilter3D.cxx \
	MinimalImageFilter3D.cxx \
	SeparableCartesianMetzImageFilter.cxx \
	TruncateToCylindricalFOVImageProcessor.cxx \
	ThresholdMinToSmallPositiveValueDataProcessor.cxx \
	ChainedDataProcessor.cxx \
	ArrayFilter1DUsingConvolution.cxx \
	SeparableConvolutionImageFilter.cxx \
	NonseparableConvolutionUsingRealDFTImageFilter.cxx \
	SSRB.cxx \
	inverse_SSRB.cxx \
	centre_of_gravity.cxx \
	DynamicDiscretisedDensity.cxx \
	DynamicProjData.cxx \
	MultipleProjData.cxx \
	GatedProjData.cxx \
	ArrayFilter2DUsingConvolution.cxx \
	ArrayFilter3DUsingConvolution.cxx \
	find_fwhm_in_image.cxx

$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk


