#
# $Id$
#
dir:=buildblock

$(dir)_LIB_SOURCES := \
  Array.cxx  convert_array.cxx \
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
  SegmentByView.cxx \
  Viewgram.cxx \
  Sinogram.cxx \
  RelatedViewgrams.cxx \
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
	SeparableCartesianMetzImageFilter.cxx \
	ThresholdMinToSmallPositiveValueImageProcessor.cxx \
	ChainedImageProcessor.cxx \
	ArrayFilter1DUsingConvolution.cxx \
	SeparableConvolutionImageFilter.cxx \
	SSRB.cxx \
	centre_of_gravity.cxx
 

$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk


