#
# $Id$
#

dir := local/buildblock
$(dir)_LIB_SOURCES:= \
	fft.cxx \
	ML_norm.cxx \
	multiply_plane_scale_factorsImageProcessor.cxx \
	Quaternion.cxx \
	cleanup966ImageProcessor.cxx \
        AbsTimeIntervalWithParsing.cxx \
        AbsTimeIntervalFromDynamicData.cxx

ifeq ($(HAVE_LLN_MATRIX),1)
$(dir)_LIB_SOURCES += \
        AbsTimeIntervalFromECAT7ACF.cxx
endif

currently_disabled:=	DAVArrayFilter3D.cxx \
	DAVImageFilter3D.cxx \
	ModifiedInverseAveragingImageFilterAll.cxx \
	fwd_and_bck_manipulation_for_SAF.cxx \
	SeparableLowPassArrayFilter.cxx \
	SeparableLowPassImageFilter.cxx \
	ModifiedInverseAverigingArrayFilter.cxx \
	ModifiedInverseAverigingImageFilter.cxx \
	SeparableGaussianImageFilter.cxx \
	SeparableGaussianArrayFilter.cxx \
	NonseparableSpatiallyVaryingFilters.cxx \
	NonseparableSpatiallyVaryingFilters3D.cxx \
	local_helping_functions.cxx 

$(dir)_REGISTRY_SOURCES:= local_buildblock_registries.cxx

include $(WORKSPACE)/lib.mk

