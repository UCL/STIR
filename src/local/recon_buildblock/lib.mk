#
# $Id$
#

dir := local/recon_buildblock
$(dir)_LIB_SOURCES:= \
	ProjMatrixByBinUsingSolidAngle.cxx \
	ProjMatrixByBinUsingInterpolation.cxx \
	PostsmoothingForwardProjectorByBin.cxx \
	PresmoothingForwardProjectorByBin.cxx \
	PostsmoothingBackProjectorByBin.cxx \
	DataSymmetriesForDensels.cxx \
	DataSymmetriesForDensels_PET_CartesianGrid.cxx \
	ProjMatrixByDensel.cxx \
	ProjMatrixElemsForOneDensel.cxx \
	BackProjectorByBinUsingSquareProjMatrixByBin.cxx \
        ProjMatrixByDenselOnCartesianGridUsingElement.cxx \
	ProjMatrixByDenselUsingRayTracing.cxx \
	QuadraticPrior.cxx \
	BinNormalisationUsingProfile.cxx \
	BinNormalisationSinogramRescaling.cxx \
	ProjMatrixByBinSinglePhoton.cxx \
	ProjMatrixByBinFromFile.cxx
ifeq ($(HAVE_LLN_MATRIX),1)
$(dir)_LIB_SOURCES += \
	BinNormalisationFromECAT7.cxx
endif
  
#	oldForwardProjectorByBinUsingRayTracing.cxx \
#	oldForwardProjectorByBinUsingRayTracing_Siddon.cxx \
#	oldBackProjectorByBinUsingInterpolation.cxx \
#	oldBackProjectorByBinUsingInterpolation_linear.cxx \
#	oldBackProjectorByBinUsingInterpolation_piecewise_linear.cxx 



$(dir)_REGISTRY_SOURCES:= local_recon_buildblock_registries.cxx

include $(WORKSPACE)/lib.mk

