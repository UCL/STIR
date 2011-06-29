#
# $Id$
#

dir := local/recon_buildblock
$(dir)_LIB_SOURCES:= \
	ProjMatrixByBinUsingSolidAngle.cxx \
	PostsmoothingForwardProjectorByBin.cxx \
	PresmoothingForwardProjectorByBin.cxx \
	PostsmoothingBackProjectorByBin.cxx \
	DataSymmetriesForDensels_PET_CartesianGrid.cxx \
	ProjMatrixByDensel.cxx \
        ProjMatrixByDenselOnCartesianGridUsingElement.cxx \
	ProjMatrixByDenselUsingRayTracing.cxx \
	BinNormalisationUsingProfile.cxx \
	BinNormalisationSinogramRescaling.cxx \
	ProjMatrixByBinSinglePhoton.cxx \
	ParametricQuadraticPrior.cxx 


#	PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData.cxx \
#	BackProjectorByBinDistanceDriven.cxx \
#	ForwardProjectorByBinDistanceDriven.cxx
#	oldForwardProjectorByBinUsingRayTracing.cxx \
#	oldForwardProjectorByBinUsingRayTracing_Siddon.cxx \
#	oldBackProjectorByBinUsingInterpolation.cxx \
#	oldBackProjectorByBinUsingInterpolation_linear.cxx \
#	oldBackProjectorByBinUsingInterpolation_piecewise_linear.cxx 



$(dir)_REGISTRY_SOURCES:= local_recon_buildblock_registries.cxx

include $(WORKSPACE)/lib.mk

