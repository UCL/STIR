#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common


all: $(DEST)librecon_buildblock.a $(DEST)local_recon_buildblock_registries.o

clean:
	$(RM) $(DEST)*
SOURCES = \
	ProjMatrixByBinUsingSolidAngle.cxx \
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
	ProjMatrixByBinSinglePhoton.cxx 
ifeq ($(HAVE_LLN_MATRIX),1)
SOURCES += \
	BinNormalisationFromECAT7.cxx
endif
  
#	oldForwardProjectorByBinUsingRayTracing.cxx \
#	oldForwardProjectorByBinUsingRayTracing_Siddon.cxx \
#	oldBackProjectorByBinUsingInterpolation.cxx \
#	oldBackProjectorByBinUsingInterpolation_linear.cxx \
#	oldBackProjectorByBinUsingInterpolation_piecewise_linear.cxx 

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
  include $(DEST)local_recon_buildblock_registries.d
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
  -include $(DEST)local_recon_buildblock_registries.P
endif
endif


.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)librecon_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


