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
	oldForwardProjectorByBinUsingRayTracing.cxx \
	oldForwardProjectorByBinUsingRayTracing_Siddon.cxx \
	oldBackProjectorByBinUsingInterpolation.cxx \
	oldBackProjectorByBinUsingInterpolation_linear.cxx \
	oldBackProjectorByBinUsingInterpolation_piecewise_linear.cxx \
	PostsmoothingForwardProjectorByBin.cxx \
	DataSymmetriesForDensels.cxx \
	DataSymmetriesForDensels_PET_CartesianGrid.cxx \
	ProjMatrixByDensel.cxx \
	ProjMatrixElemsForOneDensel.cxx \
	ProjMatrixByDenselUsingRayTracing.cxx


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)librecon_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


