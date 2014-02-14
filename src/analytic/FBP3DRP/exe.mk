#
#

dir:=analytic/FBP3DRP

$(dir)_SOURCES = \
	FBP3DRP.cxx

include $(WORKSPACE)/exe.mk

${DEST}$(dir)/FBP3DRP: ${DEST}$(dir)/FBP3DRP$(O_SUFFIX) \
   $(STIR_LIB) \
   $(filter %recon_buildblock_registries$(O_SUFFIX) %IO_registries$(O_SUFFIX),$(STIR_REGISTRIES))
	$(LINK) $(EXE_OUTFLAG)$@ $< \
                 $(filter %recon_buildblock_registries $(O_SUFFIX) %IO_registries$(O_SUFFIX),$(STIR_REGISTRIES)) \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)
