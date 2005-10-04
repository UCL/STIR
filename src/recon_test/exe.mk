#
# $Id$
#
dir := recon_test

$(dir)_SOURCES = bcktest.cxx fwdtest.cxx 

# rules that do not link with all registries to save time during linking
# Beware: the pattern below is dangerous as it relies on a naming style.
# Note: have to be before the include below as that resets $(dir)

#${DEST}$(dir)/bcktest: ${DEST}$(dir)/bcktest$(O_SUFFIX) \
#   $(STIR_LIB) $(filter %recon_buildblock_registries$(O_SUFFIX),$(STIR_REGISTRIES))
#	$(LINK) $(EXE_OUTFLAG)$@ $< \
#		$(filter %recon_buildblock_registries$(O_SUFFIX),$(STIR_REGISTRIES)) \
#		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)


#${DEST}$(dir)/fwdtest: ${DEST}$(dir)/fwdtest$(O_SUFFIX) \
#   $(STIR_LIB) $(filter %recon_buildblock_registries$(O_SUFFIX),$(STIR_REGISTRIES))
#	$(LINK) $(EXE_OUTFLAG)$@ $< \
#		$(filter %recon_buildblock_registries$(O_SUFFIX),$(STIR_REGISTRIES)) \
#		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

include $(WORKSPACE)/exe.mk
