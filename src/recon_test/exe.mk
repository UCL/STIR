#
# $Id$
#
dir := recon_test

$(dir)_SOURCES = bcktest.cxx fwdtest.cxx 

# rules that do not link with all registries to save time during linking
# Beware: the pattern below is dangerous as it relies on a naming style.
# Note: have to be before the include below as that resets $(dir)

${DEST}$(dir)/bcktest: ${DEST}$(dir)/bcktest.o \
   $(STIR_LIB) $(filter %recon_buildblock_registries.o,$(STIR_REGISTRIES))
	$(CXX) $(CFLAGS)  -o $@ $< \
		$(filter %recon_buildblock_registries.o,$(STIR_REGISTRIES)) \
		 $(STIR_LIB)  $(LINK_OPT) $(SYS_LIBS)


${DEST}$(dir)/fwdtest: ${DEST}$(dir)/fwdtest.o \
   $(STIR_LIB) $(filter %recon_buildblock_registries.o,$(STIR_REGISTRIES))
	$(CXX) $(CFLAGS)  -o $@ $< \
		$(filter %recon_buildblock_registries.o,$(STIR_REGISTRIES)) \
		 $(STIR_LIB)  $(LINK_OPT) $(SYS_LIBS)

include $(WORKSPACE)/exe.mk
