#
# $Id$
#
include ../../Makefile_common


LIB_SOURCES =  \
         OSMAPOSLParameters.cxx \
         OSMAPOSLReconstruction.cxx 

MAIN_SOURCES = OSMAPOSL.cxx

LIB_OBJS = $(LIB_SOURCES:%.cxx=$(DEST)%.o)

MAIN_OBJS = $(MAIN_SOURCES:%.cxx=$(DEST)%.o)

EXES = $(MAIN_SOURCES:%.cxx=$(DEST)%)

.PRECIOUS: $(LIB_OBJS) $(MAIN_OBJS) $(EXES)


all: $(EXES)

clean:
	$(RM) $(DEST)*


install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)


include ../../Makefile_bblibs


${DEST}OSMAPOSL: ${DEST}OSMAPOSL.o $(LIB_OBJS) \
	  ../../buildblock/${DEST}registrystuff.o\
	  $(BB_LIBS) ${LIB_LOGLIK_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)-o $@ $< \
        $(LIB_OBJS) \
	../../buildblock/${DEST}registrystuff.o \
	${LIB_LOGLIK_BUILDINGBLOCK} \
	$(LIB_RECON_BUILDINGBLOCK) \
	$(LIB_BUILDINGBLOCK) \
	$(LINK_OPT) $(SYS_LIBS)



# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
include $(MAIN_SOURCES:%.cxx=$(DEST)%.d)

include $(LIB_SOURCES:%.cxx=$(DEST)%.d)
endif




