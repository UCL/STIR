#
# $Id$
#
WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common


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


include $(WORKSPACE)/Makefile_bblibs


${DEST}OSMAPOSL: ${DEST}OSMAPOSL.o $(LIB_OBJS) \
	  $(BB_LIBS) ${LIB_LOGLIK_BUILDINGBLOCK}  $(BB_REGISTRIES)
	$(CXX) $(CFLAGS)-o $@ $< \
        $(LIB_OBJS) \
	 $(BB_REGISTRIES) \
	${LIB_LOGLIK_BUILDINGBLOCK}  $(BB_LIBS) \
	$(LINK_OPT) $(SYS_LIBS)



# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(MAIN_SOURCES:%.cxx=$(DEST)%.d)
  include $(LIB_SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(MAIN_SOURCES:%.cxx=$(DEST)%.P)
  -include $(LIB_SOURCES:%.cxx=$(DEST)%.P)
endif
endif




