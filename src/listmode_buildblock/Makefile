#
# $Id$
#

WORKSPACE=$(CURDIR)/../..


include $(WORKSPACE)/Makefile_common

LIB_SOURCES = \
	lm.cxx

SOURCES = \
	lm_to_projdata.cxx \
	lm_fansums.cxx

LIB_OBJS = $(LIB_SOURCES:%.cxx=${DEST}%.o)

OBJS = $(SOURCES:%.cxx=${DEST}%.o)

EXES = $(SOURCES:%.cxx=${DEST}%)

all: $(EXES)


include $(WORKSPACE)/Makefile_bblibs

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif


${DEST}lm_to_projdata : ${DEST}lm_to_projdata.o ${LIB_OBJS} $(BB_LIBS) 
	$(CXX) $(CFLAGS)  -o $@ $< $(LIB_OBJS) \
	$(BB_LIBS)  $(LINK_OPT) $(SYS_LIBS)

${DEST}lm_fansums : ${DEST}lm_fansums.o ${LIB_OBJS} $(BB_LIBS) 
	$(CXX) $(CFLAGS)  -o $@ $< $(LIB_OBJS) \
	$(BB_LIBS)  $(LINK_OPT) $(SYS_LIBS)


clean :
	$(RM) $(DEST)*

install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)


