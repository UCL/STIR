#
# $Id$: $Date$
#

# AZ 03/10/99 Added
__TEST_OS = $(shell uname)
ifeq ($(__TEST_OS),Linux)
__TEST_LINUX = 1
else
__TEST_LINUX = 0
endif

WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include

#AZ 03/02/99 Added

ifeq ($(DEST),debug/)
ifeq ($(__TEST_LINUX),1)
DEBUG = -D_DEBUG -pg
else # not Linux
DEBUG = -D_DEBUG
endif # Linux?
else # not debug/
ifeq ($(DEST),opt/)
ifeq ($(__TEST_LINUX),1)
#-DNDEBUG is needed by system's assert
DEBUG = -O3 -finline-functions -ffast-math -m486 -malign-double -malign-functions=2 -malign-loops=2 -malign-jumps=2 -DCPU=686 -Winline -DNDEBUG
#TODO put this back [incompatible wiht -pg] -fomit-frame-pointer 
# -fforce-mem
else # not Linux
# KT 17/01/2000 added NDEBUG
DEBUG = -O3 -DNDEBUG
endif # Linux?
endif # opt/ ?
endif # debug/ ?

CXX = g++
# KT 15/12/99 changed /bin/rm to rm (otherwise problems on NT)
RM = rm -f 
DEST=debug/
CFLAGS = -g $(DEBUG) -I$(INCLUDE_DIR)

#KT 27/02/98 use default rules now, .cxx does not work without next line
.SUFFIXES: .cc .o .cxx .a

# KT 04/11/98 enabled built-in sccs
#.SCCS_GET:

.KEEP_STATE:

all: createdir $(DEST)libbuildblock.a

createdir: 
	-mkdir -p $(DEST)

# KT 15/12/99 added createdir depedency
# KT 18/02/2000 removed depend
Makefile: createdir

$(DEST)%.o : %.cxx 
	$(CXX) $(CFLAGS) -o $@ -c $< 

# KT 10/12/99 added linear_regression.cxx
# KT 15/12/99 removed NumericInfo, added PETerror, overlap_interpolate, NumericType and ByteOrder
# KT 17/01/2000 added zoom and mash_views
# KT   Array.cxx convert_array.cxx
SOURCES = \
  PETerror.cxx \
  sinodata.cxx convert.cxx \
  PETScannerInfo.cxx PETScanInfo.cxx \
  Tensor1D.cxx Tensors.cxx \
  viewdata.cxx imagedata.cxx \
  PETDataSymmetries.cxx \
  PETSinogramOfVolume.cxx \
  PETRelatedViewgrams.cxx \
  utilities.cxx NumericType.cxx ByteOrder.cxx \
  KeyParser.cxx line.cxx \
  interfile.cxx InterfileHeader.cxx \
  bbfilters.cxx recon_array_functions.cxx \
  sparsedata.cxx \
  linear_regression.cxx overlap_interpolate.cxx \
  zoom.cxx mash_views.cxx \
  Array.cxx convert_array.cxx


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

$(DEST)libbuildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


# KT 15/12/99 use makedepend
depend:
	#ppdep -I ../include $(SOURCES)
	makedepend -Y -- $(CFLAGS) -- -fMakefile -p${DEST} $(SOURCES) 2> /dev/null;

#KT 30/07/98 added DEST
clean: 
	$(RM) $(DEST)*.o
	$(RM) $(DEST)*.a

