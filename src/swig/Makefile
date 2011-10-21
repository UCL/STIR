
stir_wrap.cxx: stir.i
	swig -python -c++ -builtin -I../include/ -DSTART_NAMESPACE_STIR="namespace stir {" -DEND_NAMESPACE_STIR="}" $<

CFLAGS=-fpic  -I../include/ -g -I /usr/include/python2.7/ -DSWIG

%.o : %.cxx
	$(CXX) $(CFLAGS)  -o$@  -c $< 

FILES=Scanner.o interfile_keyword_functions.o utilities.o error.o warning.o VoxelsOnCartesianGrid.o ProjDataInfo.o Sinogram.o Viewgram.o SegmentBySinogram.o SegmentByView.o \
ProjDataInfoCylindrical.o ProjDataInfoCylindricalArcCorr.o ProjDataInfoCylindricalNoArcCorr.o RelatedViewgrams.o DataSymmetriesForViewSegmentNumbers.o IndexRange.o DiscretisedDensity.o

LIBS=../../build/stir/gcc/DebugShared/buildblock/libbuildblock.so \
../../build/stir/gcc/DebugShared/IO/libIO.so \
../../build/stir/gcc/DebugShared/numerics_buildblock/libnumerics_buildblock.so

_stir.so: stir_wrap.o $(LIBS)
	g++ -g -shared $^ -o _stir.so

all: _stir.so

