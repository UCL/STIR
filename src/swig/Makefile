SWIG=swig

all: _stir.so

stir_PYTHONwrap.cxx: stir.i Makefile
	$(SWIG) -python -c++ -builtin -I../include/ -I/usr/include -DSTART_NAMESPACE_STIR="namespace stir {" -DEND_NAMESPACE_STIR="}" -o $@ $< 

stir_OCTAVEwrap.cpp: stir.i Makefile
	$(SWIG) -octave -c++ -I../include/ -I/usr/include -DSTART_NAMESPACE_STIR="namespace stir {" -DEND_NAMESPACE_STIR="}"  -o $@ $<

stir.e: stir.i Makefile
	$(SWIG) -python -c++ -builtin -E  -I../include/ -DSTART_NAMESPACE_STIR="namespace stir {" -DEND_NAMESPACE_STIR="}" $< > stir.e

CFLAGS=-fpic  -I../include/ -g -I /usr/include/python2.7/ -DSWIG

%.o : %.cxx
	$(CXX) $(CFLAGS)  -o$@  -c $< 

FILES=Scanner.o interfile_keyword_functions.o utilities.o error.o warning.o VoxelsOnCartesianGrid.o ProjDataInfo.o Sinogram.o Viewgram.o SegmentBySinogram.o SegmentByView.o \
ProjDataInfoCylindrical.o ProjDataInfoCylindricalArcCorr.o ProjDataInfoCylindricalNoArcCorr.o RelatedViewgrams.o DataSymmetriesForViewSegmentNumbers.o IndexRange.o DiscretisedDensity.o

LIBS=../../build/stir/gcc/DebugShared/buildblock/libbuildblock.so \
../../build/stir/gcc/DebugShared/data_buildblock/libdata_buildblock.so \
../../build/stir/gcc/DebugShared/IO/libIO.so \
../../build/stir/gcc/DebugShared/modelling_buildblock/libmodelling_buildblock.so \
../../build/stir/gcc/DebugShared/numerics_buildblock/libnumerics_buildblock.so \
../../build/stir/gcc/DebugShared/local/IO/GE/liblocal_IO_GE.so \
../../build/stir/gcc/DebugShared/recon_buildblock/librecon_buildblock.so \
../../build/stir/gcc/DebugShared/listmode_buildblock/liblistmode_buildblock.so \
../../build/stir/gcc/DebugShared/iterative/OSMAPOSL/libiterative_OSMAPOSL.so \
../../build/stir/gcc/DebugShared/swig/CMakeFiles/_stir.dir/__/IO/IO_registries.cxx.o \
../../build/stir/gcc/DebugShared/swig/CMakeFiles/_stir.dir/__/recon_buildblock/recon_buildblock_registries.cxx.o \
../../build/stir/gcc/DebugShared/swig/CMakeFiles/_stir.dir/__/buildblock/buildblock_registries.cxx.o


LIBSa=-lbuildblock \
-ldata_buildblock \
-lIO \
-lmodelling_buildblock \
-lnumerics_buildblock \
-llocal_IO_GE \
-lrecon_buildblock \
-llistmode_buildblock \
-literative_OSMAPOSL \
../../build/stir/gcc/DebugShared/swig/CMakeFiles/_stir.dir/__/IO/IO_registries.cxx.o \
../../build/stir/gcc/DebugShared/swig/CMakeFiles/_stir.dir/__/recon_buildblock/recon_buildblock_registries.cxx.o \
../../build/stir/gcc/DebugShared/swig/CMakeFiles/_stir.dir/__/buildblock/buildblock_registries.cxx.o \
-L/home/kris/binDebugShared/lib


.PRECIOUS: stir_PYTHONwrap.cxx stir_OCTAVEwrap.cpp

_stir.so: stir_PYTHONwrap.o $(LIBS) Makefile
	g++ -g -shared stir_PYTHONwrap.o $(LIBS) -o $@


stir.oct: stir_OCTAVEwrap.cpp $(LIBS) Makefile
	CXXFLAGS=-g mkoctfile -v  -I../include/ -g  -DSWIG -o $@ $< $(LIBSa)