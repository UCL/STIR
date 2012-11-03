all: _stir.so

stir_wrap.cxx: stir.i Makefile
	swig -python -c++ -builtin -I../include/ -I/usr/include -DSTART_NAMESPACE_STIR="namespace stir {" -DEND_NAMESPACE_STIR="}" $<

stir.e: stir.i Makefile
	swig -python -c++ -builtin -E  -I../include/ -DSTART_NAMESPACE_STIR="namespace stir {" -DEND_NAMESPACE_STIR="}" $< > stir.e

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


# TODO problem: need -fPIC for registries.o
# pushd /home/kris/devel/parapet/build/stir/gcc/DebugShared/swig/CMakeFiles/_stir.dir/__/ && for f in IO/IO_registries.cxx recon_buildblock/recon_buildblock_registries.cxx buildblock/buildblock_registries.cxx;do /usr/bin/c++    -DSTIR_USE_BOOST_SPTR -g -I/home/kris/devel/lln/ecat -I/home/kris/devel/GE/pettoolbox/io_utils/C/include -I/home/kris/devel/parapet/PPhead/include    -D HAVE_LLN_MATRIX -D HAVE_RDF -D STIR_USE_GE_IO -Wall -Wno-deprecated -o ${f}.o -c /home/kris/devel/parapet/PPhead/${f} -fPIC;done;popd

.PRECIOUS: stir_wrap.o

_stir.so: stir_wrap.o $(LIBS) Makefile
	g++ -g -shared stir_wrap.o $(LIBS) -o _stir.so


