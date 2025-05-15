/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2013, 2014, 2015, 2018 - 2023 University College London
    Copyright (C) 2022 National Physical Laboratory
    Copyright (C) 2022 Positrigo
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: main

  \author Kris Thielemans
  \author Daniel Deidda
  \author Markus Jehl
*/


%module stir
%{
#define SWIG_DOC_DOXYGEN_STYLE
#define SWIG_FILE_WITH_INIT

 /* Include the following headers in the wrapper code */
#include <string>
#include <list>
#include <cstdio> // for size_t
#include <sstream>
#include <iterator>

#ifdef SWIGOCTAVE
// TODO terrible work-around to avoid conflict between stir::error and Octave error
// they are in conflict with eachother because we need "using namespace stir" below (swig bug)
#define __stir_error_H__
#endif

#include "stir/stream.h" // to get access to stream output for STIR types for ADD_REPR etc
#include "stir/num_threads.h"

 #include "stir/find_STIR_config.h"
 #include "stir/Succeeded.h"
 #include "stir/DetectionPosition.h"
 #include "stir/Scanner.h"
 #include "stir/Bin.h"
 #include "stir/ProjDataInfoCylindricalArcCorr.h"
 #include "stir/ProjDataInfoCylindricalNoArcCorr.h"
 #include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
 #include "stir/ProjDataInfoGenericNoArcCorr.h"

 #include "stir/Viewgram.h"
 #include "stir/RelatedViewgrams.h"
 #include "stir/Sinogram.h"
 #include "stir/SegmentByView.h"
 #include "stir/SegmentBySinogram.h"
 #include "stir/ExamInfo.h"
 #include "stir/ExamData.h"
 #include "stir/Verbosity.h"
 #include "stir/ProjData.h"
 #include "stir/ProjDataInMemory.h"
 #include "stir/copy_fill.h"
 #include "stir/ProjDataInterfile.h"
#include "stir/format.h"

 #include "stir/Radionuclide.h"
 #include "stir/RadionuclideDB.h"

 #include "stir/DataSymmetriesForViewSegmentNumbers.h"
 #include "stir/recon_buildblock/BinNormalisationFromProjData.h"
 #include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"
 #include "stir/recon_buildblock/TrivialBinNormalisation.h"
 #include "stir/listmode/ListRecord.h"
 #include "stir/listmode/ListEvent.h"
 #include "stir/listmode/CListRecord.h"
 #include "stir/listmode/ListModeData.h"
 #include "stir/listmode/CListModeData.h"
 #include "stir/listmode/LmToProjData.h"

#include "stir/CartesianCoordinate2D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/LORCoordinates.h"
#include "stir/IndexRange.h"
#include "stir/IndexRange3D.h"
#include "stir/IndexRange4D.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/PixelsOnCartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/zoom.h"

#include "stir/GeneralisedPoissonNoiseGenerator.h"
  
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/IO/InterfileOutputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT7OutputFileFormat.h"
#endif

#ifdef HAVE_ITK
#include "stir/IO/ITKOutputFileFormat.h"
#endif

#include "stir/Shape/Ellipsoid.h"
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/Shape/Box3D.h"


#include "stir/evaluation/ROIValues.h"
#include "stir/evaluation/compute_ROI_values.h"


#include "stir/ChainedDataProcessor.h"
#include "stir/SeparableCartesianMetzImageFilter.h"
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/SeparableConvolutionImageFilter.h"
#include "stir/TruncateToCylindricalFOVImageProcessor.h"

#include "stir/HUToMuImageProcessor.h"

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h" 
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/OSSPS/OSSPSReconstruction.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"

#ifdef STIR_WITH_Parallelproj_PROJECTOR
#include "stir/recon_buildblock/Parallelproj_projector/ForwardProjectorByBinParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/BackProjectorByBinParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/ProjectorByBinPairUsingParallelproj.h"
#endif

#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixByBinSPECTUB.h"
#include "stir/recon_buildblock/ProjMatrixByBinPinholeSPECTUB.h"
#include "stir/recon_buildblock/QuadraticPrior.h"
#include "stir/recon_buildblock/PLSPrior.h"
#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/recon_buildblock/LogcoshPrior.h"


#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"

#include "stir/analytic/FBP2D/FBP2DReconstruction.h"
#include "stir/analytic/FBP3DRP/FBP3DRPReconstruction.h"

#include "stir/recon_buildblock/SqrtHessianRowSum.h"

#include "stir/multiply_crystal_factors.h"
#include "stir/decay_correction_factor.h"
#include "stir/ML_norm.h"
#include "stir/spatial_transformation/InvertAxis.h"

#include "stir/scatter/ScatterEstimation.h"
#include "stir/scatter/ScatterSimulation.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/scatter/CreateTailMaskFromACFs.h"

#include "stir/SSRB.h"
#include "stir/inverse_SSRB.h"

#include <boost/iterator/reverse_iterator.hpp>
#include <stdexcept>

   // TODO need this (bug in swig)
   // this bug occurs (only?) when using "%template(name) someclass;" inside the namespace
   // as opposed to "%template(name) stir::someclass" outside the namespace
   using namespace stir;
   using std::iostream;

#if defined(SWIGPYTHON)
   // need to declare this internal SWIG function as we're using it in the
   // helper code below. It is used to convert a Python object to a float.
   SWIGINTERN int
   SWIG_AsVal_double (PyObject * obj, double *val);
#endif

   // local helper functions for conversions etc. These are not "exposed" to the target language 
   // (but only enter in the wrapper)
   namespace swigstir {
#if defined(SWIGPYTHON)
   // helper function to translate a tuple to a BasicCoordinate
   // returns zero on failure
   template <int num_dimensions, typename coordT>
     int coord_from_tuple(stir::BasicCoordinate<num_dimensions, coordT>& c, PyObject* const args)
    { return 0;  }
    template<> int coord_from_tuple(stir::BasicCoordinate<1, int>& c, PyObject* const args)
    { return PyArg_ParseTuple(args, "i", &c[1]);  }
    template<> int coord_from_tuple(stir::BasicCoordinate<2, int>& c, PyObject* const args)
    { return PyArg_ParseTuple(args, "ii", &c[1], &c[2]);  }
    template<> int coord_from_tuple(stir::BasicCoordinate<3, int>& c, PyObject* const args)
      { return PyArg_ParseTuple(args, "iii", &c[1], &c[2], &c[3]);  }
    template<> int coord_from_tuple(stir::BasicCoordinate<4, int>& c, PyObject* const args)
      { return PyArg_ParseTuple(args, "iiii", &c[1], &c[2], &c[3], &c[4]);  }
    template<> int coord_from_tuple(stir::BasicCoordinate<1, float>& c, PyObject* const args)
    { return PyArg_ParseTuple(args, "f", &c[1]);  }
    template<> int coord_from_tuple(stir::BasicCoordinate<2, float>& c, PyObject* const args)
    { return PyArg_ParseTuple(args, "ff", &c[1], &c[2]);  }
    template<> int coord_from_tuple(stir::BasicCoordinate<3, float>& c, PyObject* const args)
      { return PyArg_ParseTuple(args, "fff", &c[1], &c[2], &c[3]);  }
    template<> int coord_from_tuple(stir::BasicCoordinate<4, float>& c, PyObject* const args)
      { return PyArg_ParseTuple(args, "ffff", &c[1], &c[2], &c[3], &c[4]);  }

    template <int num_dimensions>
      PyObject* tuple_from_coord(const stir::BasicCoordinate<num_dimensions, int>& c)
      {
	PyObject* p = PyTuple_New(num_dimensions);
	for (int d=1; d<=num_dimensions; ++d)
	  PyTuple_SET_ITEM(p, d-1, PyInt_FromLong(c[d]));
	return p;
      }
    template <int num_dimensions>
      PyObject* tuple_from_coord(const stir::BasicCoordinate<num_dimensions, std::size_t>& c)
      {
	PyObject* p = PyTuple_New(num_dimensions);
	for (int d=1; d<=num_dimensions; ++d)
	  PyTuple_SET_ITEM(p, d-1, PyInt_FromSize_t(c[d]));
	return p;
      }
    template <int num_dimensions>
      PyObject* tuple_from_coord(const stir::BasicCoordinate<num_dimensions,float>& c)
      {
	PyObject* p = PyTuple_New(num_dimensions);
	for (int d=1; d<=num_dimensions; ++d)
	  PyTuple_SET_ITEM(p, d-1, PyFloat_FromDouble(double(c[d])));
	return p;
      }

    // fill an array from a Python sequence
    // (could be trivially modified to just write to a C++ iterator)
    template <int num_dimensions, typename elemT>
      void fill_Array_from_Python_iterator(stir::Array<num_dimensions, elemT> * array_ptr, PyObject* const arg)
    {
      if (!PyIter_Check(arg))
	throw std::runtime_error("STIR-Python internal error: fill_Array_from_Python_iterators called but input is not an iterator");

      {
	PyObject *iterator = PyObject_GetIter(arg);
	
	PyObject *item;
	typename stir::Array<num_dimensions, elemT>::full_iterator array_iter = array_ptr->begin_all();
	while ((item = PyIter_Next(iterator)) && array_iter != array_ptr->end_all()) 
        {
	  double val;
	  // TODO currently hard-wired as double which might imply extra conversions
	  int ecode = SWIG_AsVal_double(item, &val);
	  if (SWIG_IsOK(ecode)) 
	  {
	    *array_iter++ = static_cast<elemT>(val);
	  }
	  else
	  {
	    Py_DECREF(item);
	    Py_DECREF(iterator);
	    char str[1000];
	    snprintf(str, 1000, "Wrong type used for fill(): iterator elements are of type %s but needs to be convertible to double",
			 item->ob_type->tp_name);
	    throw std::invalid_argument(str);
	  }
	  Py_DECREF(item);
	}

	if (PyIter_Next(iterator) != NULL || array_iter != array_ptr->end_all())
        {
	  throw std::runtime_error("fill() called with incorrect range of iterators, array needs to have the same number of elements");
	}
	Py_DECREF(iterator);

	if (PyErr_Occurred()) 
	{
	  throw std::runtime_error("Error during fill()");
	}
      }

    }

#if 0
    
    // TODO  does not work yet.
    // it doesn't compile as includes are in init section, which follows after this in the wrapper
    // Even if it did compile, it might not work anyway as I haven't tested it.
    template <typename IterT>
      void fill_nparray_from_iterator(PyObject * np, IterT iterator)
    {
      // This code is more or less a copy of the "simple iterator example" (!) in the Numpy doc
      // see e.g. http://students.mimuw.edu.pl/~pbechler/numpy_doc/reference/c-api.iterator.html
      typedef float elemT;
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp* strideptr,* innersizeptr;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(np) == 0) {
        return;
    }

    /*
     * Create and use an iterator to count the nonzeros.
     *   flag NPY_ITER_READONLY
     *     - The array is never written to.
     *   flag NPY_ITER_EXTERNAL_LOOP
     *     - Inner loop is done outside the iterator for efficiency.
     *   flag NPY_ITER_NPY_ITER_REFS_OK
     *     - Reference types are acceptable.
     *   order NPY_KEEPORDER
     *     - Visit elements in memory order, regardless of strides.
     *       This is good for performance when the specific order
     *       elements are visited is unimportant.
     *   casting NPY_NO_CASTING
     *     - No casting is required for this operation.
     */
    iter = NpyIter_New(np, NPY_ITER_WRITEONLY|
                             NPY_ITER_EXTERNAL_LOOP,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
      throw std::runtime_error("Error creating numpy iterator");
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
	throw std::runtime_error("Error creating numpy iterator function");
    }
    /* The location of the data pointer which the iterator may update */
    dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    /* The iteration loop */
    do {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
        while (count--) {
	  *(reinterpret_cast<elemT *>(data)) = static_cast<elemT>(*iterator++);
            data += stride;
        }

        /* Increment the iterator to the next inner loop */
    } while(iternext(iter));

    NpyIter_Deallocate(iter);
    }
#endif

#elif defined(SWIGMATLAB)
     // convert stir::Array to matlab (currently always converting to double)
     // note that the order of the dimensions is reversed to take row-first vs column-first ordering into account
     template <int num_dimensions, typename elemT>
     mxArray * Array_to_matlab(const stir::Array<num_dimensions, elemT>& a)
     {
       mwSize dims[num_dimensions];
       BasicCoordinate<num_dimensions,int> minind,maxind;
       a.get_regular_range(minind,maxind);
       const BasicCoordinate<num_dimensions,int> sizes=maxind-minind+1;
       // copy dimensions in reverse order
       std::copy(boost::make_reverse_iterator(sizes.end()), boost::make_reverse_iterator(sizes.begin()), dims);
       mxArray *pm = mxCreateNumericArray(num_dimensions, dims, mxDOUBLE_CLASS, mxREAL);
       double * data_ptr = mxGetPr(pm);
       std::copy(a.begin_all(), a.end_all(), data_ptr);
       return pm;
     }

     template <int num_dimensions, typename elemT>
     void fill_Array_from_matlab_scalar(stir::Array<num_dimensions, elemT>& a, const mxArray *pm)
     {
       if (mxIsDouble(pm))
       {
         double const* data_ptr = mxGetPr(pm);
         a.fill(static_cast<elemT>(*data_ptr));
       } else if (mxIsSingle(pm))
       {
         float const* data_ptr = (float *)mxGetData(pm);
         a.fill(static_cast<elemT>(*data_ptr));
       } else
       { 
         throw std::runtime_error("currently only supporting double or single arrays for filling a stir array");
       }
     }

     template <int num_dimensions, typename elemT>
     void fill_Array_from_matlab(stir::Array<num_dimensions, elemT>& a, const mxArray *pm, bool do_resize)
     {
       mwSize matlab_num_dims = mxGetNumberOfDimensions(pm);
       const mwSize * m_sizes = mxGetDimensions(pm);
       // matlab represents scalars/vectors as a matrix, so let's check this first
       if (matlab_num_dims == static_cast<mwSize>(2) && m_sizes[1]==static_cast<mwSize>(1))
       {
         if (m_sizes[0] ==static_cast<mwSize>(1))
         {
           // it's a scalar
           fill_Array_from_matlab_scalar(a, pm);
           return;
         }
         matlab_num_dims=static_cast<mwSize>(1); // set it to a 1-dimensional array
       }
       if (matlab_num_dims > static_cast<mwSize>(num_dimensions))
       {
         throw std::runtime_error(format("number of dimensions in matlab array is incorrect for constructing a stir array of dimension {}", 
                                             num_dimensions)); 
       }
       if (do_resize)
       {
         BasicCoordinate<num_dimensions,int>  sizes;
         // first set all to 1 to cope with lower-dimensional arrays from matlab
         sizes.fill(1);
         std::copy(m_sizes, m_sizes+matlab_num_dims, boost::make_reverse_iterator(sizes.end()));
         a.resize(sizes);
       }
       else
       { 
         // check sizes
         BasicCoordinate<num_dimensions,int> minind,maxind;
         a.get_regular_range(minind,maxind);
         const BasicCoordinate<num_dimensions,int> sizes=maxind-minind+1;
         if (!std::equal(m_sizes, m_sizes+matlab_num_dims, boost::make_reverse_iterator(sizes.end())))
         {
           throw std::runtime_error("sizes of matlab array incompatible with stir array");
         }
         for (int d=1; d<= num_dimensions-static_cast<int>(matlab_num_dims); ++d)
         {
           if (sizes[d]!=1)
           {
             throw std::runtime_error("sizes of first dimensions of the stir array have to be 1 if initialising from a lower dimensional matlab array");
           }
         }
       }       
       if (mxIsDouble(pm))
       {
         double * data_ptr = mxGetPr(pm);
         std::copy(data_ptr, data_ptr+a.size_all(), a.begin_all());
       } else if (mxIsSingle(pm))
       {
         float * data_ptr = (float *)mxGetData(pm);
         std::copy(data_ptr, data_ptr+a.size_all(), a.begin_all());
       } else
       { 
         throw std::runtime_error("currently only supporting double or single arrays for constructing a stir array");
       }
     }     


     //////////// same for Coordinate
     // convert stir::BasicCoordinate to matlab (currently always converting to double)
     template <int num_dimensions, typename elemT>
     mxArray * BasicCoordinate_to_matlab(const stir::BasicCoordinate<num_dimensions, elemT>& a)
     {
       mwSize dims[2];
       dims[0]=mwSize(num_dimensions);
       dims[1]=mwSize(1);
       mxArray *pm = mxCreateNumericArray(mwSize(2), dims, mxDOUBLE_CLASS, mxREAL);
       double * data_ptr = mxGetPr(pm);
       std::copy(a.begin(), a.end(), data_ptr);
       return pm;
     }

     template <int num_dimensions, typename elemT>
     void fill_BasicCoordinate_from_matlab_scalar(stir::BasicCoordinate<num_dimensions, elemT>& a, const mxArray *pm)
     {
       if (mxIsDouble(pm))
       {
         double const* data_ptr = mxGetPr(pm);
         a.fill(static_cast<elemT>(*data_ptr));
       } else if (mxIsSingle(pm))
       {
         float const* data_ptr = (float *)mxGetData(pm);
         a.fill(static_cast<elemT>(*data_ptr));
       } else
       { 
         throw std::runtime_error("currently only supporting double or single arrays for filling a stir coordinate");
       }
     }

     template <int num_dimensions, typename elemT>
     void fill_BasicCoordinate_from_matlab(stir::BasicCoordinate<num_dimensions, elemT>& a, const mxArray *pm)
     {
       mwSize matlab_num_dims = mxGetNumberOfDimensions(pm);
       const mwSize * m_sizes = mxGetDimensions(pm);
       // matlab represents scalars/vectors as a matrix, so let's check this first
       if (matlab_num_dims == static_cast<mwSize>(2) && m_sizes[1]==static_cast<mwSize>(1))
       {
         if (m_sizes[0] ==static_cast<mwSize>(1))
         {
           // it's a scalar
           fill_BasicCoordinate_from_matlab_scalar(a, pm);
           return;
         }
         matlab_num_dims=static_cast<mwSize>(1); // set it to a 1-dimensional array
       }
       if (matlab_num_dims != static_cast<mwSize>(1))
       {
         throw std::runtime_error(format("number of dimensions {} of matlab array is incorrect for constructing a stir coordinate of dimension {} (expecting a column vector)", 
                                             matlab_num_dims , num_dimensions)); 
       }
       if (m_sizes[0]!=static_cast<mwSize>(num_dimensions))
       {
	 throw std::runtime_error("length of matlab array incompatible with stir coordinate");
       }
       if (mxIsDouble(pm))
       {
         double * data_ptr = mxGetPr(pm);
         std::copy(data_ptr, data_ptr+a.size(), a.begin());
       } else if (mxIsSingle(pm))
       {
         float * data_ptr = (float *)mxGetData(pm);
         std::copy(data_ptr, data_ptr+a.size(), a.begin());
       } else
       { 
         throw std::runtime_error("currently only supporting double or single arrays for constructing a stir array");
       }
     }     
#endif
  } // end namespace swigstir
 %}

#if defined(SWIGPYTHON)
%include "numpy.i"
%fragment("NumPy_Fragments");
#endif

%include "attribute.i"
%include "factory_shared.i"

%init %{
#if defined(SWIGPYTHON)
  // numpy support
  import_array();
   #include <numpy/ndarraytypes.h>
#endif
%}

%feature("autodoc", "1");
// Use include set by build
#if defined(DOXY2SWIG_XML_INCLUDE_FILE)
%include DOXY2SWIG_XML_INCLUDE_FILE
#endif

// TODO doesn't work
%warnfilter(315) std::unique_ptr;

// disable warnings about unknown base-class 401
// disable warnings about "no access specified given for base class" as we use this correctly for private derivation 319
// disable warnings about nested bass-class 325
#pragma SWIG nowarn=319,401,325

// catch all C++ exceptions in python
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  } catch (const std::string& e) {
    SWIG_exception(SWIG_RuntimeError, e.c_str());
  }
} 

// declare some functions that return a new pointer such that SWIG can release memory properly
%newobject *::clone;
%newobject *::get_empty_copy;
%newobject *::read_from_file;

// SWIG complains "Warning 503: Can't wrap 'stir::swap' unless renamed to a valid identifier."
// But it's probably dangerous to expose swap anyway, so let's ignore it.
%ignore **::swap;
%ignore stir::swap;
%ignore *::ask_parameters;
%ignore *::create_shared_clone;
%ignore *::read_from_stream;
%ignore *::get_data_ptr;
%ignore *::get_const_data_ptr;
%ignore *::release_data_ptr;
%ignore *::release_const_data_ptr;
%ignore *::get_full_data_ptr;
%ignore *::get_const_full_data_ptr;
%ignore *::release_full_data_ptr;
%ignore *::release_const_full_data_ptr;

#if defined(SWIGPYTHON)
%rename(__assign__) *::operator=; 
#endif

// include standard swig support for some bits of the STL (i.e. standard C++ lib)
%include <stl.i>
%include <std_list.i>
 // ignore iterators, as they don't make sense in the target language
%ignore *::begin;
%ignore *::rbegin;
%ignore *::begin_all;
%ignore *::rbegin_all;
%ignore *::end;
%ignore *::rend;
%ignore *::end_all;
%ignore *::rend_all;
%ignore *::begin_all_const;
%ignore *::rbegin_all_const;
%ignore *::end_all_const;
%ignore *::rend_all_const;

// always ignore these as they are unsafe in out-of-range index access (use at() instead)
%ignore *::operator[];
 // this will be replaced by __getitem__ etc, we could keep this for languages not supported by ADD_indexvalue
%ignore *::at;

#ifdef STIRMATLAB
%ignore *::operator>>;
%ignore *::operator<<;
%ignore *::operator+=;
%ignore *::operator-=;
%ignore *::operator*=;
%ignore *::operator/=;

// use isequal, not eq at the moment. This might change later.
//%rename(isequal) *::operator==;
%rename(isequal) *::eq;
#endif

#ifndef SWIGOCTAVE
%include <std_ios.i>
// do not need this at present
// %include <std_iostream.i>
#endif

// Instantiate STL templates used by stir
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(FloatVector) vector<float>;
   %template(StringList) list<string>;
}

// section for helper classes for creating new iterators. 
// The code here is nearly a copy of what's in PyIterators.swg,
// except that the decr() function isn't defined. This is because we need it for some STIR iterators
// which are forward_iterators.
// Ideally this code would be moved to SWIG.
//
// Note: this needs to be defined after including some stl stuff, as otherwise the necessary fragments
// haven't been included in the wrap.cxx yet.
%{
  namespace swigstir {
#ifdef SWIGPYTHON
  template<typename OutIterator, 
	   typename ValueType = typename std::iterator_traits<OutIterator>::value_type,
    typename FromOper = swig::from_oper<ValueType> >
    class SwigPyForwardIteratorClosed_T :  public swig::SwigPyIterator_T<OutIterator>
  {
  public:
    FromOper from;
    typedef OutIterator out_iterator;
    typedef ValueType value_type;
    typedef swig::SwigPyIterator_T<out_iterator>  base;    
    typedef SwigPyForwardIteratorClosed_T<OutIterator, ValueType, FromOper> self_type;
    
    SwigPyForwardIteratorClosed_T(out_iterator curr, out_iterator first, out_iterator last, PyObject *seq)
      : swig::SwigPyIterator_T<OutIterator>(curr, seq), begin(first), end(last)
    {
    }
    
    PyObject *value() const {
      if (base::current == end) {
	throw swig::stop_iteration();
      } else {
	return swig::from(static_cast<const value_type&>(*(base::current)));
      }
    }
    
  swig::SwigPyIterator *copy() const
    {
      return new self_type(*this);
    }

    swig::SwigPyIterator *incr(size_t n = 1)
    {
      while (n--) {
	if (base::current == end) {
	  throw swig::stop_iteration();
	} else {
	  ++base::current;
	}
      }
      return this;
    }

  private:
    out_iterator begin;
    out_iterator end;
  };

  template<typename OutIter>
  inline swig::SwigPyIterator*
  make_forward_iterator(const OutIter& current, const OutIter& begin,const OutIter& end, PyObject *seq = 0)
  {
    return new SwigPyForwardIteratorClosed_T<OutIter>(current, begin, end, seq);
  }


#endif
  static Array<4,float> create_array_for_proj_data(const ProjData& proj_data)
  {
    const int num_non_tof_sinos = proj_data.get_num_non_tof_sinograms();
 Array<4,float> array(IndexRange4D(proj_data.get_num_tof_poss(),num_non_tof_sinos, proj_data.get_num_views(), proj_data.get_num_tangential_poss()));
      return array;
  }

  // a function for  converting ProjData to a 4D array as that's what is easy to use
  static Array<4,float> projdata_to_4D(const ProjData& proj_data)
  {

      Array<4,float> array = create_array_for_proj_data(proj_data);
      Array<4,float>::full_iterator array_iter = array.begin_all();
      //    for (int s=0; s<= proj_data.get_max_segment_num(); ++s)
      //      {
      //        SegmentBySinogram<float> segment=proj_data.get_segment_by_sinogram(s);
      //        std::copy(segment.begin_all_const(), segment.end_all_const(), array_iter);
      //        std::advance(array_iter, segment.size_all());
      //        if (s!=0)
      //          {
      //            segment=proj_data.get_segment_by_sinogram(-s);
      //            std::copy(segment.begin_all_const(), segment.end_all_const(), array_iter);
      //            std::advance(array_iter, segment.size_all());
      //          }
      //      }
      proj_data.copy_to(array_iter);
      return array;
  }

  // inverse of the above function
  void fill_proj_data_from_4D(ProjData& proj_data, const Array<4,float>& array)
  {
      //    int num_sinos=proj_data.get_num_axial_poss(0);
      //    for (int s=1; s<= proj_data.get_max_segment_num(); ++s)
      //      {
      //        num_sinos += 2*proj_data.get_num_axial_poss(s);
      //      }
      //    if (array.size() != static_cast<std::size_t>(num_sinos)||
      //        array[0].size() != static_cast<std::size_t>(proj_data.get_num_views()) ||
      //        array[0][0].size() != static_cast<std::size_t>(proj_data.get_num_tangential_poss()))
      //      {
      //        throw std::runtime_error("Incorrect size for filling this projection data");
      //      }
      Array<4,float>::const_full_iterator array_iter = array.begin_all();
      //
      //    for (int s=0; s<= proj_data.get_max_segment_num(); ++s)
      //      {
      //        SegmentBySinogram<float> segment=proj_data.get_empty_segment_by_sinogram(s);
      //        // cannot use std::copy sadly as needs end-iterator for range
      //        for (SegmentBySinogram<float>::full_iterator seg_iter = segment.begin_all();
      //             seg_iter != segment.end_all();
      //             /*empty*/)
      //          *seg_iter++ = *array_iter++;
      //        proj_data.set_segment(segment);
      //
      //        if (s!=0)
      //          {
      //            segment=proj_data.get_empty_segment_by_sinogram(-s);
      //            for (SegmentBySinogram<float>::full_iterator seg_iter = segment.begin_all();
      //                 seg_iter != segment.end_all();
      //                 /*empty*/)
      //              *seg_iter++ = *array_iter++;
      //            proj_data.set_segment(segment);
      //          }
      //      }
      proj_data.fill_from(array_iter);
  }

//  static Array<3,float> create_array_for_proj_data(const ProjData& proj_data)
//  {
//      //    int num_sinos=proj_data.get_num_axial_poss(0);
//      //    for (int s=1; s<= proj_data.get_max_segment_num(); ++s)
//      //      {
//      //        num_sinos += 2*proj_data.get_num_axial_poss(s);
//      //      }
//      int num_sinos = proj_data.get_num_sinograms();

//      Array<3,float> array(IndexRange3D(num_sinos, proj_data.get_num_views(), proj_data.get_num_tangential_poss()));
//      return array;
//  }

//  static Array<3,float> projdata_to_3D(const ProjData& proj_data)
//  {
//      Array<3,float> array = create_array_for_proj_data(proj_data);
//      Array<3,float>::full_iterator array_iter = array.begin_all();
//      copy_to(proj_data, array_iter);
//      return array;
//  }

  
 } // end of namespace

  %} // end of initial code specification for inclusino in the SWIG wrapper

// doesn't work (yet?) because of bug in int template arguments
// %rename(__getitem__) *::at; 

// MACROS to define index access (Work-in-progress)
%define %ADD_indexaccess(INDEXTYPE,RETTYPE,TYPE...)
#if defined(SWIGPYTHON)
#if defined(SWIGPYTHON_BUILTIN)
 //  %feature("python:slot", "sq_item", functype="ssizeargfunc") TYPE##::__getitem__;
 // %feature("python:slot", "sq_ass_item", functype="ssizeobjargproc") TYPE##::__setitem__;
  %feature("python:slot", "mp_subscript", functype="binaryfunc") TYPE##::__getitem__;
  %feature("python:slot", "mp_ass_subscript", functype="objobjargproc") TYPE##::__setitem__;

#endif
%extend TYPE {
    %exception __getitem__ {
      try
	{
	  $action
	}
      catch (std::out_of_range& e) {
        SWIG_exception(SWIG_IndexError,const_cast<char*>(e.what()));
      }
      catch (std::invalid_argument& e) {
        SWIG_exception(SWIG_TypeError,const_cast<char*>(e.what()));
      }
    }
    %newobject __getitem__;
    RETTYPE __getitem__(const INDEXTYPE i) { return (*self).at(i); }
    %exception __setitem__ {
      try
	{
	  $action
	}
      catch (std::out_of_range& e) {
        SWIG_exception(SWIG_IndexError,const_cast<char*>(e.what()));
      }
      catch (std::invalid_argument& e) {
        SWIG_exception(SWIG_TypeError,const_cast<char*>(e.what()));
      }
    }
    void __setitem__(const INDEXTYPE i, const RETTYPE val) { (*self).at(i)=val; }
 }
#elif defined(SWIGOCTAVE)
%extend TYPE {
    %exception __brace__ {
      try
	{
	  $action
	}
      catch (std::out_of_range& e) {
        SWIG_exception(SWIG_IndexError,const_cast<char*>(e.what()));
      }
      catch (std::invalid_argument& e) {
        SWIG_exception(SWIG_TypeError,const_cast<char*>(e.what()));
      }
    }
    %newobject __brace__;
    RETTYPE __brace__(const INDEXTYPE i) { return (*self).at(i); }
    %exception __brace_asgn__ {
      try
	{
	  $action
	}
      catch (std::out_of_range& e) {
        SWIG_exception(SWIG_IndexError,const_cast<char*>(e.what()));
      }
      catch (std::invalid_argument& e) {
        SWIG_exception(SWIG_TypeError,const_cast<char*>(e.what()));
      }
    }
    void __brace_asgn__(const INDEXTYPE i, const RETTYPE val) { (*self).at(i)=val; }
 }
#elif defined(SWIGMATLAB)
%extend TYPE {
    %exception paren {
      try
	{
	  $action
	}
      catch (std::out_of_range& e) {
        SWIG_exception(SWIG_IndexError,const_cast<char*>(e.what()));
      }
      catch (std::invalid_argument& e) {
        SWIG_exception(SWIG_TypeError,const_cast<char*>(e.what()));
      }
    }
    %newobject paren;
    RETTYPE paren(const INDEXTYPE i) { return (*self).at(i); }
    %exception paren_asgn {
      try
	{
	  $action
	}
      catch (std::out_of_range& e) {
        SWIG_exception(SWIG_IndexError,const_cast<char*>(e.what()));
      }
      catch (std::invalid_argument& e) {
        SWIG_exception(SWIG_TypeError,const_cast<char*>(e.what()));
      }
    }
    void paren_asgn(const INDEXTYPE i, const RETTYPE val) { (*self).at(i)=val; }
 }
#endif
%enddef

 // Macros for adding __repr_()_ for Python and disp() for MATLAB

 // example usage: ADD_REPR(stir::ImagingModality, %arg($self->get_name()));
 // second argument piped to stream, so could be a std::string, but also another type
%define ADD_REPR(classname, defrepr)
%extend classname
{
#if defined(SWIGPYTHON_BUILTIN)
  %feature("python:slot", "tp_repr", functype="reprfunc") __repr__; 
#endif

#if defined(SWIGPYTHON)
    std::string __repr__()
    {
      std::stringstream s;
      s << "<classname::";
      s << (defrepr);
      s << ">";
      return s.str();
    }
#endif
#if defined(SWIGMATLAB)
    void disp()
    {
      std::stringstream s;
      s << "<classname::";
      s << (defrepr);
      s << ">";
      mexPrintf(repr.c_str());
    }
#endif
}
%enddef

 // use this one for classes that have parameter_info()
 // example usage: ADD_REPR_PARAMETER_INFO(stir::Radionuclide);
%define ADD_REPR_PARAMETER_INFO(classname)
  ADD_REPR(classname, "use parameter_info() for details");
%enddef

 // Finally, start with STIR specific definitions

 // General renaming of *sptr functions
%ignore *::get_scanner_sptr;
%rename (get_scanner) *::get_scanner_ptr;
%rename (get_proj_data_info) *::get_proj_data_info_sptr;
%ignore *::get_exam_info_sptr; // we do have get_exam_info in C++
%rename (set_input_proj_data) *::set_input_projdata_sptr; // warning: extra _
%rename (set_output_proj_data) *::set_output_projdata_sptr; // warning: extra _
%rename (get_output_proj_data) *::get_output_projdata_sptr;
%rename (get_output_proj_data) *::get_output_proj_data_sptr;

%rename (get_symmetries) *::get_symmetries_ptr;
%ignore *::get_symmetries_sptr;
/* would be nice, but needs swig to be compiled with PCRE support 
%rename("rstrip:[_ptr]")
%rename("rstrip:[_sptr]")
*/

%include "stir/num_threads.h"
%include "stir/find_STIR_config.h"

// #define used below to check what to do
#define STIRSWIG_SHARED_PTR

 // internally convert all pointers to shared_ptr. This prevents problems
 // with STIR functions which accept a shared_ptr as input argument.
#define SWIG_SHARED_PTR_NAMESPACE stir
#ifdef SWIGOCTAVE
 // TODO temp work-around
%include <boost_shared_ptr_test.i>
#else
%include <boost_shared_ptr.i>
#endif

%shared_ptr(stir::TimedObject);
%shared_ptr(stir::ParsingObject);

%shared_ptr(stir::Verbosity);

//  William S Fulton trick for passing templates (with commas) through macro arguments
// (already defined in swgmacros.swg)
//#define %arg(X...) X

%include "stir/TimedObject.h"
%include "stir/ParsingObject.h"
#if 0
%include "stir/Object.h"
%include "stir/RegisteredObject.h"
#else
  // use simpler version for SWIG to make the hierarchy a bit easier
  namespace stir {
    template <typename Root>
      class RegisteredObject 
      {
      public:
	//! List all possible registered names to the stream
	/*! Names are separated with newlines. */
	inline static void list_registered_names(std::ostream& stream);
      };
  }
#endif

// disabled warning about nested class. we don't need this class anyway
%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) stir::RegisteredParsingObject::RegisterIt;
// in some cases, swig generates a default constructor.
// we have to explictly disable this for RegisteredParsingObject as it could be an abstract class.
%nodefaultctor stir::RegisteredParsingObject;
%include "stir/RegisteredParsingObject.h"

 /* Parse the header files to generate wrappers */
//%include "stir/shared_ptr.h"
%include "stir/Succeeded.h"
ADD_REPR(stir::Succeeded, %arg($self->succeeded() ? "yes" : "no"));

%include "stir/NumericType.h"
%include "stir/ByteOrder.h"

%include "stir_coordinates.i"
%include "stir_LOR.i"

%include "stir_array.i"
%include "stir_exam.i"

%shared_ptr(stir::DataSymmetriesForViewSegmentNumbers);
%include "stir_projdata_coords.i"
%include "stir/DataSymmetriesForViewSegmentNumbers.h"
%include "stir_projdata.i"
%include "stir_listmode.i"

%include "stir_voxels.i"
%include "stir_voxels_IO.i"

%include "stir/ZoomOptions.h"
%include "stir/zoom.h"

%include "stir/Verbosity.h"

// shapes
%include "stir_shapes.i"

// ROIValues class and compute compute_ROI_values
%shared_ptr(stir::ROIValues)
%include "stir/evaluation/ROIValues.h"
%include "stir/evaluation/compute_ROI_values.h"

// filters
%include "stir_dataprocessors.i"

%include "stir/GeneralisedPoissonNoiseGenerator.h"

%include "stir_projectors.i"
%include "stir_normalisation.i"

%include "stir_priors.i"
%include "stir_objectivefunctions.i"
%include "stir_reconstruction.i"

%include "stir/multiply_crystal_factors.h"
%include "stir/decay_correction_factor.h"

%shared_ptr(stir::ScatterSimulation);
%shared_ptr(stir::RegisteredParsingObject<stir::SingleScatterSimulation,
  stir::ScatterSimulation, stir::ScatterSimulation>);
%shared_ptr(stir::SingleScatterSimulation);

%include "stir/scatter/ScatterSimulation.h"

%template (internalRPSingleScatterSimulation) 
  stir::RegisteredParsingObject<stir::SingleScatterSimulation,
  stir::ScatterSimulation, stir::ScatterSimulation>;
%include "stir/scatter/SingleScatterSimulation.h"

%shared_ptr(stir::ScatterEstimation);
%include "stir/scatter/ScatterEstimation.h"

%shared_ptr(stir::CreateTailMaskFromACFs);
%include "stir/scatter/CreateTailMaskFromACFs.h"

%shared_ptr(stir::FanProjData);
%shared_ptr(stir::GeoData3D);
%ignore operator<<;
%ignore operator>>;
%ignore stir::DetPairData::operator()(const int a, const int b) const;
%ignore stir::DetPairData3D::operator()(const int a, const int b) const;
%ignore stir::FanProjData::operator()(const int, const int, const int, const int) const;
%ignore stir::GeoData3D::operator()(const int, const int, const int, const int) const;
%include "stir/ML_norm.h"

%shared_ptr(stir::InvertAxis);
%include "stir/spatial_transformation/InvertAxis.h"
