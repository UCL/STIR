/*
 Copyright (C) 2011-07-01 - 2012, Kris Thielemans
 Copyright (C) 2013 University College London
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.1 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 See STIR/LICENSE.txt for details
 */
/*!
 \file
 \brief Interface file for SWIG
 
 \author Kris Thielemans
 */


%module stir
%{
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
    
#include "stir/num_threads.h"
    
#include "stir/Succeeded.h"
#include "stir/DetectionPosition.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Sinogram.h"
#include "stir/SegmentByView.h"
#include "stir/SegmentBySinogram.h"
#include "stir/ExamInfo.h"
#include "stir/IO/ExamData.h"
#include "stir/Verbosity.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInMemory.h"
#include "stir/ProjDataInterfile.h"
    
#include "stir/CartesianCoordinate2D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange.h"
#include "stir/IndexRange3D.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/PixelsOnCartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
    
#include "stir/GeneralisedPoissonNoiseGenerator.h"
    
#include "stir/IO/read_from_file.h"
#include "stir/IO/InterfileOutputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT7OutputFileFormat.h"
#endif
    
#include "stir/ChainedDataProcessor.h"
#include "stir/SeparableCartesianMetzImageFilter.h"
    
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/OSSPS/OSSPSReconstruction.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
    
#include "stir/analytic/FBP2D/FBP2DReconstruction.h"
#include "stir/analytic/FBP3DRP/FBP3DRPReconstruction.h"
    
#include <boost/iterator/reverse_iterator.hpp>
#include <boost/format.hpp>
#include <stdexcept>

#include "stir/IO/write_to_file.h"
#include "stir/scatter/ScatterSimulation.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/scatter/SingleScatterLikelihoodAndGradient.h"
#include "stir/scatter/ScatterEstimation.h"
    
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
        template<> int coord_from_tuple(stir::BasicCoordinate<1, float>& c, PyObject* const args)
        { return PyArg_ParseTuple(args, "f", &c[1]);  }
        template<> int coord_from_tuple(stir::BasicCoordinate<2, float>& c, PyObject* const args)
        { return PyArg_ParseTuple(args, "ff", &c[1], &c[2]);  }
        template<> int coord_from_tuple(stir::BasicCoordinate<3, float>& c, PyObject* const args)
        { return PyArg_ParseTuple(args, "fff", &c[1], &c[2], &c[3]);  }
        
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
                throw std::runtime_error(boost::str(boost::format("number of dimensions in matlab array is incorrect for constructing a stir array of dimension %d") %
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
                throw std::runtime_error(boost::str(boost::format("number of dimensions %d of matlab array is incorrect for constructing a stir coordinate of dimension %d (expecting a column vector)") %
                                                    matlab_num_dims % num_dimensions));
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

%init %{
#if defined(SWIGPYTHON)
    // numpy support
    import_array();
#include <numpy/ndarraytypes.h>
#endif
    %}

%feature("autodoc", "1");

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
%newobject *::ask_parameters;

%ignore *::create_shared_clone;

#if defined(SWIGPYTHON)
%rename(__assign__) *::operator=;
#endif

// include standard swig support for some bits of the STL (i.e. standard C++ lib)
%include <stl.i>
%include <std_list.i>
%ignore *::begin;
%ignore *::begin_all;
%ignore *::end;
%ignore *::end_all;
%ignore *::begin_all_const;
%ignore *::end_all_const;

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
        static Array<3,float> create_array_for_proj_data(const ProjData& proj_data)
        {
            //    int num_sinos=proj_data.get_num_axial_poss(0);
            //    for (int s=1; s<= proj_data.get_max_segment_num(); ++s)
            //      {
            //        num_sinos += 2*proj_data.get_num_axial_poss(s);
            //      }
            int num_sinos = proj_data.get_num_sinograms();
            
            Array<3,float> array(IndexRange3D(num_sinos, proj_data.get_num_views(), proj_data.get_num_tangential_poss()));
            return array;
        }
        
        // a function for  converting ProjData to a 3D array as that's what is easy to use
        static Array<3,float> projdata_to_3D(const ProjData& proj_data)
        {
            Array<3,float> array = create_array_for_proj_data(proj_data);
            Array<3,float>::full_iterator array_iter = array.begin_all();
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
        void fill_proj_data_from_3D(ProjData& proj_data, const Array<3,float>& array)
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
            Array<3,float>::const_full_iterator array_iter = array.begin_all();
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

// Finally, start with STIR specific definitions

%include "stir/num_threads.h"

#if 1
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

%shared_ptr(stir::TimeFrameDefinitions);
%shared_ptr(stir::ExamInfo);
%shared_ptr(stir::ExamData);
%shared_ptr(stir::Verbosity);
%shared_ptr(stir::Scanner);
%shared_ptr(stir::ProjDataInfo);
%shared_ptr(stir::ProjDataInfoCylindrical);
%shared_ptr(stir::ProjDataInfoCylindricalArcCorr);
%shared_ptr(stir::ProjDataInfoCylindricalNoArcCorr);
%shared_ptr(stir::ProjData);
%shared_ptr(stir::ProjDataFromStream);
%shared_ptr(stir::ProjDataInterfile);
%shared_ptr(stir::ProjDataInMemory);
// TODO cannot do this yet as then the FloatArray1D(FloatArray1D&) construction fails in test.py
//%shared_ptr(stir::Array<1,float>);
%shared_ptr(stir::Array<2,float>);
%shared_ptr(stir::Array<3,float>);
%shared_ptr(stir::DiscretisedDensity<3,float>);
%shared_ptr(stir::DiscretisedDensityOnCartesianGrid<3,float>);
%shared_ptr(stir::VoxelsOnCartesianGrid<float>);
%shared_ptr(stir::SegmentBySinogram<float>);
%shared_ptr(stir::SegmentByView<float>);
%shared_ptr(stir::Segment<float>);
%shared_ptr(stir::Sinogram<float>);
%shared_ptr(stir::Viewgram<float>);
#else
namespace boost {
    template<class T> class shared_ptr
    {
    public:
        T * operator-> () const;
    };
}
#endif

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
%include "stir/NumericType.h"
%include "stir/ByteOrder.h"
%include "stir/DetectionPosition.h"
%newobject stir::Scanner::get_scanner_from_name;
%include "stir/Scanner.h"

/* First do coordinates, indices, images.
 We first include them, and sort out template instantiation and indexing below.
 */
//%include <boost/operators.hpp>

%include "stir/BasicCoordinate.h"
%include "stir/Coordinate3D.h"
// ignore non-const versions
%ignore  stir::CartesianCoordinate3D::z();
%ignore  stir::CartesianCoordinate3D::y();
%ignore  stir::CartesianCoordinate3D::x();
%include "stir/CartesianCoordinate3D.h"
%include "stir/Coordinate2D.h"
// ignore const versions
%ignore  stir::CartesianCoordinate2D::x() const;
%ignore  stir::CartesianCoordinate2D::y() const;
%include "stir/CartesianCoordinate2D.h"

// we have to ignore the following because of a bug in SWIG 2.0.4, but we don't need it anyway
%ignore *::IndexRange(const VectorWithOffset<IndexRange<num_dimensions-1> >& range);
%include "stir/IndexRange.h"

%ignore stir::VectorWithOffset::get_const_data_ptr() const;
%ignore stir::VectorWithOffset::get_data_ptr();
%ignore stir::VectorWithOffset::release_const_data_ptr() const;
%ignore stir::VectorWithOffset::release_data_ptr();
%include "stir/VectorWithOffset.h"

#if defined(SWIGPYTHON)
// TODO ideally would use %swig_container_methods but we don't have getslice yet
#if defined(SWIGPYTHON_BUILTIN)
%feature("python:slot", "nb_nonzero", functype="inquiry") __nonzero__;
%feature("python:slot", "sq_length", functype="lenfunc") __len__;
#endif // SWIGPYTHON_BUILTIN

%extend stir::VectorWithOffset {
    bool __nonzero__() const {
        return !(self->empty());
    }
    
    /* Alias for Python 3 compatibility */
    bool __bool__() const {
        return !(self->empty());
    }
    
    size_type __len__() const {
        return self->size();
    }
#if 0
    // TODO this does not work yet
    /*
     %define %emit_swig_traits(_Type...)
     %traits_swigtype(_Type);
     %fragment(SWIG_Traits_frag(_Type));
     %enddef
     
     %emit_swig_traits(MyPrice)
     */
    %swig_sequence_iterator(stir::VectorWithOffset<T>);
#else
    %newobject _iterator(PyObject **PYTHON_SELF);
    swig::SwigPyIterator* _iterator(PyObject **PYTHON_SELF) {
        return swig::make_output_iterator(self->begin(), self->begin(), self->end(), *PYTHON_SELF);
    }
#if defined(SWIGPYTHON_BUILTIN)
    %feature("python:slot", "tp_iter", functype="getiterfunc") _iterator;
#else
    %pythoncode {def __iter__(self): return self._iterator()}
#endif
#endif
}
#endif

%include "stir/NumericVectorWithOffset.h"

#ifdef SWIGPYTHON
// ignore as we will add a version that returns a tuple instead
%ignore stir::Array::shape() const;
#endif

%include "stir/Array.h"

// ignore this one and add it later (see below)
%ignore stir::DiscretisedDensity::read_from_file(const std::string& filename);
%include "stir/DiscretisedDensity.h"
%include "stir/DiscretisedDensityOnCartesianGrid.h"

%include "stir/VoxelsOnCartesianGrid.h"

%extend stir::VoxelsOnCartesianGrid {
    // add read_from_file to this class, as currently there is no way
    // to convert the swigged DiscretisedDensity to a VoxelsOnCartesianGrid
    static stir::VoxelsOnCartesianGrid<elemT> * read_from_file(const std::string& filename)
    {
        using namespace stir;
        unique_ptr<DiscretisedDensity<3,elemT> >
        ret(read_from_file<DiscretisedDensity<3,elemT> >(filename));
        return dynamic_cast<VoxelsOnCartesianGrid<elemT> *>(ret.release());
    }
}

//%ADD_indexaccess(int,stir::BasicCoordinate::value_type,stir::BasicCoordinate);
namespace stir {
#ifdef SWIGPYTHON
    // add extra features to the coordinates to make them a bit more Python friendly
    %extend BasicCoordinate {
        //%feature("autodoc", "construct from tuple, e.g. (2,3,4) for a 3d coordinate")
        BasicCoordinate(PyObject* args)
        {
            BasicCoordinate<num_dimensions,coordT> *c=new BasicCoordinate<num_dimensions,coordT>;
            if (!swigstir::coord_from_tuple(*c, args))
            {
                throw std::invalid_argument("Wrong type of argument to construct Coordinate used");
            }
            return c;
        };
        
        // print as (1,2,3) as opposed to non-informative default provided by SWIG
        std::string __str__()
        {
            std::ostringstream s;
            s<<'(';
            for (int d=1; d<=num_dimensions-1; ++d)
                s << (*$self)[d] << ", ";
            s << (*$self)[num_dimensions] << ')';
            return s.str();
        }
        
        // print as classname((1,2,3)) as opposed to non-informative default provided by SWIG
        std::string __repr__()
        {
#if SWIG_VERSION < 0x020009
            // don't know how to get the Python typename
            std::string repr = "stir.Coordinate";
#else
            std::string repr = "$parentclasssymname";
#endif
            // TODO attempt to re-use __str__ above, but it doesn't compile, so we replicate the code
            // repr += $self->__str__() + ')';
            std::ostringstream s;
            s<<"((";
            for (int d=1; d<=num_dimensions-1; ++d)
                s << (*$self)[d] << ", ";
            s << (*$self)[num_dimensions] << ')';
            repr += s.str() + ")";
            return repr;
        }
        
        bool __nonzero__() const {
            return true;
        }
        
        /* Alias for Python 3 compatibility */
        bool __bool__() const {
            return true;
        }
        
        size_type __len__() const {
            return $self->size();
        }
#if defined(SWIGPYTHON_BUILTIN)
        %feature("python:slot", "tp_str", functype="reprfunc") __str__;
        %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
        %feature("python:slot", "nb_nonzero", functype="inquiry") __nonzero__;
        %feature("python:slot", "sq_length", functype="lenfunc") __len__;
#endif // SWIGPYTHON_BUILTIN
        
    }
#elif defined(SWIGMATLAB)
    %extend BasicCoordinate {
        // print as [1;2;3] as opposed to non-informative default provided by SWIG
        void disp()
        {
            std::ostringstream s;
            s<<'[';
            for (int d=1; d<=num_dimensions-1; ++d)
                s << (*$self)[d] << "; ";
            s << (*$self)[num_dimensions] << "]\n";
            mexPrintf(s.str().c_str());
            
        }
        //%feature("autodoc", "construct from vector, e.g. [2;3;4] for a 3d coordinate")
        BasicCoordinate(const mxArray *pm)
        {
            $parentclassname * array_ptr = new $parentclassname();
            swigstir::fill_BasicCoordinate_from_matlab(*array_ptr, pm);
            return array_ptr;
        }
        
        %newobject to_matlab;
        mxArray * to_matlab()
        { return swigstir::BasicCoordinate_to_matlab(*$self); }
        
        void fill(const mxArray *pm)
        { swigstir::fill_BasicCoordinate_from_matlab(*$self, pm); }
    }
#endif // PYTHON, MATLAB extension of BasicCoordinate
    
    %ADD_indexaccess(int, coordT, BasicCoordinate);
    %template(Int3BasicCoordinate) BasicCoordinate<3,int>;
    %template(Size3BasicCoordinate) BasicCoordinate<3,std::size_t>;
    %template(Float3BasicCoordinate) BasicCoordinate<3,float>;
    %template(Float3Coordinate) Coordinate3D< float >;
    %template(FloatCartesianCoordinate3D) CartesianCoordinate3D<float>;
    %template(IntCartesianCoordinate3D) CartesianCoordinate3D<int>;
    
    %template(Int2BasicCoordinate) BasicCoordinate<2,int>;
    %template(Size2BasicCoordinate) BasicCoordinate<2,std::size_t>;
    %template(Float2BasicCoordinate) BasicCoordinate<2,float>;
    // TODO not needed in python case?
    %template(Float2Coordinate) Coordinate2D< float >;
    %template(FloatCartesianCoordinate2D) CartesianCoordinate2D<float>;
    
    //#ifndef SWIGPYTHON
    // not necessary for Python as we can use tuples there
    %template(make_IntCoordinate) make_coordinate<int>;
    %template(make_FloatCoordinate) make_coordinate<float>;
    //#endif
    
    %template(IndexRange1D) IndexRange<1>;
    //    %template(IndexRange1DVectorWithOffset) VectorWithOffset<IndexRange<1> >;
    %template(IndexRange2D) IndexRange<2>;
    //%template(IndexRange2DVectorWithOffset) VectorWithOffset<IndexRange<2> >;
    %template(IndexRange3D) IndexRange<3>;
    
    %ADD_indexaccess(int,T,VectorWithOffset);
    %template(FloatVectorWithOffset) VectorWithOffset<float>;
    
    // TODO need to instantiate with name?
    %template (FloatNumericVectorWithOffset) NumericVectorWithOffset<float, float>;
    
#ifdef SWIGPYTHON
    // TODO this extends ND-Arrays, but apparently not 1D Arrays (because specialised template?)
    %extend Array{
        // add "flat" iterator, using begin_all()
        %newobject flat(PyObject **PYTHON_SELF);
        %feature("autodoc", "create a Python iterator over all elements, e.g. array.flat()") flat;
        swig::SwigPyIterator* flat(PyObject **PYTHON_SELF) {
            return swigstir::make_forward_iterator(self->begin_all(), self->begin_all(), self->end_all(), *PYTHON_SELF);
        }
        %feature("autodoc", "tuple indexing, e.g. array[(1,2,3)]") __getitem__;
        elemT __getitem__(PyObject* const args)
        {
            stir::BasicCoordinate<num_dimensions, int> c;
            if (!swigstir::coord_from_tuple(c, args))
            {
                throw std::invalid_argument("Wrong type of indexing argument used");
            }
            return (*$self).at(c);
        };
        %feature("autodoc", "tuple indexing, e.g. array[(1,2,3)]=4") __setitem__;
        void __setitem__(PyObject* const args, const elemT value)
        {
            stir::BasicCoordinate<num_dimensions, int> c;
            if (!swigstir::coord_from_tuple(c, args))
            {
                throw std::invalid_argument("Wrong type of indexing argument used");
            }
            (*$self).at(c) = value;
        };
        
        %feature("autodoc", "return number of elements per dimension as a tuple, (almost) compatible with numpy. Use as array.shape()") shape;
        const PyObject* shape()
        {
            //const stir::BasicCoordinate<num_dimensions, std::size_t> c = (*$self).shape();
            stir::BasicCoordinate<num_dimensions,int> minind,maxind;
            if (!$self->get_regular_range(minind, maxind))
                throw std::range_error("shape called on irregular array");
            stir::BasicCoordinate<num_dimensions, int> sizes=maxind-minind+1;
            return swigstir::tuple_from_coord(sizes);
        }
        
        %feature("autodoc", "fill from a Python iterator, e.g. array.fill(numpyarray.flat)") fill;
        void fill(PyObject* const arg)
        {
            if (PyIter_Check(arg))
            {
                swigstir::fill_Array_from_Python_iterator($self, arg);
            }
            else
            {
                char str[1000];
                snprintf(str, 1000, "Wrong argument-type used for fill(): should be a scalar or an iterator or so, but is of type %s",
                         arg->ob_type->tp_name);
                throw std::invalid_argument(str);
            }
        }
    }
#endif
    
    %extend Array{
        %feature("autodoc", "return number of dimensions in the array") get_num_dimensions;
        int get_num_dimensions()
        {
            return num_dimensions;
        }
    }
    
#ifdef SWIGMATLAB
    %extend Array {
        Array(const mxArray *pm)
        {
            $parentclassname * array_ptr = new $parentclassname();
            swigstir::fill_Array_from_matlab(*array_ptr, pm, true /* do resize */);
            return array_ptr;
        }
        
        %newobject to_matlab;
        mxArray * to_matlab()
        { return swigstir::Array_to_matlab(*$self); }
        
        void fill(const mxArray *pm)
        { swigstir::fill_Array_from_matlab(*$self, pm, false /*do not resize */); }
    }
    // repeat this for 1D due to template (partial) specialisation (TODO, get round that somehow)
    %extend Array<1,float> {
        Array<1,float>(const mxArray *pm)
        {
            $parentclassname * array_ptr = new $parentclassname();
            swigstir::fill_Array_from_matlab(*array_ptr, pm, true /* do resize */);
            return array_ptr;
        }
        
        %newobject to_matlab;
        mxArray * to_matlab()
        { return swigstir::Array_to_matlab(*$self); }
        
        void fill(const mxArray *pm)
        { swigstir::fill_Array_from_matlab(*$self, pm, false /*do not resize */); }
    }
#endif
    // TODO next line doesn't give anything useful as SWIG doesn't recognise that
    // the return value is an array. So, we get a wrapped object that we cannot handle
    //%ADD_indexaccess(int,Array::value_type, Array);
    
    %ADD_indexaccess(%arg(const BasicCoordinate<num_dimensions,int>&),elemT, Array);
    
    %template(FloatArray1D) Array<1,float>;
    
    // this doesn't work because of bug in swig (incorrectly parses num_dimensions)
    //%ADD_indexaccess(int,%arg(Array<num_dimensions -1,elemT>), Array);
    // In any case, even if the above is made to work (e.g. explicit override for every class as below)
    //  then setitem still doesn't modify the object for more than 1 level
#if 1
    // note: next line has no memory allocation problems because all Array<1,...> objects
    // are auto-converted to shared_ptrs.
    // however, cannot use setitem to modify so ideally we would define getitem only (at least for python) (TODO)
    // TODO DISABLE THIS
    %ADD_indexaccess(int,%arg(Array<1,float>),%arg(Array<2,float>));
#endif
    
} // namespace stir

%rename (get_scanner) *::get_scanner_ptr;
%ignore *::get_proj_data_info_ptr;
%rename (get_proj_data_info) *::get_proj_data_info_sptr;
%ignore *::get_exam_info_ptr;
%rename (get_exam_info) *::get_exam_info_sptr;

%rename (set_objective_function) *::set_objective_function_sptr;

// Todo need to instantiate with name?
// TODO Swig doesn't see that Array<2,float> is derived from it anyway becuse of num_dimensions bug
%template (FloatNumericVectorWithOffset2D) stir::NumericVectorWithOffset<stir::Array<1,float>, float>;

%template(FloatArray2D) stir::Array<2,float>;
// TODO name
%template (FloatNumericVectorWithOffset3D) stir::NumericVectorWithOffset<stir::Array<2,float>, float>;
%template(FloatArray3D) stir::Array<3,float>;
#if 0
%ADD_indexaccess(int,%arg(stir::Array<2,float>),%arg(stir::Array<3,float>));
#endif

%template(Float3DDiscretisedDensity) stir::DiscretisedDensity<3,float>;
%template(Float3DDiscretisedDensityOnCartesianGrid) stir::DiscretisedDensityOnCartesianGrid<3,float>;
//%template() stir::DiscretisedDensity<3,float>;
//%template() stir::DiscretisedDensityOnCartesianGrid<3,float>;
%template(FloatVoxelsOnCartesianGrid) stir::VoxelsOnCartesianGrid<float>;

%include "stir/IO/write_to_file.h"
%template(write_image_to_file) stir::write_to_file<DiscretisedDensity<3, float> >;

#ifdef STIRSWIG_SHARED_PTR
#define DataT stir::DiscretisedDensity<3,float>
%shared_ptr(stir::OutputFileFormat<stir::DiscretisedDensity<3,float> >);
%shared_ptr(stir::RegisteredObject< stir::OutputFileFormat< stir::DiscretisedDensity< 3,float > > >);
%shared_ptr(stir::RegisteredParsingObject< stir::InterfileOutputFileFormat, stir::OutputFileFormat<DataT >, stir::OutputFileFormat<DataT > >);
#undef DataT
%shared_ptr(stir::InterfileOutputFileFormat);
#ifdef HAVE_LLN_MATRIX
%shared_ptr(stir::RegisteredParsingObject<stir::ecat::ecat7::ECAT7OutputFileFormat, stir::OutputFileFormat<DataT >, stir::OutputFileFormat<DataT > >);
%shared_ptr(stir::ecat::ecat7::ECAT7OutputFileFormat);
#endif
#endif

%include "stir/IO/OutputFileFormat.h"

#define DataT stir::DiscretisedDensity<3,float>
%template(Float3DDiscretisedDensityOutputFileFormat) stir::OutputFileFormat<DataT >;
//cannot do that as pure virtual functions
//%template(ROOutputFileFormat3DFloat) RegisteredObject< OutputFileFormat< DiscretisedDensity< 3,float > > >;
%template(RPInterfileOutputFileFormat) stir::RegisteredParsingObject<stir::InterfileOutputFileFormat, stir::OutputFileFormat<DataT >, stir::OutputFileFormat<DataT > >;
#undef DataT

%include "stir/IO/InterfileOutputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
%include "stir/IO/ECAT7OutputFileFormat.h"
#endif

/* Now do ProjDataInfo, Sinogram et al
 */
%include "stir/TimeFrameDefinitions.h"
%include "stir/ExamInfo.h"
%include "stir/IO/ExamData.h"
%include "stir/Verbosity.h"

%attributeref(stir::Bin, int, segment_num);
%attributeref(stir::Bin, int, axial_pos_num);
%attributeref(stir::Bin, int, view_num);
%attributeref(stir::Bin, int, tangential_pos_num);
%attribute(stir::Bin, float, bin_value, get_bin_value, set_bin_value);
%include "stir/Bin.h"

%newobject stir::ProjDataInfo::ProjDataInfoGE;
%newobject stir::ProjDataInfo::ProjDataInfoCTI;

// ignore this to avoid problems with unique_ptr, and add it later
%ignore stir::ProjDataInfo::construct_proj_data_info;

%include "stir/ProjDataInfo.h"
%newobject *::construct_proj_data_info;

%extend stir::ProjDataInfo
{
    // work around the current SWIG limitation that it doesn't wrap unique_ptr.
    // we do this with the crazy (and ugly) way to let SWIG create a new function
    // which is the same as the original, but returns a bare pointer.
    // (This will be wrapped as a shared_ptr in the end).
    // This work-around is fragile however as it depends on knowledge of the
    // exact signature of the function.
    static ProjDataInfo *
    construct_proj_data_info(const shared_ptr<Scanner>& scanner_sptr,
                             const int span, const int max_delta,
                             const int num_views, const int num_tangential_poss,
                             const bool arc_corrected = true)
    {
        return
        construct_proj_data_info(scanner_sptr,
                                 span, max_delta, num_views, num_tangential_poss,
                                 arc_corrected).get();
    }
}
%include "stir/ProjDataInfoCylindrical.h"
%include "stir/ProjDataInfoCylindricalArcCorr.h"
%include "stir/ProjDataInfoCylindricalNoArcCorr.h"

%include "stir/Viewgram.h"
%include "stir/RelatedViewgrams.h"
%include "stir/Sinogram.h"
%include "stir/Segment.h"
%include "stir/SegmentByView.h"
%include "stir/SegmentBySinogram.h"

%include "stir/ProjData.h"

namespace stir {
    %extend ProjData
    {
#ifdef SWIGPYTHON
        %feature("autodoc", "create a stir 3D Array from the projection data (internal)") to_array;
        %newobject to_array;
        Array<3,float> to_array()
        {
            Array<3,float> array = swigstir::projdata_to_3D(*$self);
            return array;
        }
        
        %feature("autodoc", "fill from a Python iterator, e.g. proj_data.fill(numpyarray.flat)") fill;
        void fill(PyObject* const arg)
        {
            if (PyIter_Check(arg))
            {
                Array<3,float> array = swigstir::create_array_for_proj_data(*$self);
                swigstir::fill_Array_from_Python_iterator(&array, arg);
                swigstir::fill_proj_data_from_3D(*$self, array);
            }
            else
            {
                char str[1000];
                snprintf(str, 1000, "Wrong argument-type used for fill(): should be a scalar or an iterator or so, but is of type %s",
                         arg->ob_type->tp_name);
                throw std::invalid_argument(str);
            }
        }
        
#elif defined(SWIGMATLAB)
        %newobject to_matlab;
        mxArray * to_matlab()
        {
            Array<3,float> array = swigstir::projdata_to_3D(*$self);
            return swigstir::Array_to_matlab(array);
        }
        
        void fill(const mxArray *pm)
        {
            Array<3,float> array;
            swigstir::fill_Array_from_matlab(array, pm, true);
            swigstir::fill_proj_data_from_3D(*$self, array);
        }
#endif
    }
}

%include "stir/ProjDataFromStream.h"
%include "stir/ProjDataInterfile.h"
%include "stir/ProjDataInMemory.h"

namespace stir {
    %template(FloatViewgram) Viewgram<float>;
    %template(FloatSinogram) Sinogram<float>;
    // TODO don't want to give a name
    %template(FloatSegment) Segment<float>;
    %template(FloatSegmentBySinogram) SegmentBySinogram<float>;
    %template(FloatSegmentByView) SegmentByView<float>;
    // should not have the following if using boost_smart_ptr.i
    //  %template(SharedScanner) boost::shared_ptr<Scanner>;
    //%template(SharedProjData) boost::shared_ptr<ProjData>;
    
}

// filters
#ifdef STIRSWIG_SHARED_PTR
#define elemT float
%shared_ptr(stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >)
%shared_ptr(stir::RegisteredParsingObject<
            stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >,
            stir::DataProcessor<DiscretisedDensity<3,elemT> >,
            stir::DataProcessor<DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >)
%shared_ptr(stir::RegisteredParsingObject<stir::SeparableCartesianMetzImageFilter<elemT>,
            stir::DataProcessor<DiscretisedDensity<3,elemT> >,
            stir::DataProcessor<DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::SeparableCartesianMetzImageFilter<elemT>)
#undef elemT
#endif

%include "stir/DataProcessor.h"
%include "stir/ChainedDataProcessor.h"
%include "stir/SeparableCartesianMetzImageFilter.h"

#define elemT float
%template(DataProcessor3DFloat) stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >;
%template(RPChainedDataProcessor3DFloat) stir::RegisteredParsingObject<
stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >,
stir::DataProcessor<DiscretisedDensity<3,elemT> >,
stir::DataProcessor<DiscretisedDensity<3,elemT> > >;
%template(ChainedDataProcessor3DFloat) stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >;
%template(RPSeparableCartesianMetzImageFilter3DFloat) stir::RegisteredParsingObject<
stir::SeparableCartesianMetzImageFilter<elemT>,
stir::DataProcessor<DiscretisedDensity<3,elemT> >,
stir::DataProcessor<DiscretisedDensity<3,elemT> > >;

%template(SeparableCartesianMetzImageFilter3DFloat) stir::SeparableCartesianMetzImageFilter<elemT>;
#undef elemT

%include "stir/GeneralisedPoissonNoiseGenerator.h"

// reconstruction
#ifdef STIRSWIG_SHARED_PTR
#define TargetT stir::DiscretisedDensity<3,float>

%shared_ptr(stir::GeneralisedObjectiveFunction<TargetT >);
%shared_ptr(stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT >);
%shared_ptr(stir::RegisteredParsingObject<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >,
            stir::GeneralisedObjectiveFunction<TargetT >,
            stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT > >);

%shared_ptr(stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >);

%shared_ptr(stir::Reconstruction<TargetT >);
%shared_ptr(stir::IterativeReconstruction<TargetT >);

%shared_ptr(stir::RegisteredParsingObject<
            stir::OSMAPOSLReconstruction <TargetT > ,
            stir::Reconstruction < TargetT >,
            stir::IterativeReconstruction < TargetT >
            >)
%shared_ptr(stir::RegisteredParsingObject<
            stir::OSSPSReconstruction <TargetT > ,
            stir::Reconstruction < TargetT >,
            stir::IterativeReconstruction < TargetT >
            >)

%shared_ptr(stir::OSMAPOSLReconstruction<TargetT >);
%shared_ptr(stir::OSSPSReconstruction<TargetT >);
%shared_ptr(stir::AnalyticReconstruction);
%shared_ptr(stir::FBP2DReconstruction);
%shared_ptr(stir::FBP3DRPReconstruction);

#undef TargetT

#endif

%include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
%include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"

%include "stir/recon_buildblock/Reconstruction.h"
// there's a get_objective_function, so we'll ignore the sptr version
%ignore *::get_objective_function_sptr;
%include "stir/recon_buildblock/IterativeReconstruction.h"
%include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
%include "stir/OSSPS/OSSPSReconstruction.h"

%include "stir/recon_buildblock/AnalyticReconstruction.h"
%include "stir/analytic/FBP2D/FBP2DReconstruction.h"
%include "stir/analytic/FBP3DRP/FBP3DRPReconstruction.h"

#define TargetT stir::DiscretisedDensity<3,float>


%template (GeneralisedObjectiveFunction3DFloat) stir::GeneralisedObjectiveFunction<TargetT >;
//%template () stir::GeneralisedObjectiveFunction<TargetT >;
%template (PoissonLogLikelihoodWithLinearModelForMean3DFloat) stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT >;

// TODO do we really need this name?
// Without it we don't see the parsing functions in python...
// Note: we cannot start it with __ as then we we get a run-time error when we're not using the builtin option
%template(RPPoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat)  stir::RegisteredParsingObject<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >,
stir::GeneralisedObjectiveFunction<TargetT >,
stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT > >;

%template (PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat) stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >;

%inline %{
    template <class T>
    stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<T> *
    ToPoissonLogLikelihoodWithLinearModelForMeanAndProjData(stir::GeneralisedObjectiveFunction<T> *b) {
        return dynamic_cast<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<T>*>(b);
    }
    %}

%template(ToPoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat) ToPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >;


%template (Reconstruction3DFloat) stir::Reconstruction<TargetT >;
//%template () stir::Reconstruction<TargetT >;
%template (IterativeReconstruction3DFloat) stir::IterativeReconstruction<TargetT >;
//%template () stir::IterativeReconstruction<TargetT >;

%template (RPOSMAPOSLReconstruction3DFloat) stir::RegisteredParsingObject<
stir::OSMAPOSLReconstruction <TargetT > ,
stir::Reconstruction < TargetT >,
stir::IterativeReconstruction < TargetT >
>;
%template (RPOSSPSReconstruction) stir::RegisteredParsingObject<
stir::OSSPSReconstruction <TargetT > ,
stir::Reconstruction < TargetT >,
stir::IterativeReconstruction < TargetT >
>;

%template (OSMAPOSLReconstruction3DFloat) stir::OSMAPOSLReconstruction<TargetT >;
%template (OSSPSReconstruction3DFloat) stir::OSSPSReconstruction<TargetT >;

#undef TargetT

/// projectors
%shared_ptr(stir::ForwardProjectorByBin);
%shared_ptr(stir::RegisteredParsingObject<stir::ForwardProjectorByBinUsingProjMatrixByBin,
            stir::ForwardProjectorByBin>);
%shared_ptr(stir::ForwardProjectorByBinUsingProjMatrixByBin);
%shared_ptr(stir::BackProjectorByBin);
%shared_ptr(stir::RegisteredParsingObject<stir::BackProjectorByBinUsingProjMatrixByBin,
            stir::BackProjectorByBin>);
%shared_ptr(stir::BackProjectorByBinUsingProjMatrixByBin);
%shared_ptr(stir::ProjMatrixByBin);
%shared_ptr(stir::RegisteredParsingObject<
            stir::ProjMatrixByBinUsingRayTracing,
            stir::ProjMatrixByBin,
            stir::ProjMatrixByBin
            >);
%shared_ptr(stir::ProjMatrixByBinUsingRayTracing);

%include "stir/recon_buildblock/ForwardProjectorByBin.h"
%include "stir/recon_buildblock/BackProjectorByBin.h"

%include "stir/recon_buildblock/ProjMatrixByBin.h"

%template (internalRPProjMatrixByBinUsingRayTracing) stir::RegisteredParsingObject<
stir::ProjMatrixByBinUsingRayTracing,
stir::ProjMatrixByBin,
stir::ProjMatrixByBin
>;

%include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"

%shared_ptr(  stir::AddParser<stir::ForwardProjectorByBin>);
%template (internalAddParserForwardProjectorByBin)
stir::AddParser<stir::ForwardProjectorByBin>;
%template (internalRPForwardProjectorByBinUsingProjMatrixByBin)
stir::RegisteredParsingObject<stir::ForwardProjectorByBinUsingProjMatrixByBin,
stir::ForwardProjectorByBin>;
%include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"

%shared_ptr(  stir::AddParser<stir::BackProjectorByBin>);
%template (internalAddParserBackProjectorByBin)
stir::AddParser<stir::BackProjectorByBin>;
%template (internalRPBackProjectorByBinUsingProjMatrixByBin)
stir::RegisteredParsingObject<stir::BackProjectorByBinUsingProjMatrixByBin,
stir::BackProjectorByBin>;
%include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"



//scatter
%shared_ptr(stir::ScatterSimulation);

%shared_ptr(stir::RegisteredParsingObject<
            stir::SingleScatterSimulation,
            stir::ScatterSimulation,
            stir::ScatterSimulation
            >);

%shared_ptr(stir::SingleScatterSimulation);

shared_ptr(stir::RegisteredParsingObject<
            stir::SingleScatterLikelihoodAndGradient,
            stir::SingleScatterSimulation,
            stir::SingleScatterSimulation
            >);

%shared_ptr(stir::SingleScatterLikelihoodAndGradient);


%include "stir/scatter/ScatterSimulation.h"
%include "stir/scatter/SingleScatterSimulation.h"

%include "stir/scatter/SingleScatterLikelihoodAndGradient.h"

%template (internalRPSingleScatterSimulation) stir::RegisteredParsingObject<
stir::SingleScatterSimulation,
stir::ScatterSimulation,
stir::ScatterSimulation
>;



%shared_ptr(stir::ScatterEstimation);
%shared_ptr(stir::ParsingObject);
%include "stir/scatter/ScatterEstimation.h"


