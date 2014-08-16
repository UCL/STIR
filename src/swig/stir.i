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
 /* Include the following headers in the wrapper code */
#include <string>
#include <list>
#include <cstdio> // for size_t
#include <sstream>

#ifdef SWIGOCTAVE
// TODO terrible work-around to avoid conflict between stir::error and Octave error
// they are in conflict with eachother because we need "using namespace stir" below (swig bug)
#define __stir_error_H__
#endif

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
 #include "stir/ProjData.h"
 #include "stir/ProjDataInMemory.h"
 #include "stir/ProjDataInterfile.h"

#include "stir/CartesianCoordinate2D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/PixelsOnCartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"

#include "stir/IO/read_from_file.h"
#include "stir/IO/InterfileOutputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT7OutputFileFormat.h"
#endif

#include "stir/ChainedDataProcessor.h"
#include "stir/SeparableCartesianMetzImageFilter.h"

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h" 
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"

#include <boost/iterator/reverse_iterator.hpp>

   // TODO need this (bug in swig)
   // this bug occurs (only?) when using "%template(name) someclass;" inside the namespace
   // as opposed to "%template(name) stir::someclass" outside the namespace
   using namespace stir;
   using std::iostream;

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
         mexWarnMsgIdAndTxt("stir:array","currently only supporting double or single arrays for filling a stir array");
         mexErrMsgTxt("Exiting");
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
         mexWarnMsgIdAndTxt("stir:array",
                            "number of dimensions in matlab array is incorrect for constructing a stir array of dimension %d", 
                            num_dimensions); 
         mexErrMsgTxt("Exiting");
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
           mexWarnMsgIdAndTxt("stir:array","sizes of matlab array incompatible with stir array");
           mexErrMsgTxt("Exiting");
         }
         for (int d=1; d<= num_dimensions-matlab_num_dims; ++d)
         {
           if (sizes[d]!=1)
           {
             mexWarnMsgIdAndTxt("stir:array",
                               "sizes of first dimensions of the stir array have to be 1 if initialising from a lower dimensional matlab array");
             mexErrMsgTxt("Exiting");
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
         mexWarnMsgIdAndTxt("stir:array","currently only supporting double or single arrays for constructing a stir array");
         mexErrMsgTxt("Exiting");
       }
     }     
#endif
  } // end namespace swigstir
 %}

%feature("autodoc", "1");

// TODO doesn't work
%warnfilter(315) std::auto_ptr;

// disable warnings about unknown base-class 401
// disable warnings about "no access specified given for base class" as we use this correctly for private derivation 319
// disable warnings about nested bass-class 325
#pragma SWIG nowarn=319,401,325

# catch all C++ exceptions in python
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

#if defined(SWIGPYTHON)
%rename(__assign__) *::operator=; 
#endif

// include standard swig support for some bits of the STL (i.e. standard C++ lib)
%include <stl.i>
%include <std_list.i>
#ifdef STIRMATLAB
%ignore *::operator>>;
%ignore *::operator<<;
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
 } // end of namespace

  %}

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
    %exception getel {
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
    %newobject getel;
    RETTYPE getel(const INDEXTYPE i) { return (*self).at(i); }
    %exception setel {
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
    void setel(const INDEXTYPE i, const RETTYPE val) { (*self).at(i)=val; }
 }
#endif
%enddef

 // more or less redefinition of the above, when returning "by value"
 // but doesn't seem to work yet
%define %ADD_indexaccessValue(TYPE...)
 //TODO cannot do next line yet because of commas
 //%ADD_indexaccess(int,TYPE##::value_type, TYPE);
#if defined(SWIGPYTHON)
#if defined(SWIGPYTHON_BUILTIN)
 //  %feature("python:slot", "sq_item", functype="ssizeargfunc") TYPE##::__getitem__;
 //  %feature("python:slot", "sq_ass_item", functype="ssizeobjargproc") TYPE##::__setitem__;
%feature("python:slot", "mp_subscript", functype="binaryfunc") TYPE##::__getitem__;
%feature("python:slot", "mp_ass_subscript", functype="objobjargproc") TYPE##::__setitem__;
#endif
%extend TYPE {
        TYPE##::value_type __getitem__(int i) 
	  { 
	    return (*self)[i]; 
	  };
	void __setitem__(int i, const TYPE##::value_type val) { (*self)[i]=val; }
 }
#endif
%enddef

 // more or less redefinition of the above, when returning "by reference"
 // but doesn't seem to work yet
%define %ADD_indexaccessReference(TYPE...)
 //TODO cannot do this because of commas
 //%ADD_indexaccess(int,TYPE##::reference, TYPE);
#if defined(SWIGPYTHON)
#if defined(SWIGPYTHON_BUILTIN)
 //  %feature("python:slot", "sq_item", functype="ssizeargfunc") TYPE##::__getitem__;
 // %feature("python:slot", "sq_ass_item", functype="ssizeobjargproc") TYPE##::__setitem__;
  %feature("python:slot", "mp_subscript", functype="binaryfunc") TYPE##::__getitem__;
%feature("python:slot", "mp_ass_subscript", functype="objobjargproc") TYPE##::__setitem__;
#endif
%extend TYPE {
  TYPE##::reference __getitem__(int i) { return (*self)[i]; };
  void __setitem__(int i, const TYPE##::reference val) { (*self)[i]=val; }
 }
#endif
%enddef

 // MACROS to call the above, but also instantiate the template
%define %template_withindexaccess(NAME,RETTYPE,TYPE...)
%template(NAME) TYPE;
%ADD_indexaccess(int,RETTYPE,TYPE);
%enddef
%define %template_withindexaccessValue(NAME,TYPE...)
%template(NAME) TYPE;
%ADD_indexaccessValue(TYPE);
%enddef
%define %template_withindexaccessReference(NAME,TYPE...)
%template(NAME) TYPE;
%ADD_indexaccessReference(TYPE);
%enddef

 // Finally, start with STIR specific definitions

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
// TODO we probably need a list of other classes here
#else
namespace boost {
template<class T> class shared_ptr
{
public:
T * operator-> () const;
};
}
#endif

#if defined(SWIGPYTHON)
 // these will be replaced by __getitem__ etc
%ignore *::at(int);
#endif
// always ignore these as they are unsafe in out-of-range index access (use at() instead)
%ignore *::operator[](const int);
%ignore *::operator[](int);
%ignore *::operator[](const int) const;
%ignore *::operator[](int) const;
// always ignore const versions as for swig they're the same
%ignore *::at(int) const;

//  William S Fulton trick for passing templates (wtih commas) through macro arguments
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
// ignore these as problems with num_dimensions-1
%ignore stir::Array::begin_all();
%ignore stir::Array::begin_all() const;
%ignore stir::Array::begin_all_const() const;
%ignore stir::Array::end_all();
%ignore stir::Array::end_all() const;
%ignore stir::Array::end_all_const() const;

// need to ignore at(int) because of SWIG template bug with recursive num_dimensions
%ignore stir::Array::at(int) const;
%ignore stir::Array::at(int);

// ignore as we will use ADD_indexvalue
%ignore stir::Array::at(const BasicCoordinate<num_dimensions, int>&);
%ignore stir::Array::at(const BasicCoordinate<num_dimensions, int>&) const;
%ignore stir::Array::at(const BasicCoordinate<1, int>&);
%ignore stir::Array::at(const BasicCoordinate<1, int>&) const;

// ignore as unsafe index access (and using ADD_indexvalue)
%ignore stir::Array::operator[](const BasicCoordinate<num_dimensions, int>&);
%ignore stir::Array::operator[](const BasicCoordinate<num_dimensions, int>&) const;
%ignore stir::Array::operator[](const BasicCoordinate<1, int>&);
%ignore stir::Array::operator[](const BasicCoordinate<1, int>&) const;

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
      std::auto_ptr<stir::DiscretisedDensity<3,elemT> > 
	ret(stir::read_from_file<stir::DiscretisedDensity<3,elemT> >(filename));
      return dynamic_cast<stir::VoxelsOnCartesianGrid<elemT> *>(ret.release());
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
#endif // PYTHONCODE

  %ADD_indexaccess(int, coordT, BasicCoordinate);
  %template(Int3BasicCoordinate) BasicCoordinate<3,int>;
  %template(Size3BasicCoordinate) BasicCoordinate<3,std::size_t>;
  %template(Float3BasicCoordinate) BasicCoordinate<3,float>;
  %template(Float3Coordinate) Coordinate3D< float >;
  %template(FloatCartesianCoordinate3D) CartesianCoordinate3D<float>;

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
    swig::SwigPyIterator* flat(PyObject **PYTHON_SELF) {
      return swigstir::make_forward_iterator(self->begin_all(), self->begin_all(), self->end_all(), *PYTHON_SELF);
    }
    // add tuple indexing
    elemT __getitem__(PyObject* const args)
    {
      stir::BasicCoordinate<num_dimensions, int> c;
      if (!swigstir::coord_from_tuple(c, args))
	{
	  throw std::invalid_argument("Wrong type of indexing argument used");
	}
      return (*$self).at(c);
    };
    void __setitem__(PyObject* const args, const elemT value)
    {
      stir::BasicCoordinate<num_dimensions, int> c;
      if (!swigstir::coord_from_tuple(c, args))
	{
	  throw std::invalid_argument("Wrong type of indexing argument used");
	}
      (*$self).at(c) = value;
    };

    //! shape version returning a tuple (almost) compatible with numpy
    // use as array.shape()
    const PyObject* shape()
    {
      //const stir::BasicCoordinate<num_dimensions, std::size_t> c = (*$self).shape();
      stir::BasicCoordinate<num_dimensions,int> minind,maxind;
      if (!$self->get_regular_range(minind, maxind))
	throw std::range_error("shape called on irregular array");
      stir::BasicCoordinate<num_dimensions, int> sizes=maxind-minind+1;
      return swigstir::tuple_from_coord(sizes);
    }

  }
#endif

  %extend Array{
    // add way to know how many dimensions there are
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
// ignore non-const versions
%ignore stir::Bin::segment_num();
%ignore stir::Bin::axial_pos_num();
%ignore stir::Bin::view_num();
%ignore stir::Bin::tangential_pos_num();
%include "stir/Bin.h"
%newobject stir::ProjDataInfo::ProjDataInfoGE;
%newobject stir::ProjDataInfo::ProjDataInfoCTI;
%ignore *::get_scanner_ptr;
%include "stir/ProjDataInfo.h"
namespace stir{
  %extend ProjDataInfo
  {
    const Scanner& get_scanner() const
    { return *$self->get_scanner_ptr(); }
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

%ignore *::get_proj_data_info_ptr;
%ignore *::get_exam_info_ptr;
%rename(get_exam_info) *::get_exam_info_sptr;
%define %extend_with_proj_data_info(CLASS)
namespace stir{
  %extend CLASS
  {
    const ProjDataInfo& get_proj_data_info() const
    { return *$self->get_proj_data_info_ptr(); }
  }
 }
%enddef
%include "stir/ProjData.h"
%include "stir/ProjDataFromStream.h"
%include "stir/ProjDataInterfile.h"
%include "stir/ProjDataInMemory.h"

%extend_with_proj_data_info(ProjData);
//%extend_with_proj_data_info(MultipleProjData);
%extend_with_proj_data_info(Sinogram);
%extend_with_proj_data_info(Viewgram);
%extend_with_proj_data_info(Segment);
%extend_with_proj_data_info(RelatedViewgrams);

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

 // reconstruction
#ifdef STIRSWIG_SHARED_PTR
%shared_ptr(stir::GeneralisedObjectiveFunction<stir::DiscretisedDensity<3,float> >);
%shared_ptr(stir::PoissonLogLikelihoodWithLinearModelForMean<stir::DiscretisedDensity<3,float> >);
#define TargetT stir::DiscretisedDensity<3,float>
%shared_ptr(stir::RegisteredParsingObject<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >,
	    stir::GeneralisedObjectiveFunction<TargetT >,
	    stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT > >);
#undef TargetT

%shared_ptr(stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<stir::DiscretisedDensity<3,float> >);

%shared_ptr(stir::Reconstruction<stir::DiscretisedDensity<3,float> >);
%shared_ptr(stir::IterativeReconstruction<stir::DiscretisedDensity<3,float> >);
%shared_ptr(stir::OSMAPOSLReconstruction<stir::DiscretisedDensity<3,float> >);
#endif

%include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
%include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"

%include "stir/recon_buildblock/Reconstruction.h"
%include "stir/recon_buildblock/IterativeReconstruction.h"
%include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"


%template (GeneralisedObjectiveFunction3DFloat) stir::GeneralisedObjectiveFunction<stir::DiscretisedDensity<3,float> >;
//%template () stir::GeneralisedObjectiveFunction<stir::DiscretisedDensity<3,float> >;
%template (PoissonLogLikelihoodWithLinearModelForMean3DFloat) stir::PoissonLogLikelihoodWithLinearModelForMean<stir::DiscretisedDensity<3,float> >;

#define TargetT stir::DiscretisedDensity<3,float>
// TODO do we really need this name?
// Without it we don't see the parsing functions in python...
// Note: we cannot start it with __ as then we we get a run-time error when we're not using the builtin option
%template(RPPoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat)  stir::RegisteredParsingObject<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >,
  stir::GeneralisedObjectiveFunction<TargetT >,
  stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT > >;

%template (PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat) stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<stir::DiscretisedDensity<3,float> >;

%inline %{
  template <class T>
    stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<T> *
    ToPoissonLogLikelihoodWithLinearModelForMeanAndProjData(stir::GeneralisedObjectiveFunction<T> *b) {
    return dynamic_cast<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<T>*>(b);
}
%}

%template(ToPoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat) ToPoissonLogLikelihoodWithLinearModelForMeanAndProjData<stir::DiscretisedDensity<3,float> >;


%template (Reconstruction3DFloat) stir::Reconstruction<stir::DiscretisedDensity<3,float> >;
//%template () stir::Reconstruction<stir::DiscretisedDensity<3,float> >;
%template (IterativeReconstruction3DFloat) stir::IterativeReconstruction<stir::DiscretisedDensity<3,float> >;
//%template () stir::IterativeReconstruction<stir::DiscretisedDensity<3,float> >;
%template (OSMAPOSLReconstruction3DFloat) stir::OSMAPOSLReconstruction<stir::DiscretisedDensity<3,float> >;


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
