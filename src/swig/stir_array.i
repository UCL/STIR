/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2013, 2014, 2015, 2018 - 2022 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::Array etc

  \author Kris Thielemans
*/

// TODO cannot do this yet as then the FloatArray1D(FloatArray1D&) construction fails in test.py
//%shared_ptr(stir::Array<1,float>);
%shared_ptr(stir::Array<2,float>);
%shared_ptr(stir::Array<3,float>);
%shared_ptr(stir::Array<4,float>);

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

%ignore stir::NumericVectorWithOffset::xapyb;
%ignore stir::NumericVectorWithOffset::axpby;
%include "stir/NumericVectorWithOffset.h"

#ifdef SWIGPYTHON
// ignore as we will add a version that returns a tuple instead
%ignore stir::Array::shape() const;
#endif

%include "stir/Array.h"

namespace stir
{
  %template(IndexRange1D) IndexRange<1>;
  //    %template(IndexRange1DVectorWithOffset) VectorWithOffset<IndexRange<1> >;
  %template(IndexRange2D) IndexRange<2>;
  //%template(IndexRange2DVectorWithOffset) VectorWithOffset<IndexRange<2> >;
  %template(IndexRange3D) IndexRange<3>;
  %template(IndexRange4D) IndexRange<4>;

  %ADD_indexaccess(int,T,VectorWithOffset);
  %template(FloatVectorWithOffset) VectorWithOffset<float>;
  %template(IntVectorWithOffset) VectorWithOffset<int>;

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
  // are auto-converted to _ptrs.
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
  %template (FloatNumericVectorWithOffset4D) stir::NumericVectorWithOffset<stir::Array<3,float>, float>;
  %template(FloatArray4D) stir::Array<4,float>;
