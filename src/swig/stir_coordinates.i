/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2013 - 2015, 2022 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::Array etc

  \author Kris Thielemans
*/

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
  
  %template(Int4BasicCoordinate) BasicCoordinate<4,int>;
  %template(Size4BasicCoordinate) BasicCoordinate<4,std::size_t>;
  %template(Float4BasicCoordinate) BasicCoordinate<4,float>;
  // TODO not needed in python case?
  %template(Float2Coordinate) Coordinate2D< float >;
  %template(FloatCartesianCoordinate2D) CartesianCoordinate2D<float>;

  //#ifndef SWIGPYTHON
  // not necessary for Python as we can use tuples there
  %template(make_IntCoordinate) make_coordinate<int>;
  %template(make_FloatCoordinate) make_coordinate<float>;
  //#endif

}
