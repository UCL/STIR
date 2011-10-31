 %module stir
 %{
 /* Includes the header in the wrapper code */
#include <string>
#include <list>
#include <cstdio> // for size_t
 #include "stir/common.h"
 #include "stir/Succeeded.h"
 #include "stir/DetectionPosition.h"
 #include "stir/Scanner.h"

 #include "stir/BasicCoordinate.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange.h"
#include "stir/VectorWithOffset.h"
#include "stir/NumericVectorWithOffset.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
 #include "stir/PixelsOnCartesianGrid.h"
 #include "stir/VoxelsOnCartesianGrid.h"
 %}

%feature("autodoc", "1");

/* work-around for messages when using "using" in BasicCoordinate.h*/
%warnfilter(302) size_t; // suppress surprising warning about size_t redefinition (compilers don't do this)
%warnfilter(302) ptrdiff_t;
struct std::random_access_iterator_tag {};
typedef int ptrdiff_t;
typedef unsigned int size_t;

// disable warnings about unknown base-class
%warnfilter(401);

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

%rename(__assign__) *::operator=; 
 /* Parse the header file to generate wrappers */
%include <stl.i>
%include <std_list.i>
%include "stir/Succeeded.h"
%include "stir/DetectionPosition.h"
%include "stir/Scanner.h"

// Instantiate templates used by stir
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(FloatVector) vector<float>;
   %template(StringList) list<string>;
}
// doesn't work (yet?) because of bug in int template arguments
// %rename(__getitem__) *::operator[]; 
%define %ADD_indexaccess(RETTYPE,TYPE...)
#if defined(SWIGPYTHON)
#if defined(SWIGPYTHON_BUILTIN)
  %feature("python:slot", "sq_item", functype="ssizeargfunc") TYPE##::__getitem__;
  %feature("python:slot", "sq_ass_item", functype="ssizeobjargproc") TYPE##::__setitem__;
#endif
%extend TYPE {
        RETTYPE __getitem__(int i) { return (*self)[i]; };
	void __setitem__(int i, const RETTYPE val) { (*self)[i]=val; }
 }
#endif
%enddef

%define %ADD_indexaccessValue(TYPE...)
 //TODO cannot do this because of commas
 //%ADD_indexaccess(TYPE##::value_type, TYPE);
#if defined(SWIGPYTHON)
#if defined(SWIGPYTHON_BUILTIN)
  %feature("python:slot", "sq_item", functype="ssizeargfunc") TYPE##::__getitem__;
  %feature("python:slot", "sq_ass_item", functype="ssizeobjargproc") TYPE##::__setitem__;
#endif
%extend TYPE {
        TYPE##::value_type __getitem__(int i) { return (*self)[i]; };
	void __setitem__(int i, const TYPE##::value_type val) { (*self)[i]=val; }
 }
#endif
%enddef

%define %ADD_indexaccessReference(TYPE...)
 //TODO cannot do this because of commas
 //%ADD_indexaccess(TYPE##::reference, TYPE);
#if defined(SWIGPYTHON)
#if defined(SWIGPYTHON_BUILTIN)
  %feature("python:slot", "sq_item", functype="ssizeargfunc") TYPE##::__getitem__;
  %feature("python:slot", "sq_ass_item", functype="ssizeobjargproc") TYPE##::__setitem__;
#endif
%extend TYPE {
  TYPE##::reference __getitem__(int i) { return (*self)[i]; };
	void __setitem__(int i, const TYPE##::reference val) { (*self)[i]=val; }
 }
#endif
%enddef

%define %template_withindexaccess(NAME,RETTYPE,TYPE...)
%template(NAME) TYPE;
%ADD_indexaccess(RETTYPE,TYPE);
%enddef
%define %template_withindexaccessValue(NAME,TYPE...)
%template(NAME) TYPE;
%ADD_indexaccessValue(TYPE);
%enddef
%define %template_withindexaccessReference(NAME,TYPE...)
%template(NAME) TYPE;
%ADD_indexaccessReference(TYPE);
%enddef

%ignore stir::BasicCoordinate::operator[](const int);
%ignore stir::BasicCoordinate::operator[](const int) const;
%include "stir/BasicCoordinate.h"
 //%ADD_indexaccess(stir::BasicCoordinate::value_type,stir::BasicCoordinate);
%include "stir/Coordinate3D.h"
// ignore non-const versions
%ignore  stir::CartesianCoordinate3D::z();
%ignore  stir::CartesianCoordinate3D::y();
%ignore  stir::CartesianCoordinate3D::x();
%include "stir/CartesianCoordinate3D.h"

%ignore *::IndexRange(const VectorWithOffset<IndexRange<num_dimensions-1> >& range);
%include "stir/IndexRange.h"
%ignore stir::VectorWithOffset::operator[](int);
%ignore stir::VectorWithOffset::operator[](int) const;
%include "stir/VectorWithOffset.h"
%include "stir/NumericVectorWithOffset.h"
    // ignore a constructor that causes problems in swig 2.0.4 (and we don't need it anyway)
    // note: have to use *::Array, using stir::Array doesn't work
    %ignore *::Array(const base_type &t);    
// ignore these as problems with num_dimensions-1
%ignore stir::Array::begin_all();
%ignore stir::Array::begin_all() const;
%ignore stir::Array::begin_all_const() const;
%ignore stir::Array::end_all();
%ignore stir::Array::end_all() const;
%ignore stir::Array::end_all_const() const;

%ignore stir::Array::operator[](int) const;
%ignore stir::Array::operator[](int);
%ignore stir::Array::operator[](const BasicCoordinate<num_dimensions, int>&);
%ignore stir::Array::operator[](const BasicCoordinate<num_dimensions, int>&) const;
%ignore stir::Array::operator[](const BasicCoordinate<1, int>&);
%ignore stir::Array::operator[](const BasicCoordinate<1, int>&) const;
%include "stir/Array.h"
%include "stir/DiscretisedDensity.h"
%include "stir/DiscretisedDensityOnCartesianGrid.h"

%include "stir/VoxelsOnCartesianGrid.h"

namespace stir { 
  %template_withindexaccess(Int3BasicCoordinate,int, BasicCoordinate<3,int>);
  %template_withindexaccessValue(Float3BasicCoordinate, BasicCoordinate<3,float>);
  %template(Float3Coordinate) Coordinate3D< float >;
  %template(FloatCartesianCoordinate3D) CartesianCoordinate3D<float>;


    %template(make_FloatCoordinate) make_coordinate<float>;
  %template(IndexRange1D) IndexRange<1>;
    //    %template(IndexRange1DVectorWithOffset) VectorWithOffset<IndexRange<1> >;
    %template(IndexRange2D) IndexRange<2>;
    //%template(IndexRange2DVectorWithOffset) VectorWithOffset<IndexRange<2> >;
    %template(IndexRange3D) IndexRange<3>;

  //    %template(FloatVectorWithOffset) VectorWithOffset<float>;
  %template_withindexaccess(FloatVectorWithOffset, float, VectorWithOffset<float>);

  %template_withindexaccess(FloatArray1D,float, Array<1,float>);
%template(FloatArray2D) Array<2,float>;
%template(FloatArray3D) Array<3,float>;

     %template(Float3DDiscretisedDensity) DiscretisedDensity<3,float>;
  %template(Float3DDiscretisedDensityOnCartesianGrid) DiscretisedDensityOnCartesianGrid<3,float>;
  %template(FloatVoxelsOnCartesianGrid) VoxelsOnCartesianGrid<float>;
}
