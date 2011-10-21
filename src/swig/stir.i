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
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
 #include "stir/PixelsOnCartesianGrid.h"
 #include "stir/VoxelsOnCartesianGrid.h"
 %}
/* work-around for messages when using "using" in BasicCoordinate.h*/
%warnfilter(302) size_t; // suppress surprising warning about size_t redefinition (compilers don't do this)
%warnfilter(302) ptrdiff_t;
struct std::random_access_iterator_tag {};
typedef int ptrdiff_t;
typedef unsigned int size_t;

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

%include "stir/BasicCoordinate.h"
%extend BasicCoordinate {
 }
#if 1
 %include "stir/Coordinate3D.h"
#else
namespace stir {
template <typename coordT>
class Coordinate3D : public BasicCoordinate<3, coordT>
{
public:
  inline Coordinate3D();
  inline Coordinate3D(const coordT&, const coordT&, const coordT&);
  inline Coordinate3D(const BasicCoordinate<3, coordT>& c);
};
}
#endif
#if 1
%ignore  stir::CartesianCoordinate3D::z();
%ignore  stir::CartesianCoordinate3D::y();
%ignore  stir::CartesianCoordinate3D::x();
%include "stir/CartesianCoordinate3D.h"
#else
namespace stir {
template <class coordT>
class CartesianCoordinate3D : public Coordinate3D<coordT>
{
protected:
  typedef Coordinate3D<coordT> base_type;
  typedef typename base_type::base_type basebase_type;

public:
  inline CartesianCoordinate3D();
  inline CartesianCoordinate3D(const coordT&, const coordT&, const coordT&);
  inline CartesianCoordinate3D(const BasicCoordinate<3,coordT>& c);

  //inline coordT& z();
  inline coordT z() const;
  //inline coordT& y();
  inline coordT y() const;
  //inline coordT& x();
    inline coordT x() const;

};
}
#endif
%ignore *::IndexRange(const VectorWithOffset<IndexRange<num_dimensions-1> >& range);
%include "stir/IndexRange.h"
%include "stir/VectorWithOffset.h"
%include "stir/NumericVectorWithOffset.h"
    // next line probably doesn't work
    %ignore *::Array(const NumericVectorWithOffset<elemT,elemT> &il);    
    %ignore *::Array(const base_type &t);    
// ignore these as problems with num_dimensions-1
%ignore stir::Array::begin_all();
%ignore stir::Array::begin_all() const;
%ignore stir::Array::begin_all_const() const;
%ignore stir::Array::end_all();
%ignore stir::Array::end_all() const;
%ignore stir::Array::end_all_const() const;
%include "stir/Array.h"
%include "stir/DiscretisedDensity.h"
%include "stir/DiscretisedDensityOnCartesianGrid.h"

%include "stir/VoxelsOnCartesianGrid.h"

namespace stir { 
  %template(Int3BasicCoordinate) BasicCoordinate<3,int>;
  %template(Float3BasicCoordinate) BasicCoordinate<3,float>;
  %template(Float3Coordinate) Coordinate3D< float >;
  %template(FloatCartesianCoordinate3D) CartesianCoordinate3D<float>;
    %template(make_FloatCoordinate) make_coordinate<float>;
    %template(IndexRange1D) IndexRange<1>;
    //    %template(IndexRange1DVectorWithOffset) VectorWithOffset<IndexRange<1> >;
    %template(IndexRange2D) IndexRange<2>;
    //%template(IndexRange2DVectorWithOffset) VectorWithOffset<IndexRange<2> >;
    %template(IndexRange3D) IndexRange<3>;

    %template(FloatVectorWithOffset) VectorWithOffset<float>;

%template(FloatArray1D) Array<1,float>;
%template(FloatArray2D) Array<2,float>;

     %template(Float3DDiscretisedDensity) DiscretisedDensity<3,float>;
  %template(Float3DDiscretisedDensityOnCartesianGrid) DiscretisedDensityOnCartesianGrid<3,float>;
  %template(FloatVoxelsOnCartesianGrid) VoxelsOnCartesianGrid<float>;
}
