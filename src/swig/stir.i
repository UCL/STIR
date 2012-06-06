// $Id$
/*
    Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
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

  $Date$

  $Revision$

*/


 %module stir
 %{
 /* Include the following headers in the wrapper code */
#include <string>
#include <list>
#include <cstdio> // for size_t

 #include "boost/shared_ptr.hpp"
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
 #include "stir/ProjData.h"
 #include "stir/ProjDataInterfile.h"

 #include "stir/BasicCoordinate.h"
#include "stir/CartesianCoordinate2D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange.h"
#include "stir/VectorWithOffset.h"
#include "stir/NumericVectorWithOffset.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
 #include "stir/PixelsOnCartesianGrid.h"
 #include "stir/VoxelsOnCartesianGrid.h"

#include "stir/ChainedDataProcessor.h"
#include "stir/SeparableCartesianMetzImageFilter.h"

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h" 
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"

   // TODO seem to need this for using shared_ptr (bug in swig?)
   using namespace stir;
   using std::iostream;
 %}

%feature("autodoc", "1");

/* work-around for messages when using "using" in BasicCoordinate.h*/
%warnfilter(302) size_t; // suppress surprising warning about size_t redefinition (compilers don't do this)
%warnfilter(302) ptrdiff_t;
struct std::random_access_iterator_tag {};
typedef int ptrdiff_t;
typedef unsigned int size_t;

// TODO doesn't work
%warnfilter(315) std::auto_ptr;

// disable warnings about unknown base-class
%warnfilter(401);

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

#if defined(SWIGPYTHON)
%rename(__assign__) *::operator=; 
#endif

// include standard swig support for some bits of the STL (i.e. standard C++ lib)
%include <stl.i>
%include <std_list.i>
%include <std_ios.i>
%include <std_iostream.i>

// Instantiate STL templates used by stir
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(FloatVector) vector<float>;
   %template(StringList) list<string>;
}

// doesn't work (yet?) because of bug in int template arguments
// %rename(__getitem__) *::operator[]; 

// MACROS to define index access (Work-in-progress)
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

 // more or less redefinition of the above, when returning "by value"
 // but doesn't seem to work yet
%define %ADD_indexaccessValue(TYPE...)
 //TODO cannot do next line yet because of commas
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

 // more or less redefinition of the above, when returning "by reference"
 // but doesn't seem to work yet
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

 // MACROS to call the above, but also instantiate the template
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

 // Finally, start with STIR specific definitions

#if 1
 // first support shared_ptr
 // note: due to a current bug in swig, we have to explicitly tell it to use 
 // stir::shared_ptr (even though it's identical to boost::shared_ptr)
#define SWIG_SHARED_PTR_NAMESPACE stir
%include <boost_shared_ptr.i>
%shared_ptr(stir::ParsingObject);
%include "stir/ParsingObject.h"
%ignore stir::RegisteredParsingObject::RegisteredIt;
%include "stir/RegisteredParsingObject.h"

%shared_ptr(stir::Scanner);
%shared_ptr(stir::ProjDataInfo);
%shared_ptr(stir::ProjDataInfoCylindrical);
%shared_ptr(stir::ProjDataInfoCylindricalArcCorr);
%shared_ptr(stir::ProjDataInfoCylindricalNoArcCorr);
%shared_ptr(stir::ProjData);
%shared_ptr(stir::ProjDataFromStream);
%shared_ptr(stir::ProjDataInterfile);

# TODO cannot do this yet as then the FloatArray1D(FloatArray1D&) construction fails in test.py
#%shared_ptr(stir::Array<1,float>);
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

 /* Parse the header files to generate wrappers */
//%include "stir/shared_ptr.h"
%include "stir/Succeeded.h"
%include "stir/DetectionPosition.h"
%include "stir/Scanner.h"

 /* First do coordinates, indices, images.
    We first include them, and sort out template instantiation and indexing below.
 */
#%include <boost/operators.hpp>

%ignore stir::BasicCoordinate::operator[](const int);
%ignore stir::BasicCoordinate::operator[](const int) const;
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

%ignore stir::VectorWithOffset::operator[](int);
%ignore stir::VectorWithOffset::operator[](int) const;
%include "stir/VectorWithOffset.h"
%include "stir/NumericVectorWithOffset.h"
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

 //%ADD_indexaccess(stir::BasicCoordinate::value_type,stir::BasicCoordinate);
namespace stir { 
  %template_withindexaccess(Int3BasicCoordinate,int, BasicCoordinate<3,int>);
  %template_withindexaccessValue(Float3BasicCoordinate, BasicCoordinate<3,float>);
  %template(Float3Coordinate) Coordinate3D< float >;
  %template(FloatCartesianCoordinate3D) CartesianCoordinate3D<float>;

  %template_withindexaccess(Int2BasicCoordinate,int, BasicCoordinate<2,int>);
  %template_withindexaccessValue(Float2BasicCoordinate, BasicCoordinate<2,float>);
  %template(Float2Coordinate) Coordinate2D< float >;
  %template(FloatCartesianCoordinate2D) CartesianCoordinate2D<float>;

  %template(make_FloatCoordinate) make_coordinate<float>;
  %template(IndexRange1D) IndexRange<1>;
  //    %template(IndexRange1DVectorWithOffset) VectorWithOffset<IndexRange<1> >;
  %template(IndexRange2D) IndexRange<2>;
  //%template(IndexRange2DVectorWithOffset) VectorWithOffset<IndexRange<2> >;
  %template(IndexRange3D) IndexRange<3>;

  //    %template(FloatVectorWithOffset) VectorWithOffset<float>;
  %template_withindexaccess(FloatVectorWithOffset, float, VectorWithOffset<float>);

  # TODO need to instantiate with name?
  %template (FloatNumericVectorWithOffset) NumericVectorWithOffset<float, float>;
  %template_withindexaccess(FloatArray1D,float, Array<1,float>);
  // next doesn't work: memory leak of type 'stir::Array< 1,float >::value_type *', no destructor found.
  // and value is not float
  //%template_withindexaccessValue(FloatArray1D,Array<1,float>);

  //  William S Fulton trick (already defined in swgmacros.swg)
  //#define %arg(X...) X
  //%template_withindexaccess(FloatArray2D, %arg(Array<1,float>), %arg(Array<2,float>));
  # Todo need to instantiate with name?
  %template (_FloatNumericVectorWithOffset2D) NumericVectorWithOffset<Array<1,float>, float>;
  %template(FloatArray2D) Array<2,float>;
  // TODO these only work for getitem. setitem claims to work, but it doesn't modify the object for more than 1 level
  %ADD_indexaccess(%arg(Array<1,float>),%arg(Array<2,float>));

  %template () NumericVectorWithOffset<Array<2,float>, float>;
  %template(FloatArray3D) Array<3,float>;
  %ADD_indexaccess(%arg(Array<2,float>),%arg(Array<3,float>));
  //%template_withindexaccess(FloatArray3D,  %arg(Array<2,float>), %arg(Array<3,float>));

  %template(Float3DDiscretisedDensity) DiscretisedDensity<3,float>;
  %template(Float3DDiscretisedDensityOnCartesianGrid) DiscretisedDensityOnCartesianGrid<3,float>;
  %template(FloatVoxelsOnCartesianGrid) VoxelsOnCartesianGrid<float>;


}

 /* Now do ProjDataInfo, Sinogram et al
 */
// ignore non-const versions
%ignore stir::Bin::segment_num();
%ignore stir::Bin::axial_pos_num();
%ignore stir::Bin::view_num();
%ignore stir::Bin::tangential_pos_num();
%include "stir/Bin.h"
%include "stir/ProjDataInfo.h"
%include "stir/ProjDataInfoCylindrical.h"
%include "stir/ProjDataInfoCylindricalArcCorr.h"
%include "stir/ProjDataInfoCylindricalNoArcCorr.h"

%include "stir/Viewgram.h"
%include "stir/RelatedViewgrams.h"
%include "stir/Sinogram.h"
%include "stir/SegmentByView.h"
%include "stir/SegmentBySinogram.h"
%include "stir/ProjData.h"

%include "stir/ProjDataFromStream.h"
%include "stir/ProjDataInterfile.h"

namespace stir { 
  %template(FloatViewgram) Viewgram<float>;
  %template(FloatSinogram) Sinogram<float>;
  %template(FloatSegmentBySinogram) SegmentBySinogram<float>;
  %template(FloatSegmentByView) SegmentByView<float>;
  // should not have the following if using boost_smart_ptr.i
  //  %template(SharedScanner) boost::shared_ptr<Scanner>;
  //%template(SharedProjData) boost::shared_ptr<ProjData>;

}

// filters
#define elemT float
%shared_ptr(stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >)
%shared_ptr(stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >)
%shared_ptr(stir::RegisteredParsingObject<stir::SeparableCartesianMetzImageFilter<elemT>,
	    stir::DataProcessor<DiscretisedDensity<3,elemT> >,
	    stir::DataProcessor<DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::SeparableCartesianMetzImageFilter<elemT>)
#undef elemT

%include "stir/DataProcessor.h"
%include "stir/ChainedDataProcessor.h"
%include "stir/SeparableCartesianMetzImageFilter.h"

#define elemT float
%template(DataProcessor3DFloat) stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >;
%template(ChainedDataProcessor3DFloat) stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >;
%template(__RPSeparableCartesianMetzImageFilter3DFloat) stir::RegisteredParsingObject<
             stir::SeparableCartesianMetzImageFilter<elemT>,
             stir::DataProcessor<DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<DiscretisedDensity<3,elemT> > >;
%template(SeparableCartesianMetzImageFilter3DFloat) stir::SeparableCartesianMetzImageFilter<elemT>;
#undef elemT

 // reconstruction
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


%include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
%include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"

%include "stir/recon_buildblock/Reconstruction.h"
%include "stir/recon_buildblock/IterativeReconstruction.h"
%include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"


%template (GeneralisedObjectiveFunction3DFloat) stir::GeneralisedObjectiveFunction<stir::DiscretisedDensity<3,float> >;
#%template () stir::GeneralisedObjectiveFunction<stir::DiscretisedDensity<3,float> >;
%template (PoissonLogLikelihoodWithLinearModelForMean3DFloat) stir::PoissonLogLikelihoodWithLinearModelForMean<stir::DiscretisedDensity<3,float> >;

#define TargetT stir::DiscretisedDensity<3,float>
# TODO do we really need this name?
%template(___RPPoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat)  stir::RegisteredParsingObject<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >,
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
%template (IterativeReconstruction3DFloat) stir::IterativeReconstruction<stir::DiscretisedDensity<3,float> >;
%template (OSMAPOSLReconstruction3DFloat) stir::OSMAPOSLReconstruction<stir::DiscretisedDensity<3,float> >;
