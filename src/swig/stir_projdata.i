/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2013, 2014, 2015, 2018 - 2022 University College London
    Copyright (C) 2022 National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::Scanner, stir::ProjDataInfo and stir::ProjData hierarchy

  \author Kris Thielemans
  \author Daniel Deidda
*/
%shared_ptr(stir::Scanner);
%shared_ptr(stir::ProjDataInfo);
%shared_ptr(stir::ProjDataInfoCylindrical);
%shared_ptr(stir::ProjDataInfoCylindricalArcCorr);
%shared_ptr(stir::ProjDataInfoCylindricalNoArcCorr);
%shared_ptr(stir::ProjDataInfoGeneric);
%shared_ptr(stir::ProjDataInfoGenericNoArcCorr);
%shared_ptr(stir::ProjDataInfoBlocksOnCylindricalNoArcCorr);

%shared_ptr(stir::ProjData);
%shared_ptr(stir::ProjDataFromStream);
%shared_ptr(stir::ProjDataInterfile);
%shared_ptr(stir::ProjDataInMemory);
%shared_ptr(stir::SegmentBySinogram<float>);
%shared_ptr(stir::SegmentByView<float>);
%shared_ptr(stir::Segment<float>);
%shared_ptr(stir::Sinogram<float>);
%shared_ptr(stir::Viewgram<float>);

%newobject stir::Scanner::get_scanner_from_name;
%include "stir/Scanner.h"

%attributeref(stir::Bin, int, segment_num);
%attributeref(stir::Bin, int, axial_pos_num);
%attributeref(stir::Bin, int, view_num);
%attributeref(stir::Bin, int, tangential_pos_num);
%attribute(stir::Bin, float, bin_value, get_bin_value, set_bin_value);
%include "stir/Bin.h"

%newobject stir::ProjDataInfo::ProjDataInfoGE;
%newobject stir::ProjDataInfo::ProjDataInfoCTI;
// ignore this to avoid problems with unique_ptr
%ignore stir::ProjDataInfo::construct_proj_data_info;
// make sure we can use the new name anyway (although this removes
// ProjDataInfoCTI from the target language)
// See also the %extend trick below which currently doesn't work
%rename(construct_proj_data_info) ProjDataInfoCTI;

%factory_shared(stir::ProjDataInfo*,
                stir::ProjDataInfoCylindricalNoArcCorr,
                stir::ProjDataInfoCylindricalArcCorr,
                stir::ProjDataInfoBlocksOnCylindricalNoArcCorr,
                stir::ProjDataInfoGenericNoArcCorr);
%factory_shared(stir::ProjDataInfo const*,
                stir::ProjDataInfoCylindricalNoArcCorr,
                stir::ProjDataInfoCylindricalArcCorr,
                stir::ProjDataInfoBlocksOnCylindricalNoArcCorr,
                stir::ProjDataInfoGenericNoArcCorr);

%include "stir/ProjDataInfo.h"

%include "stir/ProjDataInfoCylindrical.h"
%include "stir/ProjDataInfoCylindricalArcCorr.h"
%include "stir/ProjDataInfoCylindricalNoArcCorr.h"
%include "stir/ProjDataInfoGeneric.h"
%include "stir/ProjDataInfoGenericNoArcCorr.h"
%include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"

%extend stir::ProjDataInfoBlocksOnCylindricalNoArcCorr {
    
stir::LORInAxialAndNoArcCorrSinogramCoordinates<float> get_lor(const Bin bin){
    stir::LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
    $self->get_LOR(lor,bin);
    return lor;
}

stir::CartesianCoordinate3D<float>
    find_cartesian_coordinate_of_detection_1(const Bin bin) const 
{
    CartesianCoordinate3D<float> coord_1;
    CartesianCoordinate3D<float> coord_2;
    $self->find_cartesian_coordinates_of_detection(coord_1,
                                                   coord_2,
                                                   bin);
    
    return coord_1;
}

stir::CartesianCoordinate3D<float>
    find_cartesian_coordinate_of_detection_2(const Bin bin) const 
{
    CartesianCoordinate3D<float> coord_1;
    CartesianCoordinate3D<float> coord_2;
    $self->find_cartesian_coordinates_of_detection(coord_1,
                                                   coord_2,
                                                   bin);
    
    return coord_2;
}
}

%include "stir/Viewgram.h"
%include "stir/RelatedViewgrams.h"
%include "stir/Sinogram.h"
%include "stir/Segment.h"
%include "stir/SegmentByView.h"
%include "stir/SegmentBySinogram.h"

#if 0
%extend stir::ProjDataInfo 
{
  // TODO this does not work due to %ignore statement above
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
#endif

// ignore this to avoid problems with unique_ptr, and add it later
%ignore stir::ProjData::get_subset;
%include "stir/ProjData.h"

%newobject stir::ProjData::get_subset;

namespace stir {
%extend ProjData
  {
    // work around the current SWIG limitation that it doesn't wrap unique_ptr. See above
    ProjDataInMemory* get_subset(const std::vector<int>& views)
    {
      return self->get_subset(views).release();
    }

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
        // TODO avoid need for copy to Array
        Array<3,float> array = swigstir::create_array_for_proj_data(*$self);
	swigstir::fill_Array_from_Python_iterator(&array, arg);
        fill_from(*$self, array.begin_all(), array.end_all());
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
      fill_from(*$self, array.begin_all(), array.end_all());
    }
#endif
  }

  // horrible repetition of above. should be solved with a macro or otherwise
  // we need it as ProjDataInMemory has 2 fill() methods, and therefore SWIG doesn't use extended fill() from above
%extend ProjDataInMemory
  {
#ifdef SWIGPYTHON
    %feature("autodoc", "fill from a Python iterator, e.g. proj_data.fill(numpyarray.flat)") fill;
    void fill(PyObject* const arg)
    {
      if (PyIter_Check(arg))
      {
        Array<3,float> array = swigstir::create_array_for_proj_data(*$self);
	swigstir::fill_Array_from_Python_iterator(&array, arg);
        fill_from(*$self, array.begin_all(), array.end_all());
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
    void fill(const mxArray *pm)
    { 
      Array<3,float> array;
      swigstir::fill_Array_from_matlab(array, pm, true);
      fill_from(*$self, array.begin_all(), array.end_all());
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
