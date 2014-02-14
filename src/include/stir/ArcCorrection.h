//
//
/*
    Copyright (C) 2005- 2005, Hammersmith Imanet Ltd
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
  \ingroup projdata
  \brief Declaration of class stir::ArcCorrection

  \author Kris Thielemans

  */
#ifndef __stir_ArcCorrection_H__
#define __stir_ArcCorrection_H__


#include "stir/ProjDataInfo.h"
#include "stir/Array.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

class Succeeded;
class ProjDataInfoCylindricalArcCorr;
class ProjDataInfoCylindricalNoArcCorr;
template <class elemT> class Sinogram;
template <class elemT> class Viewgram;
template <class elemT> class RelatedViewgrams;
template <class elemT> class SegmentBySinogram;
template <class elemT> class SegmentByView;
class ProjData;
/*! 
  \ingroup projdata  
  \brief A class to arc-correct projection data

  Arc-correction is a common name for converting the non-uniform tangential 
  sampling from a cylindrical PET scanner to a uniform one. (GE terminology is
  'geometric correction').

  This class assumes that the input projection data have already been normalised.

  For given non-arccorrected data, the data will be first multiplied by the bin-sizes,
  then interpolated to the desired uniform sampling using overlap_interpolate,
  and then divided by the new sampling. This ensures that the normalisation 
  is preserved. Also, uniform data will result in uniform output.

  \warning You <strong>have</strong> to call one of the set_up() functions
  before use of any other member function.
*/
class ArcCorrection
{
public:
  ArcCorrection();

  //! \name set_up() functions
  /*! Set-up the arc-correction object. The parameter \a proj_data_info_sptr
      has to be a shared_ptr to a ProjDataInfoCylindricalNoArcCorr object.

      Different versions are available to allow using default parameters.
  */
  //@{
  //! Most general version
  Succeeded 
    set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr, 
	   const int num_arccorrected_tangential_poss, const float bin_size);

  //! Using default bin-size of the scanner
  /*! If the default bin-size is 0, the tangential size of the central bin
      (i.e.  <code>Bin(0,0,0,0)</code>) of the
      non-arccorrected data will be used.
  */
  Succeeded 
    set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr, 
	   const int num_arccorrected_tangential_poss);
  //! Using default bin-size of the scanner and covering the FOV
  /*! If the default bin-size is 0, the tangential size of the central bin
      (i.e.  <code>Bin(0,0,0,0)</code>) of the
      non-arccorrected data will be used.

      \c num_arccorrected_bins is chosen such that the new (radial) FOV
      is slightly larger than the one covered by the original data.      
  */
  Succeeded
    set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr);
  //@}

  /*! \name functions returning a ProjDataInfoCylindricalArcCorr
      object describing to the arc-corrected data.
  */
  //@{
  const ProjDataInfoCylindricalArcCorr& 
    get_arc_corrected_proj_data_info() const;
  //! Returning a shared_ptr to the object
  /*! \warning It is dangerous to change the object pointed to. Use
      this function only if you are sure that this will never happen.
      \todo return a shared_ptr<ProjDataInfoCylindricalArcCorr>
      after switching to boost::shared_ptr.
  */
  shared_ptr<ProjDataInfo> 
    get_arc_corrected_proj_data_info_sptr() const;
  //@}

  /*! \name functions returning a ProjDataInfoCylindricalArcCorr
      object describing to the arc-corrected data.
  */
  //@{
  const ProjDataInfoCylindricalNoArcCorr& 
    get_not_arc_corrected_proj_data_info() const;
  //! Returning a shared_ptr to the object
  /*! \warning It is dangerous to change the object pointed to. Use
      this function only if you are sure that this will never happen.
      \todo return a shared_ptr<ProjDataInfoCylindricalNoArcCorr>
      after switching to boost::shared_ptr.
  */
  shared_ptr<ProjDataInfo> 
    get_not_arc_corrected_proj_data_info_sptr() const;
  //@}

  //! \name functions to do the arc-correction
  /*! Almost all these functions come in pairs (the exception being the
      function that arc-corrects a whole ProjData). 
      The 1 argument version returns
      the arc-corrected data. In the 2 argument version, the first argument
      will be filled with the arc-corrected data. 
      \warning In the 2 argument version, the output argument has to 
      have a projection data info corresponding to the one returned by
      get_arc_corrected_proj_data_info(). This is (only) checked
      using assert().
  */
  //@{
  Sinogram<float> do_arc_correction(const Sinogram<float>& in) const;
  void do_arc_correction(Sinogram<float>& out, const Sinogram<float>& in) const;
  Viewgram<float> do_arc_correction(const Viewgram<float>& in) const;
  void do_arc_correction(Viewgram<float>& out, const Viewgram<float>& in) const;
  RelatedViewgrams<float> do_arc_correction(const RelatedViewgrams<float>& in) const;
  void do_arc_correction(RelatedViewgrams<float>& out, const RelatedViewgrams<float>& in) const;
  SegmentBySinogram<float> do_arc_correction(const SegmentBySinogram<float>& in) const;
  void do_arc_correction(SegmentBySinogram<float>& out, const SegmentBySinogram<float>& in) const;
  SegmentByView<float> do_arc_correction(const SegmentByView<float>& in) const;
  void do_arc_correction(SegmentByView<float>& out, const SegmentByView<float>& in) const;
  Succeeded do_arc_correction(ProjData& out, const ProjData& in) const;
  //@}

private:
  shared_ptr<ProjDataInfo> _noarc_corr_proj_data_info_sptr;
  shared_ptr<ProjDataInfo> _arc_corr_proj_data_info_sptr;
  Array<1,float> _arccorr_coords;
  Array<1,float> _noarccorr_coords;
  Array<1,float> _noarccorr_bin_sizes;
  float tangential_sampling;

  void do_arc_correction(Array<1,float>& out, const Array<1,float>& in) const;
};

END_NAMESPACE_STIR
#endif

