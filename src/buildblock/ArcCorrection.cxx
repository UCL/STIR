//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \brief Implementation of class stir::ArcCorrection

  \author Kris Thielemans

  $Date$
  $Revision$
  */
#include "stir/ArcCorrection.h"
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/SegmentBySinogram.h"
#include "stir/SegmentByView.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include "stir/numerics/overlap_interpolate.h"
#include <typeinfo>

START_NAMESPACE_STIR
ArcCorrection::
ArcCorrection()
{}

const ProjDataInfoCylindricalNoArcCorr&
ArcCorrection::
get_not_arc_corrected_proj_data_info() const
{ 
  return
    static_cast<const ProjDataInfoCylindricalNoArcCorr&>(*_noarc_corr_proj_data_info_sptr); 
}

const ProjDataInfoCylindricalArcCorr&
ArcCorrection::
get_arc_corrected_proj_data_info() const
{ 
  return
    static_cast<const ProjDataInfoCylindricalArcCorr&>(*_arc_corr_proj_data_info_sptr); 
}

shared_ptr<ProjDataInfo> 
ArcCorrection::
get_arc_corrected_proj_data_info_sptr() const
{
  return _arc_corr_proj_data_info_sptr;
}

shared_ptr<ProjDataInfo> 
ArcCorrection::
get_not_arc_corrected_proj_data_info_sptr() const
{
  return _noarc_corr_proj_data_info_sptr;
}


Succeeded
ArcCorrection::
set_up(const shared_ptr<ProjDataInfo>& noarc_corr_proj_data_info_sptr, 
       const int num_arccorrected_tangential_poss, 
       const float bin_size)
{
  if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>
      (noarc_corr_proj_data_info_sptr.get()) == 0)
    {
      // give friendly warning message
      if (dynamic_cast<ProjDataInfoCylindricalArcCorr const *>
	  (noarc_corr_proj_data_info_sptr.get()) != 0)
	warning("ArcCorrection called with arc-corrected proj_data_info");
      else
	warning("ArcCorrection called with proj_data_info of the wrong type:\n\t%s",
		typeid(*noarc_corr_proj_data_info_sptr).name());
      return Succeeded::no;
    }

  _noarc_corr_proj_data_info_sptr = noarc_corr_proj_data_info_sptr;
  const int min_segment_num =
    _noarc_corr_proj_data_info_sptr->get_min_segment_num();
  const int max_segment_num =
    _noarc_corr_proj_data_info_sptr->get_max_segment_num();
    
  VectorWithOffset<int> min_ring_diff(min_segment_num, max_segment_num); 
  VectorWithOffset<int> max_ring_diff(min_segment_num, max_segment_num);
  VectorWithOffset<int> num_axial_pos_per_segment(min_segment_num, max_segment_num);
  for (int segment_num=min_segment_num; segment_num<=max_segment_num; ++segment_num)
    {
      min_ring_diff[segment_num] =
	get_not_arc_corrected_proj_data_info().get_min_ring_difference(segment_num);
      max_ring_diff[segment_num] =
	get_not_arc_corrected_proj_data_info().get_max_ring_difference(segment_num);
      num_axial_pos_per_segment[segment_num] =
	_noarc_corr_proj_data_info_sptr->get_num_axial_poss(segment_num);
    }

  _arc_corr_proj_data_info_sptr =
    new ProjDataInfoCylindricalArcCorr(
				       new Scanner(*_noarc_corr_proj_data_info_sptr->get_scanner_ptr()),
				       bin_size,
				       num_axial_pos_per_segment,
				       min_ring_diff, 
				       max_ring_diff,
				       noarc_corr_proj_data_info_sptr->get_num_views(),
				       num_arccorrected_tangential_poss);

  tangential_sampling =
    get_arc_corrected_proj_data_info().get_tangential_sampling();

  _noarccorr_coords.resize(_noarc_corr_proj_data_info_sptr->get_min_tangential_pos_num(),
			   _noarc_corr_proj_data_info_sptr->get_max_tangential_pos_num()+1);
  _noarccorr_bin_sizes.resize(_noarc_corr_proj_data_info_sptr->get_min_tangential_pos_num(),
			      _noarc_corr_proj_data_info_sptr->get_max_tangential_pos_num());
   {
    const float angular_increment =
      get_not_arc_corrected_proj_data_info().get_angular_increment();
    const float ring_radius =
      get_not_arc_corrected_proj_data_info().get_ring_radius();
    int tang_pos_num = _noarc_corr_proj_data_info_sptr->get_min_tangential_pos_num();
    _noarccorr_coords[tang_pos_num] =
      ring_radius * sin((tang_pos_num-.5F)*angular_increment);	
    for (;
	 tang_pos_num <= _noarc_corr_proj_data_info_sptr->get_max_tangential_pos_num();
	 ++tang_pos_num)
      {
	_noarccorr_coords[tang_pos_num+1] =
	  ring_radius * sin((tang_pos_num+.5F)*angular_increment);
	_noarccorr_bin_sizes[tang_pos_num] =
	  _noarccorr_coords[tang_pos_num+1] -
	  _noarccorr_coords[tang_pos_num];
      }
  }
  _arccorr_coords.resize(_arc_corr_proj_data_info_sptr->get_min_tangential_pos_num(),
			 _arc_corr_proj_data_info_sptr->get_max_tangential_pos_num()+1);
  {
    int tang_pos_num;
    for (tang_pos_num = _arc_corr_proj_data_info_sptr->get_min_tangential_pos_num();
	 tang_pos_num <= _arc_corr_proj_data_info_sptr->get_max_tangential_pos_num();
	 ++tang_pos_num)
      {
	_arccorr_coords[tang_pos_num] =
	  (tang_pos_num-.5F)*tangential_sampling;
      }
    _arccorr_coords[tang_pos_num] =
      (tang_pos_num+.5F)*tangential_sampling;
  }
  return Succeeded::yes;
}
    
Succeeded
ArcCorrection::
  set_up(const shared_ptr<ProjDataInfo>& noarc_corr_proj_data_info_sptr,
	 const int num_arccorrected_tangential_poss)
{
  float tangential_sampling = 
    noarc_corr_proj_data_info_sptr->get_scanner_ptr()->get_default_bin_size();
  if (tangential_sampling<=0)
    {
      tangential_sampling = 
	noarc_corr_proj_data_info_sptr->
	get_sampling_in_s(Bin(0,0,0,0));
      warning("ArcCorrection::set_up called for a scanner with default\n"
	      "tangential sampling (aka bin size) equal to 0.\n"
	      "Using the central bin size %g.",
	      tangential_sampling);	
    }
  return
    set_up(noarc_corr_proj_data_info_sptr, 
	   num_arccorrected_tangential_poss,
	   tangential_sampling);
}


Succeeded
ArcCorrection::
set_up(const shared_ptr<ProjDataInfo>& noarc_corr_proj_data_info_sptr)
{
  float tangential_sampling = 
    noarc_corr_proj_data_info_sptr->get_scanner_ptr()->get_default_bin_size();
  if (tangential_sampling<=0)
    {
      tangential_sampling = 
	noarc_corr_proj_data_info_sptr->
	get_sampling_in_s(Bin(0,0,0,0));
      warning("ArcCorrection::set_up called for a scanner with default\n"
	      "tangential sampling (aka bin size) equal to 0.\n"
	      "Using the central bin size %g.",
	      tangential_sampling);	
    }
  const float max_s =
    std::max(
	     noarc_corr_proj_data_info_sptr->
	     get_s(Bin(0,0,0,
		       noarc_corr_proj_data_info_sptr->
		       get_max_tangential_pos_num()+2)),
	     -noarc_corr_proj_data_info_sptr->
	     get_s(Bin(0,0,0,
		       noarc_corr_proj_data_info_sptr->
		       get_min_tangential_pos_num()-2))
	     );
  const int max_arccorr_tangential_pos_num =
    static_cast<int>(ceil(max_s/tangential_sampling));
  return
    set_up(noarc_corr_proj_data_info_sptr,
	   2*max_arccorr_tangential_pos_num+1,
	   tangential_sampling);
}

void 
ArcCorrection::
do_arc_correction(Array<1,float>& out, const Array<1,float>& in) const
{
  assert(in.get_index_range() == _noarccorr_bin_sizes.get_index_range());
  assert(out.get_min_index() == _arccorr_coords.get_min_index());
  assert(out.get_max_index() == _arccorr_coords.get_max_index()-1);

  overlap_interpolate(out.begin(), out.end(),
		      _arccorr_coords.begin(), _arccorr_coords.end(),
		      in.begin(), in.end(),
		      _noarccorr_coords.begin(), _noarccorr_coords.end());
  out /= tangential_sampling;
}

void
ArcCorrection::
do_arc_correction(Sinogram<float>& out, const Sinogram<float>& in) const
{
  assert(*in.get_proj_data_info_ptr() == *_noarc_corr_proj_data_info_sptr);
  assert(*out.get_proj_data_info_ptr() == *_arc_corr_proj_data_info_sptr);
  assert(out.get_axial_pos_num() == in.get_axial_pos_num());
  assert(out.get_segment_num() == in.get_segment_num());
  for (int view_num=in.get_min_view_num(); view_num<=in.get_max_view_num(); ++view_num)
    do_arc_correction(out[view_num], in[view_num]);
}

Sinogram<float>
ArcCorrection::
do_arc_correction(const Sinogram<float>& in) const
{
  Sinogram<float> out(_arc_corr_proj_data_info_sptr,
		      in.get_axial_pos_num(),
		      in.get_segment_num());
  do_arc_correction(out, in);
  return out;
}

void
ArcCorrection::
do_arc_correction(Viewgram<float>& out, const Viewgram<float>& in) const
{
  assert(*in.get_proj_data_info_ptr() == *_noarc_corr_proj_data_info_sptr);
  assert(*out.get_proj_data_info_ptr() == *_arc_corr_proj_data_info_sptr);
  assert(out.get_view_num() == in.get_view_num());
  assert(out.get_segment_num() == in.get_segment_num());
  for (int axial_pos_num=in.get_min_axial_pos_num(); axial_pos_num<=in.get_max_axial_pos_num(); ++axial_pos_num)
    do_arc_correction(out[axial_pos_num], in[axial_pos_num]);
}

Viewgram<float>
ArcCorrection::
do_arc_correction(const Viewgram<float>& in) const
{
  Viewgram<float> out(_arc_corr_proj_data_info_sptr,
		      in.get_view_num(),
		      in.get_segment_num());
  do_arc_correction(out, in);
  return out;
}

void
ArcCorrection::
do_arc_correction(RelatedViewgrams<float>& out, const RelatedViewgrams<float>& in) const
{
  RelatedViewgrams<float>::iterator out_iter = out.begin();
  RelatedViewgrams<float>::const_iterator in_iter = in.begin();
  for (; out_iter != out.end(); ++out_iter, ++in_iter)
    do_arc_correction(*out_iter, *in_iter);
}

RelatedViewgrams<float>
ArcCorrection::
do_arc_correction(const RelatedViewgrams<float>& in) const
{
  RelatedViewgrams<float> out =
    _arc_corr_proj_data_info_sptr->get_empty_related_viewgrams(in.get_basic_view_segment_num(),
							       in.get_symmetries_sptr());
  do_arc_correction(out, in);
  return out;
}

void
ArcCorrection::
do_arc_correction(SegmentBySinogram<float>& out, const SegmentBySinogram<float>& in) const
{
  assert(*in.get_proj_data_info_ptr() == *_noarc_corr_proj_data_info_sptr);
  assert(*out.get_proj_data_info_ptr() == *_arc_corr_proj_data_info_sptr);
  assert(out.get_segment_num() == in.get_segment_num());
  for (int axial_pos_num=in.get_min_axial_pos_num(); axial_pos_num<=in.get_max_axial_pos_num(); ++axial_pos_num)
    for (int view_num=in.get_min_view_num(); view_num<=in.get_max_view_num(); ++view_num)
      do_arc_correction(out[axial_pos_num][view_num], in[axial_pos_num][view_num]);
}

SegmentBySinogram<float> 
ArcCorrection::
do_arc_correction(const SegmentBySinogram<float>& in) const
{
  SegmentBySinogram<float> out(_arc_corr_proj_data_info_sptr,
			       in.get_segment_num());
  do_arc_correction(out, in);
  return out;
}


void
ArcCorrection::
do_arc_correction(SegmentByView<float>& out, const SegmentByView<float>& in) const
{
  assert(*in.get_proj_data_info_ptr() == *_noarc_corr_proj_data_info_sptr);
  assert(*out.get_proj_data_info_ptr() == *_arc_corr_proj_data_info_sptr);
  assert(out.get_segment_num() == in.get_segment_num());
  for (int view_num=in.get_min_view_num(); view_num<=in.get_max_view_num(); ++view_num)
    for (int axial_pos_num=in.get_min_axial_pos_num(); axial_pos_num<=in.get_max_axial_pos_num(); ++axial_pos_num)
      do_arc_correction(out[view_num][axial_pos_num], in[view_num][axial_pos_num]);
}


SegmentByView<float> 
ArcCorrection::
do_arc_correction(const SegmentByView<float>& in) const
{
  SegmentByView<float> out(_arc_corr_proj_data_info_sptr,
			     in.get_segment_num());
  do_arc_correction(out, in);
  return out;
}


Succeeded
ArcCorrection::
do_arc_correction(ProjData& out, const ProjData& in) const
{
  assert(*in.get_proj_data_info_ptr() == *_noarc_corr_proj_data_info_sptr);
  assert(*out.get_proj_data_info_ptr() == *_arc_corr_proj_data_info_sptr);
  // Declare temporary viewgram out of the loop to avoid reallocation
  // There is no default constructor, so we need to set it to some junk first.
  Viewgram<float> viewgram = 
    _arc_corr_proj_data_info_sptr->get_empty_viewgram(in.get_min_view_num(), in.get_min_segment_num());
  for (int segment_num=in.get_min_segment_num(); segment_num<=in.get_max_segment_num(); ++segment_num)
    for (int view_num=in.get_min_view_num(); view_num<=in.get_max_view_num(); ++view_num)
      {
	viewgram = 
	  _arc_corr_proj_data_info_sptr->get_empty_viewgram(view_num, segment_num);
	do_arc_correction(viewgram, in.get_viewgram(view_num, segment_num));
	if (out.set_viewgram(viewgram) == Succeeded::no)
	  return Succeeded::no;
      }
  return Succeeded::yes;
}

END_NAMESPACE_STIR
