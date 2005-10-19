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
//! Perform B-Splines Interpolation

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	$Date$
	$Revision$
*/
#include "stir/ProjData.h"
//#include "stir/display.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/IndexRange.h"
#include "stir/BasicCoordinate.h"
#include "stir/Sinogram.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Succeeded.h"
#include "local/stir/BSplines.h"
#include "local/stir/BSplinesRegularGrid.h"
#include "local/stir/interpolate_projdata.h"
#include "local/stir/extend_projdata.h"
#include "local/stir/sample_array.h"

START_NAMESPACE_STIR
using namespace BSpline;

shared_ptr<ProjDataInfo>
make_non_interleaved_proj_data_info(const ProjDataInfo& proj_data_info)
{
  shared_ptr<Scanner> scanner_sptr =
    new Scanner(*proj_data_info.get_scanner_ptr());

  /*
    const int new_num_detectors_per_ring =
    scanner_sptr->get_num_detectors_per_ring()*2;
  scanner_sptr->set_num_detectors_per_ring(new_num_detectors_per_ring);
  */
  return 
    ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
				  /*span*/ 1,/* max_delta*/ 0,
				  /* num_views */ proj_data_info.get_num_views()*2,
				  /* num_tangential_poss */ proj_data_info.get_num_tangential_poss(),
				  /* arc_corrected */ false);
}

void
make_non_interleaved_sinogram(Sinogram<float>& out_sinogram,
			      const Sinogram<float>& in_sinogram)
{
  assert(out_sinogram.get_min_view_num() == 0);
  assert(in_sinogram.get_min_view_num() == 0);
  assert(out_sinogram.get_num_views() == in_sinogram.get_num_views()*2);
  assert(in_sinogram.get_segment_num() == 0);
  assert(out_sinogram.get_segment_num() == 0);

  const int in_num_views = in_sinogram.get_num_views();

  for (int view_num = out_sinogram.get_min_view_num();
       view_num <= out_sinogram.get_max_view_num();
       ++view_num)
    {
      // TODO don't put in outer tangential poss for now to avoid boundary stuff
      for (int tangential_pos_num = out_sinogram.get_min_tangential_pos_num()+1;
	   tangential_pos_num <= out_sinogram.get_max_tangential_pos_num()-1;
	   ++tangential_pos_num)
	{
	  if ((view_num+tangential_pos_num)%2 == 0)
		{
		  const int in_view_num =
		    view_num%2==0 ? view_num/2 : (view_num+1)/2;
		  out_sinogram[view_num][tangential_pos_num] =
		    in_sinogram[in_view_num%in_num_views][(in_view_num>=in_num_views? -1: 1)*tangential_pos_num];
		}
	      else
		{
		  const int next_in_view = view_num/2+1;
		  const int other_in_view = (view_num+1)/2;

		  out_sinogram[view_num][tangential_pos_num] =
		    (in_sinogram[view_num/2][tangential_pos_num] +
		     in_sinogram[next_in_view%in_num_views][(next_in_view>=in_num_views ? -1 : 1)*tangential_pos_num] +
		     in_sinogram[other_in_view%in_num_views][(other_in_view>=in_num_views ? -1 : 1)*(tangential_pos_num-1)] +
		     in_sinogram[other_in_view%in_num_views][(other_in_view>=in_num_views ? -1 : 1)*(tangential_pos_num+1)]
		     )/4;
		}
	}
    }
}

Sinogram<float>
make_non_interleaved_sinogram(const ProjDataInfo& non_interleaved_proj_data_info,
			      const Sinogram<float>& in_sinogram)
{
  Sinogram<float> out_sinogram =
    non_interleaved_proj_data_info.get_empty_sinogram(in_sinogram.get_axial_pos_num(),
						      in_sinogram.get_segment_num());

  make_non_interleaved_sinogram(out_sinogram, in_sinogram);
  return out_sinogram;
}						      


void
make_non_interleaved_segment(SegmentBySinogram<float>& out_segment,
			     const SegmentBySinogram<float>& in_segment)
{
  for (int axial_pos_num = out_segment.get_min_axial_pos_num()+1;
       axial_pos_num <= out_segment.get_max_axial_pos_num()-1;
       ++axial_pos_num)
    {
      Sinogram<float> out_sinogram = out_segment.get_sinogram(axial_pos_num);
      make_non_interleaved_sinogram(out_sinogram, 
				    in_segment.get_sinogram(axial_pos_num));
      out_segment.set_sinogram(out_sinogram);
    }
}

SegmentBySinogram<float>
make_non_interleaved_segment(const ProjDataInfo& non_interleaved_proj_data_info,
			     const SegmentBySinogram<float>& in_segment)
{
  SegmentBySinogram<float> out_segment =
    non_interleaved_proj_data_info.get_empty_segment_by_sinogram(in_segment.get_segment_num());

  make_non_interleaved_segment(out_segment, in_segment);
  return out_segment;
}						      
  
Succeeded 
interpolate_projdata(ProjData& proj_data_out,
		     const ProjData& proj_data_in, const BSplineType these_types,
		     const bool remove_interleaving)
{
  BasicCoordinate<3, BSplineType> these_types_3; 
  these_types_3[1]=these_types_3[2]=these_types_3[3]=these_types;
  interpolate_projdata(proj_data_out,proj_data_in,these_types_3, remove_interleaving);
  return Succeeded::yes;
}

Succeeded 
interpolate_projdata(ProjData& proj_data_out,
		     const ProjData& proj_data_in,
		     const BasicCoordinate<3, BSplineType> & these_types,
		     const bool remove_interleaving)
{
  const ProjDataInfo & proj_data_in_info =
    *proj_data_in.get_proj_data_info_ptr();
  const ProjDataInfo & proj_data_out_info =
    *proj_data_out.get_proj_data_info_ptr();
	
  BSpline::BSplinesRegularGrid<3, float, float> proj_data_interpolator(these_types);
  BasicCoordinate<3, double>  offset,  step  ;
	
  // find relation between out_index and in_index such that they correspond to the same physical position
  // out_index * m_zoom + m_offset = in_index
  const float in_sampling_m = proj_data_in_info.get_sampling_in_m(Bin(0,0,0,0));
  const float out_sampling_m = proj_data_out_info.get_sampling_in_m(Bin(0,0,0,0));
  // offset in 'in' index units
  offset[1] = 
    (proj_data_in_info.get_m(Bin(0,0,0,0)) -
     proj_data_out_info.get_m(Bin(0,0,0,0))) / in_sampling_m;
  step[1]=
    out_sampling_m/in_sampling_m;
		
  const float in_sampling_phi = 
    (proj_data_in_info.get_phi(Bin(0,1,0,0)) - proj_data_in_info.get_phi(Bin(0,0,0,0))) /
    (remove_interleaving ? 2 : 1);

  const float out_sampling_phi = 
    proj_data_out_info.get_phi(Bin(0,1,0,0)) - proj_data_out_info.get_phi(Bin(0,0,0,0));
	
  offset[2] = 
    (proj_data_in_info.get_phi(Bin(0,0,0,0)) - proj_data_out_info.get_phi(Bin(0,0,0,0))) / in_sampling_phi;
  step[2] =
    out_sampling_phi/in_sampling_phi;
	
  const float in_sampling_s = proj_data_in_info.get_sampling_in_s(Bin(0,0,0,0));
  const float out_sampling_s = proj_data_out_info.get_sampling_in_s(Bin(0,0,0,0));
  offset[3] = 
    (proj_data_out_info.get_s(Bin(0,0,0,0)) -
     proj_data_in_info.get_s(Bin(0,0,0,0))) / in_sampling_s;
  step[3]=
    out_sampling_s/in_sampling_s;
	
  // initialise interpolator
  if (remove_interleaving)
  {
    shared_ptr<ProjDataInfo> non_interleaved_proj_data_info_sptr =
      make_non_interleaved_proj_data_info(proj_data_in_info);

    const SegmentBySinogram<float> non_interleaved_segment =
      make_non_interleaved_segment(*non_interleaved_proj_data_info_sptr,
				   proj_data_in.get_segment_by_sinogram(0));
    //    display(non_interleaved_segment, non_interleaved_segment.find_max(),"non-inter");
    const Array<3,float> input_extended_view = 
      extend_segment(non_interleaved_segment, 2, 2);
    proj_data_interpolator.set_coef(input_extended_view);
  }
  else
  {
    const Array<3,float> input_extended_view = 
      extend_segment(proj_data_in.get_segment_by_sinogram(0), 2, 2);
    proj_data_interpolator.set_coef(input_extended_view);
  }
	
  // now do interpolation		
  SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0) ;
  sample_at_regular_array(sino_3D_out, proj_data_interpolator, offset, step);

  proj_data_out.set_segment(sino_3D_out);
  if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
    return Succeeded::no;	   
  return Succeeded::yes;
}

END_NAMESPACE_STIR
