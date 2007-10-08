//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
  \brief Implementation of stir::SSRB

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/SSRB.h"
#include "stir/Sinogram.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::min;
using std::max;
#endif

START_NAMESPACE_STIR

// TODO this function needs work to reliable handle segments with unequal 'num_segments_to_combine' (as GE Advance)
// parts with 'num_segments_to_combine' should be revised, all the rest is fine
ProjDataInfo *
SSRB(const ProjDataInfo& in_proj_data_info,
     const int num_segments_to_combine,
     const int num_views_to_combine,
     const int num_tang_poss_to_trim,
     const int max_in_segment_num_to_process_argument
     )
{
  if (num_segments_to_combine%2==0)
    error("SSRB: num_segments_to_combine (%d) needs to be odd\n", 
	  num_segments_to_combine);
  const int max_in_segment_num_to_process =
    max_in_segment_num_to_process_argument >= 0
    ? max_in_segment_num_to_process_argument 
    : in_proj_data_info.get_max_segment_num();

  if (in_proj_data_info.get_max_segment_num() < max_in_segment_num_to_process)
    error("SSRB: max_in_segment_num_to_process (%d) is too large\n"
	  "Input data has maximum segment number %d.",
	  max_in_segment_num_to_process,
	  in_proj_data_info.get_max_segment_num());
  if (in_proj_data_info.get_num_tangential_poss() <=
      num_tang_poss_to_trim)
    error("SSRB: too large number of tangential positions to trim (%d)\n",
	  num_tang_poss_to_trim);
  const ProjDataInfoCylindrical * const in_proj_data_info_ptr =
    dynamic_cast<ProjDataInfoCylindrical const * >
    (&in_proj_data_info);
  if (in_proj_data_info_ptr== NULL)
    {
      error("SSRB works only on segments with proj_data_info of "
	    "type ProjDataInfoCylindrical\n");
    }
  ProjDataInfoCylindrical * out_proj_data_info_ptr =
    dynamic_cast<ProjDataInfoCylindrical * >
    (in_proj_data_info_ptr->clone());

  out_proj_data_info_ptr->
    set_num_views(
		  in_proj_data_info.get_num_views()/
		  num_views_to_combine);
  out_proj_data_info_ptr->
    set_num_tangential_poss(in_proj_data_info.get_num_tangential_poss() -
			    num_tang_poss_to_trim);

  // Find new maximum segment_num
  // To understand this formula, check how the out_segment_num is related to 
  // the in_segment_num below
  const int out_max_segment_num = 
    ((max_in_segment_num_to_process == -1
      ? in_proj_data_info.get_max_segment_num()
      : max_in_segment_num_to_process
      )-(num_segments_to_combine/2))/num_segments_to_combine;
  if (out_max_segment_num <0)
    error("SSRB: max_in_segment_num_to_process  %d is too small. No output segments\n",
	  max_in_segment_num_to_process);

  const int in_axial_compression =
      in_proj_data_info_ptr->get_max_ring_difference(0) -
      in_proj_data_info_ptr->get_min_ring_difference(0)
      + 1;

  out_proj_data_info_ptr->reduce_segment_range(-out_max_segment_num,out_max_segment_num);
  for (int out_segment_num = -out_max_segment_num; 
       out_segment_num <= out_max_segment_num;
       ++out_segment_num)
    {
      const int in_min_segment_num = out_segment_num*num_segments_to_combine - num_segments_to_combine/2;
      const int in_max_segment_num = out_segment_num*num_segments_to_combine + num_segments_to_combine/2;
      out_proj_data_info_ptr->
	set_min_ring_difference(in_proj_data_info_ptr->get_min_ring_difference(in_min_segment_num),
				out_segment_num);
      out_proj_data_info_ptr->
	set_max_ring_difference(in_proj_data_info_ptr->get_max_ring_difference(in_max_segment_num),
				out_segment_num);

      out_proj_data_info_ptr->
	set_min_axial_pos_num(0,out_segment_num);
      
      // find number of axial_poss in out_segment
      // get_m could be replaced by get_t  
      {
	float min_m =1.E37F;
	float max_m =-1.E37F;
	for (int in_segment_num = in_min_segment_num; 
	     in_segment_num <= in_max_segment_num;
	     ++in_segment_num)
	  {
            if (in_axial_compression !=
                (in_proj_data_info_ptr->get_max_ring_difference(in_segment_num) -
                 in_proj_data_info_ptr->get_min_ring_difference(in_segment_num) +
                 1))
              warning("SSRB: in_proj_data_info with non-identical axial compression for all segments.\n"
		      "That's ok, but results might not be what you expect.\n");

	    min_m =
	      min(min_m,
		  in_proj_data_info_ptr->
		  get_m(Bin(in_segment_num,0,in_proj_data_info_ptr->get_min_axial_pos_num(in_segment_num), 0)));
	    max_m =
	      max(max_m,
		  in_proj_data_info_ptr->
		  get_m(Bin(in_segment_num,0,in_proj_data_info_ptr->get_max_axial_pos_num(in_segment_num), 0)));
	  }
	const float number_of_ms =
	  (max_m - min_m)/out_proj_data_info_ptr->get_axial_sampling(out_segment_num)+1;
	if (fabs(round(number_of_ms)-number_of_ms) > 1.E-3)
	  error("SSRB: number of axial positions to be found in out_segment %d is non-integer %g\n",
		out_segment_num, number_of_ms);
	out_proj_data_info_ptr->
	  set_max_axial_pos_num(round(number_of_ms) - 1,
				out_segment_num);
      }
    }
  return out_proj_data_info_ptr;
}

void 
SSRB(const string& output_filename,
     const ProjData& in_proj_data,
     const int num_segments_to_combine,
     const int num_views_to_combine,
     const int num_tang_poss_to_trim,
     const bool do_norm,
     const int max_in_segment_num_to_process
     )
{
  if (num_segments_to_combine%2==0)
    error("SSRB: num_segments_to_combine (%d) needs to be odd\n", 
	  num_segments_to_combine);
  shared_ptr<ProjDataInfo> out_proj_data_info_ptr =
    SSRB(*in_proj_data.get_proj_data_info_ptr(),
         num_segments_to_combine,
	 num_views_to_combine,
	 num_tang_poss_to_trim,
         max_in_segment_num_to_process
     );
  ProjDataInterfile out_proj_data(out_proj_data_info_ptr, output_filename, ios::out); 

  SSRB(out_proj_data, in_proj_data, do_norm);
}

void 
SSRB(ProjData& out_proj_data,
     const ProjData& in_proj_data,
     const bool do_norm
     )
{
  const ProjDataInfoCylindrical * const in_proj_data_info_ptr =
    dynamic_cast<ProjDataInfoCylindrical const * >
    (in_proj_data.get_proj_data_info_ptr());
  if (in_proj_data_info_ptr== NULL)
  {
    error("SSRB works only on segments with proj_data_info of "
      "type ProjDataInfoCylindrical\n");
  }
  const ProjDataInfoCylindrical * const out_proj_data_info_ptr =
    dynamic_cast<ProjDataInfoCylindrical const * >
    (out_proj_data.get_proj_data_info_ptr());
  if (out_proj_data_info_ptr== NULL)
  {
    error("SSRB works only on segments with proj_data_info of "
      "type ProjDataInfoCylindrical\n");
  }

  const int num_views_to_combine =
    in_proj_data.get_num_views()/ out_proj_data.get_num_views();


  if (in_proj_data.get_min_view_num()!=0 || out_proj_data.get_min_view_num()!=0)
    error ("SSRB can only mash views when min_view_num==0\n");
  if (in_proj_data.get_num_views() % out_proj_data.get_num_views())
    error ("SSRB can only mash views when out_num_views divides in_num_views\n");
  
  for (int out_segment_num = out_proj_data.get_min_segment_num(); 
       out_segment_num <= out_proj_data.get_max_segment_num();
       ++out_segment_num)
    {
      // find range of input segments that fit in the current output segment
      int in_min_segment_num = in_proj_data.get_max_segment_num();
      int in_max_segment_num = in_proj_data.get_min_segment_num();
      {
        // this the only place where we need ProjDataInfoCylindrical
        // Presumably for other types, there'd be something equivalent (say range of theta)
	const int out_min_ring_diff = 
	  out_proj_data_info_ptr->get_min_ring_difference(out_segment_num);
	const int out_max_ring_diff = 
	  out_proj_data_info_ptr->get_max_ring_difference(out_segment_num);
	for (int in_segment_num = in_proj_data.get_min_segment_num(); 
	     in_segment_num <= in_proj_data.get_max_segment_num();
	     ++in_segment_num)
	  {
	    const int in_min_ring_diff = 
	      in_proj_data_info_ptr->get_min_ring_difference(in_segment_num);
	    const int in_max_ring_diff = 
	      in_proj_data_info_ptr->get_max_ring_difference(in_segment_num);
	    if (in_min_ring_diff >= out_min_ring_diff &&
		in_max_ring_diff <= out_max_ring_diff)
	      {
		// it's a in_segment that should be rebinned in the out_segment
		if (in_min_segment_num > in_segment_num)
		  in_min_segment_num = in_segment_num;
		if (in_max_segment_num < in_segment_num)
		  in_max_segment_num = in_segment_num;
	      }
	    else if (in_min_ring_diff > out_max_ring_diff ||
		     in_max_ring_diff < out_min_ring_diff)
	      {
		// this one is outside the range of the out_segment
	      }
	    else
	      {
		error("SSRB called with in and out ring difference ranges that overlap:\n"
		      "in_segment %d has ring diffs (%d,%d)\n"
		      "out_segment %d has ring diffs (%d,%d)\n",
		      in_segment_num, in_min_ring_diff, in_max_ring_diff,
		      in_segment_num, out_min_ring_diff, out_max_ring_diff);
	      }
	  }

	// keep sinograms out of the loop to avoid reallocations
	// initialise to something because there's no default constructor
	Sinogram<float> out_sino = 
	  out_proj_data.get_empty_sinogram(out_proj_data.get_min_axial_pos_num(out_segment_num),out_segment_num);
	Sinogram<float> in_sino = 
	  in_proj_data.get_empty_sinogram(in_proj_data.get_min_axial_pos_num(out_segment_num),out_segment_num);

	for (int out_ax_pos_num = out_proj_data.get_min_axial_pos_num(out_segment_num); 
	     out_ax_pos_num  <= out_proj_data.get_max_axial_pos_num(out_segment_num);
	     ++out_ax_pos_num )
	  {
	    out_sino= out_proj_data.get_empty_sinogram(out_ax_pos_num, out_segment_num);
            // get_m could be replaced by get_t  
	    const float out_m = out_proj_data_info_ptr->get_m(Bin(out_segment_num,0, out_ax_pos_num, 0));
	  
	    unsigned int num_in_ax_pos = 0;

	    for (int in_segment_num = in_min_segment_num; 
		 in_segment_num <= in_max_segment_num;
		 ++in_segment_num)
	      for (int in_ax_pos_num = in_proj_data.get_min_axial_pos_num(in_segment_num); 
		   in_ax_pos_num  <= in_proj_data.get_max_axial_pos_num(in_segment_num);
		   ++in_ax_pos_num )
		{
		  const float in_m = in_proj_data_info_ptr->get_m(Bin(in_segment_num,0, in_ax_pos_num, 0));
		  if (fabs(out_m - in_m) < 1E-4)
		    {
		      ++num_in_ax_pos;
		      
		      in_sino = in_proj_data.get_sinogram(in_ax_pos_num, in_segment_num);
		      for (int in_view_num=in_proj_data.get_min_view_num();
			   in_view_num <= in_proj_data.get_max_view_num();
			   ++in_view_num)
			for (int tangential_pos_num=
			       max(in_proj_data.get_min_tangential_pos_num(),
				   out_proj_data.get_min_tangential_pos_num());
			     tangential_pos_num <= 
			       min(in_proj_data.get_max_tangential_pos_num(),
				   out_proj_data.get_max_tangential_pos_num());
			     ++tangential_pos_num)
			  out_sino[in_view_num/num_views_to_combine][tangential_pos_num] += 
			      in_sino[in_view_num][tangential_pos_num];
		  
		      break; // out of loop over ax_pos as we found where to put it
		    }
		}
	    if (do_norm && num_in_ax_pos!=0)
	      out_sino /= static_cast<float>(num_in_ax_pos*num_views_to_combine);
	    if (num_in_ax_pos==0)
	      warning("SSRB: no sinograms contributing to output segment %d, ax_pos %d\n",
		      out_segment_num, out_ax_pos_num);

	    out_proj_data.set_sinogram(out_sino);
	  }
      }
    }
}
END_NAMESPACE_STIR
