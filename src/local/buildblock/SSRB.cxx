//
// $Id$
//
/*!

  \file

  \brief Implementation of SSRB

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "local/stir/SSRB.h"
#include "stir/interfile.h"
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
     const int max_in_segment_num_to_process
     )
{
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

  const int out_max_segment_num = 
    ((max_in_segment_num_to_process == -1
      ? in_proj_data_info.get_max_segment_num()
      : max_in_segment_num_to_process
      )-1)/num_segments_to_combine;
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
              error("SSRB: can currently only handle in_proj_data_info with identical axial compression for all segments\n");;

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
     const int max_in_segment_num_to_process,
     const bool do_norm
     )
{
  shared_ptr<ProjDataInfo> out_proj_data_info_ptr =
    SSRB(*in_proj_data.get_proj_data_info_ptr(),
         num_segments_to_combine,
         max_in_segment_num_to_process
     );
  shared_ptr<iostream> sino_stream = 
    new fstream (output_filename.c_str(), ios::out|ios::binary);
  if (!sino_stream->good())
      error("SSRB: error opening output file %s\n",
	    output_filename.c_str());

  //ProjDataInterfile out_proj_data(output_filename, out_proj_data_info_ptr, ios::out); 
  ProjDataFromStream out_proj_data(out_proj_data_info_ptr, sino_stream); 
  write_basic_interfile_PDFS_header(output_filename, out_proj_data);

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

	// keep out_sino out of the loop to avoid reallocations
	// initialise to something because there's no default constructor
	Sinogram<float> out_sino = 
	  out_proj_data.get_empty_sinogram(out_proj_data.get_min_axial_pos_num(out_segment_num),out_segment_num);

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
		      out_sino += in_proj_data.get_sinogram(in_ax_pos_num, in_segment_num);
		      break;
		    }
		}
	    if (do_norm && num_in_ax_pos!=0)
	      out_sino /= num_in_ax_pos;
	    if (num_in_ax_pos==0)
	      warning("SSRB: no sinograms contributing to segment %d, ax_pos %d\n",
		      out_segment_num, out_ax_pos_num);

	    out_proj_data.set_sinogram(out_sino);
	  }
      }
    }
}
END_NAMESPACE_STIR
