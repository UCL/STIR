//
// $Id$
//
/*!

  \file
  \ingroup projdata

  \brief Implementations for class stir::ProjDataGEAdvance

  \author Damiano Belluzzo
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-06-22, Hammersmith Imanet Ltd
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



#include "stir/ProjDataGEAdvance.h"
#include "stir/Succeeded.h"
#include "stir/Viewgram.h"
#include "stir/Sinogram.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/IO/read_data.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::vector;
using std::ios;
using std::accumulate;
using std::find;
#endif

START_NAMESPACE_STIR

ProjDataGEAdvance::ProjDataGEAdvance(iostream* s)
  :
  sino_stream(s),
  offset(0),
  on_disk_data_type(NumericType::SHORT),
  on_disk_byte_order(ByteOrder::big_endian)
{
  // TODO find from file
  const int max_delta = 11;
  shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Advance));
  proj_data_info_ptr.reset( 
    ProjDataInfo::ProjDataInfoGE(scanner_sptr, max_delta, 
                                 scanner_sptr->get_max_num_views(), 
				 scanner_sptr->get_default_num_arccorrected_bins()));
  
  
  const int min_view = proj_data_info_ptr->get_min_view_num();
  const int max_view = proj_data_info_ptr->get_max_view_num();
  view_scaling_factor.grow(min_view, max_view);
  
  const int offset_begin = 256;
  sino_stream->seekg(offset, ios::beg); // overall offset
  sino_stream->seekg(offset_begin, ios::cur); // "breakfile" offset
  
  // reads the views_scaling_factors
  {
    float scale = float(1);
    // KT 18/10/99 added on_disk_byte_order
    if (read_data(*sino_stream, view_scaling_factor,
                  NumericType::FLOAT, 
                  scale,
                  on_disk_byte_order) == Succeeded::no
        || scale != 1)
      error("ProjDataGEAdvance: error reading scale factors from header\n");
    
    // 	for ( int k=min_view ; k <= max_view ; k++)
    // 	  {
    // 	    cout << "view_scaling_factor[" << k << "] = " 
    // 		 << view_scaling_factor[k] << endl;
    // 	  }
  }
  
  //=============================================================
  // parameters as in the unsorted case (help to access the file)
  //=============================================================
  //-------------------------------------------------------------
  // WARNING: this construction works only with odd max_delta
  //-------------------------------------------------------------
  
  int num_segments_orig = proj_data_info_ptr->get_max_segment_num()+1;	
  //(int) get_max_average_ring_difference(); 
  
  num_rings_orig.resize(num_segments_orig);
  segment_sequence_orig.resize(num_segments_orig);
  
  
  const int num_rings = proj_data_info_ptr->get_scanner_ptr()->get_num_rings();
  num_rings_orig[0]=(2*num_rings-1);
  
  {
    for (unsigned int i=1; i<= num_rings_orig.size()/2; i++)
    {
      num_rings_orig[2*i-1]=(2*num_rings-1) - 4*i;
      num_rings_orig[2*i]=(2*num_rings-1) - 4*i;
      //  cout << "num_rings_orig[" << 2*i-1 << "] = " << num_rings_orig[2*i-1] << endl;
      // cout << "num_rings_orig[" << 2*i << "] = " << num_rings_orig[2*i] << endl;
    }
  }
  
  
  
  // segment sequence in the unsorted data format
  
  segment_sequence_orig[0]=0;
  
  {
    for (unsigned int i=1; i<= segment_sequence_orig.size()/2; i++)
    { 
      segment_sequence_orig[2*i-1] = static_cast<int>(i);
      segment_sequence_orig[2*i] = -static_cast<int>(i);
      // cout << "segment_sequence_orig[" << 2*i-1 << "] = " << segment_sequence_orig[2*i-1] << endl;
      // cout << "segment_sequence_orig[" << 2*i << "] = " << segment_sequence_orig[2*i] << endl;
    }
  }
  
}
  
Viewgram<float>
ProjDataGEAdvance::
get_viewgram(const int view_num, const int segment_num,
             const bool make_num_tangential_poss_odd) const
  {
    // --------------------------------------------------------
    // --------------------------------------------------------
    // 	Advance GE format reader with sort of the segments
    // --------------------------------------------------------
    // --------------------------------------------------------
    //
    // segment_num     0       +1      -1      +2      -2      +3      -3 .......
    // ave_ring_diff   0       +2      -2      +3      -3      +4      -4 .......
    // max_ring_diff   +1      +2      -2      +3      -3      +4      -4 .......
    // min_ring_diff   -1      +2      -2      +3      -3      +4      -4 .......
    
    
    int odd_ring_segment_offset;
    long num_rings_offset;
    long segment_offset;
    long jump_size;
    int jump_ring;
    
    
    //----------------------------------------------------------------
    // data with delta ring = +1 and -1 are contained in segment 0:
    // for this reason there is not a segment +1 and a segment -1 
    //----------------------------------------------------------------
    
    // segment number in the unsorted data format 
    // (no problem for segments +1 and -1)
    
    const int sign_segment_num = (segment_num == 0) ? 0 : ( (segment_num > 0) ? 1 : -1 );
    const int segment_num_orig = (int) ((segment_num+sign_segment_num) / 2);
    
    //      cout << "segment_num_orig " << segment_num_orig << endl;
    
    // segment index in the unsorted data format
    
    const int  index_orig = 
      find(segment_sequence_orig.begin(), 
	     segment_sequence_orig.end(), segment_num_orig) - 
             segment_sequence_orig.begin();
    
    //      cout << "index_orig " << index_orig << endl;
    
    
    //---------------------------------------------------------------
    // OFFSETS
    //---------------------------------------------------------------
    
    const int offset_begin = 256;
    
    // offset in rings for the odd sorted segments (value 0 or 1)
    
    odd_ring_segment_offset = abs((segment_num+sign_segment_num) % 2); 
    
    //      cout << "odd_ring_segment_offset " << odd_ring_segment_offset << endl;
    
    // offset in rings in the unsorted data format
    
    num_rings_offset = 
      accumulate(num_rings_orig.begin(), 
      num_rings_orig.begin() + index_orig, 0) +
      odd_ring_segment_offset;
    
    //      cout << "num_rings_offset " << num_rings_offset << endl;
    
    // offset to the first data in the first view_num of the segment chosen
    
    segment_offset =  
      num_rings_offset * get_num_tangential_poss() * on_disk_data_type.size_in_bytes();
    
    //      cout << "segment_offset " << segment_offset << endl;
    
    // size of the jump to the first record of the following view_num
    if (segment_num == 0)
    {
      jump_size = 
        ( 
        accumulate(num_rings_orig.begin(), num_rings_orig.end(), 0) 
        - num_rings_orig[index_orig] +
        + 2 * odd_ring_segment_offset
        ) * get_num_tangential_poss() * on_disk_data_type.size_in_bytes();
    }
    else
    {
      jump_size = 
        ( 
        accumulate(num_rings_orig.begin(), num_rings_orig.end(), 0)
        - num_rings_orig[index_orig]
        - 1  // offset to balance the jump_ring in the cycle
        + 2 * odd_ring_segment_offset
        ) * get_num_tangential_poss() * on_disk_data_type.size_in_bytes();
    }
    //      cout << "jump_size " << jump_size << endl;
    
    // size of the jump between two rings of the same view_num and of the same 
    // segment
    if (segment_num == 0)
      jump_ring = 0;
    else
      jump_ring = get_num_tangential_poss() * on_disk_data_type.size_in_bytes();
    
    int ring_offset = get_num_tangential_poss() * on_disk_data_type.size_in_bytes();
    
    // jumps the initial offset
    
    sino_stream->seekg(offset, ios::beg); // overall offset
    sino_stream->seekg(offset_begin, ios::cur); // "breakfile" offset
    sino_stream->seekg(get_num_views() * sizeof(float), ios::cur); // skip view scaling factors
    
    // jumps at the beginning of the chosen segment
    
    sino_stream->seekg(segment_offset, ios::cur);
    
    // jump to the right Viewgram
    
    int jump_ini = (view_num - get_min_view_num())*
      (ring_offset*get_num_axial_poss(segment_num)
      + jump_ring*get_num_axial_poss(segment_num)
      + jump_size);
    
    // 	  cout << "ring_offset  " << ring_offset << endl;
    // 	  cout << "jump_ring  " << jump_ring << endl;
    // 	  cout << "jump_size  " << jump_size << endl;
    // 	  cout << " jump_ini " << jump_ini << endl;
    
    sino_stream->seekg(jump_ini, ios::cur);
    
    Viewgram<float> data = get_empty_viewgram(view_num, segment_num,false);
    
    for (int ring =get_min_axial_pos_num(segment_num) ; ring <= get_max_axial_pos_num(segment_num); ring++)
    {
      {
        float scale = float(1);
        if (read_data(*sino_stream, data[ring],
                      on_disk_data_type, 
                      scale,
                      on_disk_byte_order) == Succeeded::no
            || scale != 1)
          error("ProjDataGEAdvance: error reading data\n");
      }
      sino_stream->seekg(jump_ring, ios::cur);
    }
    
    // scales the Viewgram<float>
    data *= view_scaling_factor[view_num];
    
    if (make_num_tangential_poss_odd && get_num_tangential_poss()%2==0)
    {
      const int new_max_tangential_pos = get_max_tangential_pos_num() + 1;
      data.grow(
        IndexRange2D(get_min_axial_pos_num(segment_num),
                     get_max_axial_pos_num(segment_num),        
                     get_min_tangential_pos_num(),
                     new_max_tangential_pos));   
    }  
    
    return data;
}


Succeeded ProjDataGEAdvance::set_viewgram(const Viewgram<float>& v)
{
  // TODO
  // but this is difficult: how to adjust the scale factors when writing only 1 viewgram ?
  warning("ProjDataGEAdvance::set_viewgram not implemented yet\n");
  return Succeeded::no;
}

Sinogram<float> ProjDataGEAdvance::get_sinogram(const int ax_pos_num, const int segment_num,const bool make_num_tangential_poss_odd) const
{ 
  // TODO
  error("ProjDataGEAdvance::get_sinogram not implemented yet\n"); 
  return get_empty_sinogram(ax_pos_num, segment_num);}

Succeeded ProjDataGEAdvance::set_sinogram(const Sinogram<float>& s)
{
  // TODO
  warning("ProjDataGEAdvance::set_sinogram not implemented yet\n");
  return Succeeded::no;
}

END_NAMESPACE_STIR
