//
// $Id$
//
/*!

  \file
  \ingroup utilities

  \brief Find sinogram rescaling factors

  \author Sanida Mustafovic

  $Date$
  $Revision$ 
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/




#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/SegmentBySinogram.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Succeeded.h"
#include "stir/Sinogram.h"
#include "stir/VectorWithOffset.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjDataInMemory.h"

#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/VoxelsOnCartesianGrid.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::min;
using std::max;
using std::endl;
using std::vector;
using std::fstream;
using std::iostream;
using std::ofstream;
#endif

USING_NAMESPACE_STIR



void
fill_sinogram(Sinogram<float>& sino, const float num_to_fill)
{

  for ( int view_num = sino.get_min_view_num();
	    view_num <= sino.get_max_view_num();
	    view_num++)
   for ( int tang_pos = sino.get_min_tangential_pos_num();
            tang_pos<= sino.get_max_tangential_pos_num();
	    tang_pos++)
     {
       sino[view_num][tang_pos] = num_to_fill;
     }
}

int
main( int argc, char* argv[])
{

  if ( argc !=4)
  {
    cerr << " Usage: " << argv[0] << " output_filename precorrected_proj_data fwd_data" << endl;
    return EXIT_FAILURE;
  }

  const string scaling_factors_filename =  argv[1];
  shared_ptr<ProjData> precorrected_data_sptr = ProjData::read_from_file(argv[2]);
  shared_ptr<ProjData> fwd_data_sptr = ProjData::read_from_file(argv[3]);

  ProjDataInfo * proj_data_info_ptr =  precorrected_data_sptr->get_proj_data_info_ptr()->clone();
  shared_ptr<ProjData> rescaling_factors_sptr = 
                       new ProjDataInterfile(proj_data_info_ptr,scaling_factors_filename);

  ofstream outfile;
  open_write_binary(outfile,scaling_factors_filename.c_str());
  
  const int min_seg_num = proj_data_info_ptr->get_min_segment_num();
  const int max_seg_num = proj_data_info_ptr->get_max_segment_num();
  const int max_axial_pos_in_seg_zero = proj_data_info_ptr->get_max_axial_pos_num(0);
  
 Array<2,float> scaling_factors(IndexRange2D( min_seg_num, max_seg_num, 
					        0, max_axial_pos_in_seg_zero));

 for ( int segment_num = precorrected_data_sptr->get_min_segment_num();
	segment_num <= precorrected_data_sptr->get_max_segment_num();
	++segment_num)
    {
            
      const SegmentBySinogram<float> precorrected_seg_by_sino = 
	precorrected_data_sptr->get_segment_by_sinogram(segment_num);

      const SegmentBySinogram<float> fwd_seg_by_sino = 
	fwd_data_sptr->get_segment_by_sinogram(segment_num);
     
       SegmentBySinogram<float> rescaling_factors_sino = 
	rescaling_factors_sptr->get_empty_segment_by_sinogram(segment_num);
          float precorrected_sum;
	  float fwd_sum;
      for (int axial_pos = precorrected_seg_by_sino.get_min_axial_pos_num();
	   axial_pos <= precorrected_seg_by_sino.get_max_axial_pos_num();
	   axial_pos++)
	{
         
	  const Sinogram<float> precorrected_sinogram = precorrected_seg_by_sino.get_sinogram(axial_pos);
          const Sinogram<float> fwd_sinogram = fwd_seg_by_sino.get_sinogram(axial_pos);
          Sinogram<float> pre_div_fwd_sinogram = precorrected_sinogram.get_empty_copy();

          precorrected_sum =0;
	  fwd_sum =0;

	  for ( int view_num = precorrected_sinogram.get_min_view_num();
		view_num <= precorrected_sinogram.get_max_view_num();
		view_num++)
	  for ( int tang_pos = precorrected_sinogram.get_min_tangential_pos_num();
		tang_pos <= precorrected_sinogram.get_max_tangential_pos_num();
		tang_pos++)
	    {
	      precorrected_sum += precorrected_sinogram[view_num][tang_pos];
	      fwd_sum += fwd_sinogram[view_num][tang_pos];	 
	    }
        
	   scaling_factors[segment_num][axial_pos] = fwd_sum/precorrected_sum;          
	  
#if 0
	  fill_sinogram(pre_div_fwd_sinogram,fwd_sum/precorrected_sum);
          rescaling_factors_sptr->set_sinogram(pre_div_fwd_sinogram);
#endif	  
	}
  
    }
  
   scaling_factors.write_data(outfile);   
   outfile.close();

   return EXIT_SUCCESS;

}
 
