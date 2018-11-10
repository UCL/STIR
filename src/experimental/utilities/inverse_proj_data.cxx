//
//

/*!
  \file
  \ingroup utilities

  \brief Find the inverse of ProjData


  \author Sanida Mustafovic
  
*/
/*
    Copyright (C) 2000- 2012, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
//#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
//#include "stir/display.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfo.h"
// for ask_filename...
#include "stir/utilities.h"
#include "stir/IndexRange3D.h"
#include "stir/RelatedViewgrams.h"
#include "stir/SegmentByView.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Viewgram.h"
#include "stir/info.h"
#include "stir/error.h"


#include <fstream>
#include <list>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::iostream;
using std::list;
using std::find;
#endif


START_NAMESPACE_STIR

void find_inverse(ProjData* proj_data_ptr_out, const ProjData* proj_data_ptr_in);



void 
find_inverse( ProjData* proj_data_ptr_out, const ProjData* proj_data_ptr_in)
{  
  const ProjDataInfo * proj_data_info_ptr =
    proj_data_ptr_in->get_proj_data_info_ptr();
  
  const ProjDataInfo * proj_data_info_ptr_out =
    proj_data_ptr_out->get_proj_data_info_ptr();
  
  const ProjDataFromStream*  projdatafromstream_in = 
    dynamic_cast< const ProjDataFromStream*>(proj_data_ptr_in);
  
  ProjDataFromStream*  projdatafromstream_out = 
    dynamic_cast<ProjDataFromStream*>(proj_data_ptr_out);
  float inv;
  float bin;

  int min_segment_num = ask_num("Minimum segment number to invert",
      proj_data_info_ptr->get_min_segment_num(), proj_data_info_ptr->get_max_segment_num(), 0);
  int max_segment_num = ask_num("Maximum segment number to invert",
      min_segment_num,proj_data_info_ptr->get_max_segment_num(), 
      min_segment_num);
  
 
 float max_in_viewgram =0.F;
  
  for (int segment_num = min_segment_num; segment_num<= max_segment_num;
  segment_num++) 
  {
    SegmentByView<float> segment_by_view = 
      projdatafromstream_in->get_segment_by_view(segment_num);
    const float current_max_in_viewgram = segment_by_view.find_max();
    if ( current_max_in_viewgram >= max_in_viewgram)
     max_in_viewgram = current_max_in_viewgram ;
      else
      continue;
  }
  info(boost::format("Max number in viewgram is: %1%") % max_in_viewgram);

  for (int segment_num = min_segment_num; segment_num<= max_segment_num;
  segment_num++) 
    for ( int view_num = proj_data_info_ptr->get_min_view_num();
    view_num<=proj_data_info_ptr->get_max_view_num(); view_num++)
    {
      Viewgram<float> viewgram_in = projdatafromstream_in->get_viewgram(view_num,segment_num);
      Viewgram<float> viewgram_out = proj_data_info_ptr_out->get_empty_viewgram(view_num,segment_num);

      // the following const was found in the ls_cyl.hs and the same value will
      // be used for thresholding both kappa_0 and kappa_1.
      // threshold  = 10^-4/max_in_sinogram
      // TODO - find out batter way of finding the threshold  
  //    const float max_in_viewgram = 54.0F;

	//segment_by_view.find_max();
   
     const float threshold = 0.0001F*max_in_viewgram;    
      
      for (int i= viewgram_in.get_min_axial_pos_num(); i<=viewgram_in.get_max_axial_pos_num();i++)
	for (int j= viewgram_in.get_min_tangential_pos_num(); j<=viewgram_in.get_max_tangential_pos_num();j++)
	{
	  bin= viewgram_in[i][j];
	 	 	 
	  if (bin >= threshold)
	  {
	    inv = 1.F/bin;
	    viewgram_out[i][j] = inv;
	  }
	 else
	  {	  
	   inv =1/threshold;
	   viewgram_out[i][j] = inv;
	  }

    	}
	projdatafromstream_out->set_viewgram(viewgram_out);
    }
    
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int 
main(int argc, char **argv)

{
  shared_ptr<ProjData> proj_data_ptr;
  
  if (argc!=3)
  {
    cerr << " USAGE: Inverse_proj_data: input projdata filename, output projdata (extension .s) " << endl;
    return(EXIT_FAILURE);
  }
  else
   { 
      proj_data_ptr = ProjData::read_from_file(argv[1]); 
   }

 const ProjDataInfo * proj_data_info_ptr = 
    proj_data_ptr->get_proj_data_info_ptr();

 shared_ptr<ProjData> proj_data_inv_ptr
   (new ProjDataInterfile (proj_data_info_ptr->create_shared_clone(),argv[2]));


  find_inverse(proj_data_inv_ptr.get(),proj_data_ptr.get());

  return EXIT_SUCCESS;

}
