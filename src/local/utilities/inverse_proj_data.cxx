//
// $Id: bcktest.cxx,
//

/*!
  \file
  \ingroup 

  \brief Find the inverse of ProjData


  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  
  \date $Date: 
  \version $Revision: 
*/

#include "recon_buildblock/BackProjectorByBinUsingInterpolation.h"
//#include "recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "recon_buildblock/ProjMatrixByBin.h"
#include "recon_buildblock/BackProjectorByBin.h"
//#include "display.h"
#include "interfile.h"
#include "ProjDataInfoCylindricalArcCorr.h"
#include "ProjDataFromStream.h"
#include "ProjDataInfo.h"
// for ask_filename...
#include "utilities.h"
#include "IndexRange3D.h"
#include "RelatedViewgrams.h"
#include "SegmentByView.h"
#include "VoxelsOnCartesianGrid.h"
#include "Viewgram.h"


#include <fstream>
#include <list>
#include <algorithm>

#ifndef TOMO_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::iostream;
using std::endl;
using std::list;
using std::find;
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_TOMO

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
  cerr << " Max number in viewgram is: " << max_in_viewgram;
  cerr << endl;

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

END_NAMESPACE_TOMO


USING_NAMESPACE_TOMO

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

  ProjDataInfo * data_info = proj_data_info_ptr->clone();

 
 shared_ptr<iostream> sino_stream = new fstream (argv[2], ios::out|ios::binary);
 
 shared_ptr<ProjDataFromStream> proj_data_inv_ptr =
   new ProjDataFromStream (data_info,sino_stream);


  find_inverse(proj_data_inv_ptr.get(),proj_data_ptr.get());

  write_basic_interfile_PDFS_header(argv[2],*proj_data_inv_ptr);

  return EXIT_SUCCESS;

}