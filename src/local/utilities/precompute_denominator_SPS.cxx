
/*
  \file
  \ingroup utilities 
  \brief precomputes denominator for SPS

  \author Sanida Mustafovic

  $Date$ 
  $Revision$ 
  
    This program precomputes denominator for the SPS 
    (separable paraboloidal surrogates)  in ET.
    the denominator is given by :  
    dj = sum Aij gamma h''(yi) where
    h(l) = yi log (l) - l; h''(yi) = -1/yi; 
    gamma = sum Aik;
    
      =>  dj = sum Aij(-i/yi) sum Aik
      
    if there is penalty added to it there is another term in the 
    denominator that includes penalty term
*/

/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/IO/interfile.h"
#include "stir/ProjDataInterfile.h"

#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/utilities.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Viewgram.h"
#include "stir/ProjDataFromStream.h"
#include "stir/SegmentByView.h"


#include <fstream>
#include <list>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::iostream;
using std::endl;
using std::list;
using std::find;
using std::cerr;
using std::endl;
#endif



START_NAMESPACE_STIR

void
do_segments(DiscretisedDensity<3,float>& image, 
            ProjData& proj_data_org,
	    const int start_segment_num, const int end_segment_num,
	    const int start_view, const int end_view,
	    BackProjectorByBin& back_projector,
	    bool fill_with_1);



void 
find_inverse( ProjData& proj_data_out, const ProjData& proj_data_in,
	      const int min_segment_num, const int max_segment_num)
{  
     
  float max_in_projdata =0.F;
  
 
  for (int segment_num = min_segment_num; segment_num<= max_segment_num;
       segment_num++) 
  {
    SegmentByView<float> segment_by_view = 
      proj_data_in.get_segment_by_view(segment_num);
    const float current_max_in_projdata = segment_by_view.find_max();
    if ( current_max_in_projdata >= max_in_projdata)
      max_in_projdata = current_max_in_projdata ;
    else
      continue;
  }
  
  for (int segment_num = min_segment_num; segment_num<= max_segment_num;
       segment_num++) 
    for ( int view_num = proj_data_in.get_min_view_num();
	  view_num<=proj_data_in.get_max_view_num(); view_num++)
    {
      Viewgram<float> viewgram_in = proj_data_in.get_viewgram(view_num,segment_num);
      Viewgram<float> viewgram_out = proj_data_in.get_empty_viewgram(view_num,segment_num);

      // the following const was found in the ls_cyl.hs and the same value will
      // be used for thresholding both kappa_0 and kappa_1.
   
     const float threshold = 0.000001F*max_in_projdata;    
     //cerr << threshold << endl;
      
      for (int i= viewgram_in.get_min_axial_pos_num(); i<=viewgram_in.get_max_axial_pos_num();i++)
	for (int j= viewgram_in.get_min_tangential_pos_num(); j<=viewgram_in.get_max_tangential_pos_num();j++)
	{
	  const float bin= viewgram_in[i][j];
	 	 	 
	  if (bin >= threshold)
	  {
	    viewgram_out[i][j] = 1/bin;
	  }
	 else
	  {	  
	    viewgram_out[i][j] = 1/threshold; 
	  }

    	}
	proj_data_out.set_viewgram(viewgram_out);
    }
    
}

void
do_segments(DiscretisedDensity<3,float>& image, 
            ProjData& proj_data_org,
	    const int start_segment_num, const int end_segment_num,
	    const int start_view, const int end_view,
	    BackProjectorByBin& back_projector,
	    bool fill_with_1)
{
  
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    back_projector.get_symmetries_used()->clone();
  
  
  list<ViewSegmentNumbers> already_processed;
  
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    for (int view= start_view; view<=end_view; view++)
  { 
    ViewSegmentNumbers vs(view, segment_num);
    symmetries_sptr->find_basic_view_segment_numbers(vs);
    if (find(already_processed.begin(), already_processed.end(), vs)
        != already_processed.end())
      continue;

    already_processed.push_back(vs);
    
    cerr << "Processing view " << vs.view_num()
      << " of segment " <<vs.segment_num()
      << endl;
    
    if(fill_with_1 )
    {
      RelatedViewgrams<float> viewgrams_empty= 
	proj_data_org.get_empty_related_viewgrams(vs, symmetries_sptr);

      viewgrams_empty.fill(1.F);
      
      back_projector.back_project(image,viewgrams_empty);
    }
    else
    {
      RelatedViewgrams<float> viewgrams = 
	proj_data_org.get_related_viewgrams(vs,	symmetries_sptr);
      	
      back_projector.back_project(image,viewgrams);      
    } // fill
  } // for view_num, segment_num    
    
}


END_NAMESPACE_STIR





USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{

  if(argc<4 || argc>6)
  {
    cerr<< "Usage: " << argv[0] << "output_image_filename input_proj_data  fwd_allones_projdata [normalisation_projdata [template_image]]\n";
    exit(EXIT_FAILURE);
  }

  const string output_filename = argv[1];
  shared_ptr<ProjData> proj_data_ptr = ProjData::read_from_file(argv[2]);
  shared_ptr<ProjData> fwd_ones = ProjData::read_from_file(argv[3]); 
  shared_ptr<ProjData> norm_proj_data_ptr = 
    argc>=5 ? ProjData::read_from_file(argv[4]) : 0; 

  const ProjDataInfo * proj_data_info_ptr = 
    proj_data_ptr->get_proj_data_info_ptr();
  shared_ptr<DiscretisedDensity<3,float> > image_sptr;
  if (argc >=6)
    {
      image_sptr = DiscretisedDensity<3,float>::read_from_file(argv[5]);
      image_sptr->fill(0);
    }
  else
    image_sptr =
      new VoxelsOnCartesianGrid<float>(*proj_data_info_ptr); 

  
  const int max_segment_num = 
    ask_num("Maximum absolute segment number to backproject",
	    0,proj_data_ptr->get_max_segment_num(), 
	    proj_data_ptr->get_max_segment_num());
  bool projector_type = 
    ask ( " Which back projector do you want to use Ray Tracing (Y) or Solid Angle (N) ", true) ;
  string name ;
  if ( projector_type )
   name = "Ray Tracing";
  else
    name = "Solid Angle";  
 shared_ptr<ProjMatrixByBin> PM = 
    ProjMatrixByBin :: read_registered_object(0, name);

  
  // find the inverse
  shared_ptr<ProjData>  proj_data_inv_ptr;
  {
    shared_ptr< ProjDataInfo >  data_info = proj_data_info_ptr->clone();
    data_info->reduce_segment_range(-max_segment_num, max_segment_num);
    
    const string output_file_name = "inverse.s";
    proj_data_inv_ptr =
      new  ProjDataInterfile(data_info, output_file_name, ios::trunc|ios::out|ios::in|ios::binary);
    
    find_inverse(*proj_data_inv_ptr,*proj_data_ptr,
		 -max_segment_num, max_segment_num);
    
    for (int segment_num = -max_segment_num; segment_num<= max_segment_num;segment_num++) 
      {
	SegmentByView<float> segmnet_0 = proj_data_inv_ptr->get_segment_by_view(segment_num);
	SegmentByView<float> segmnet_1 = fwd_ones->get_segment_by_view(segment_num);
	
	segmnet_0 *=segmnet_1;
	
	if (!is_null_ptr(norm_proj_data_ptr))
	  {
	    segmnet_1 = norm_proj_data_ptr->get_segment_by_view(segment_num);
	    segmnet_0 /=segmnet_1;
	    segmnet_0 /=segmnet_1;
	  }
	if (!(proj_data_inv_ptr->set_segment(segmnet_0) == Succeeded::yes))
	  warning("Error set_segment %d\n", segment_num);   
      }

  } // end of  inverse

 
 
  PM->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),image_sptr);
  shared_ptr<BackProjectorByBin>  bck_projector_ptr  =
    new BackProjectorByBinUsingProjMatrixByBin(PM);   
     
            
  CPUTimer timer;
  timer.reset();
  timer.start();
  
  do_segments(*image_sptr, 
      *proj_data_inv_ptr, 
      -max_segment_num, max_segment_num,
      proj_data_info_ptr->get_min_view_num(), proj_data_info_ptr->get_max_view_num(),
      *bck_projector_ptr,
      false);  
  
  timer.stop();
  cerr << timer.value() << " s CPU time"<<endl;
  cerr << "min and max in image " << image_sptr->find_min()
       << ", " << image_sptr->find_max() << endl;
  

  cerr <<"  - Saving " << output_filename << endl;
  write_basic_interfile(output_filename, *image_sptr);
  
  return EXIT_SUCCESS;
}

