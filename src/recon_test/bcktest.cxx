//
// $Id$ : $Date$
//

/*!
  \file
  \ingroup utilities

  \brief Test programme for back projection

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  
  \date $Date$

  \version $Revision$
*/

#include "recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "display.h"
#include "interfile.h"
#include "ProjDataInfoCylindricalArcCorr.h"
#include "ProjDataFromStream.h"
#include "ProjDataInfo.h"
// for ask_filename...
#include "utilities.h"
#include "IndexRange3D.h"
#include "RelatedViewgrams.h"


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


//USING_NAMESPACE_STD

START_NAMESPACE_TOMO

void
do_segments(DiscretisedDensity<3,float>& image, 
            ProjData& proj_data_org,
	    const int start_segment_num, const int end_segment_num,
	    const int start_axial_pos_num, const int end_axial_pos_num,
	    const int start_tang_pos_num,const int end_tang_pos_num,
	    const int start_view, const int end_view,
	    BackProjectorByBin* back_projector_ptr,
	    bool fill_with_1)
{
  
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    back_projector_ptr->get_symmetries_used()->clone();
  
  
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
	//proj_data_org.get_empty_related_viewgrams(vs.view_num(),vs.segment_num(), symmetries_sptr);
      
      RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams_empty.begin();
      while(r_viewgrams_iter!=viewgrams_empty.end())
      {
	Viewgram<float>&  single_viewgram = *r_viewgrams_iter;
	if (start_view <= single_viewgram.get_view_num() && 
	  single_viewgram.get_view_num() <= end_view &&
	  single_viewgram.get_segment_num() >= start_segment_num &&
	  single_viewgram.get_segment_num() <= end_segment_num)
	{
	  single_viewgram.fill(1.F);
	}
	r_viewgrams_iter++;	  
      } 
      
      back_projector_ptr->back_project(image,viewgrams_empty,
	max(start_axial_pos_num, viewgrams_empty.get_min_axial_pos_num()), 
	min(end_axial_pos_num, viewgrams_empty.get_max_axial_pos_num()),
	start_tang_pos_num, end_tang_pos_num);
    }
    else
    {
      RelatedViewgrams<float> viewgrams = 
	proj_data_org.get_related_viewgrams(vs,
	//proj_data_org.get_related_viewgrams(vs.view_num(),vs.segment_num(),
	symmetries_sptr);
      RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams.begin();
      
      while(r_viewgrams_iter!=viewgrams.end())
      {
	Viewgram<float>&  single_viewgram = *r_viewgrams_iter;
	{	  
	  if (start_view <= single_viewgram.get_view_num() && 
	    single_viewgram.get_view_num() <= end_view &&
	  single_viewgram.get_segment_num() >= start_segment_num &&
	  single_viewgram.get_segment_num() <= end_segment_num)
	  {
	    // ok
	  }
	  else
	  { 
	    // set to 0 to prevent it being backprojected
	    single_viewgram.fill(0);
	  }
	}
	++r_viewgrams_iter;
      }
	
      back_projector_ptr->back_project(image,viewgrams,
	  max(start_axial_pos_num, viewgrams.get_min_axial_pos_num()), 
	  min(end_axial_pos_num, viewgrams.get_max_axial_pos_num()),
	  start_tang_pos_num, end_tang_pos_num);      
    } // fill
  } // for view_num, segment_num    
    
}

END_NAMESPACE_TOMO



USING_NAMESPACE_TOMO
int
main(int argc, char **argv)
{  
  shared_ptr<ProjData> proj_data_ptr;
  bool fill;

  switch(argc)
  {
  case 2:
    { 
      proj_data_ptr = ProjData::read_from_file(argv[1]); 
      fill = ask("Do you want to backproject all 1s (Y) or the data (N) ?", true);
      break;
    }
  case 1:
    {
      shared_ptr<ProjDataInfo> data_info= ProjDataInfo::ask_parameters();
      // create an empty ProjDataFromStream object
      // such that we don't have to differentiate between code later on
      proj_data_ptr = 
	new ProjDataFromStream (data_info,static_cast<iostream *>(NULL));
      fill = true;
      break;
    }
  default:
    {
      cerr <<"Usage: " << argv[0] << "[proj_data_file]\n";
      exit(EXIT_FAILURE);
    }
  }
    
    

  
  bool disp = ask("Display images ?", false);
  
  bool save = ask("Save  images ?", true);
    
  // KT 14/08/98 write profiles
  bool save_profiles = ask("Save  horizontal profiles ?", false);

 
  const ProjDataInfo * proj_data_info_ptr =
    proj_data_ptr->get_proj_data_info_ptr();
  
  VoxelsOnCartesianGrid<float> * vox_image_ptr =
    new VoxelsOnCartesianGrid<float>(*proj_data_info_ptr);
  shared_ptr<DiscretisedDensity<3,float> > image_sptr = vox_image_ptr;
  
  
  BackProjectorByBinUsingInterpolation * bck_projector_ptr = 
    new BackProjectorByBinUsingInterpolation(proj_data_info_ptr->clone(), image_sptr);
 
  do
  {    
    
    int min_segment_num = ask_num("Minimum segment number to backproject",
      proj_data_info_ptr->get_min_segment_num(), proj_data_info_ptr->get_max_segment_num(), 0);
    int max_segment_num = ask_num("Maximum segment number to backproject",
      min_segment_num,proj_data_info_ptr->get_max_segment_num(), 
      min_segment_num);
    
    // find max_axial_pos_num in the range of segments
    // TODO relies on axial_pos_num starting from 0
    assert(proj_data_info_ptr->get_min_axial_pos_num(0) == 0);
#if 1  
    int max_axial_pos_num;
    if (min_segment_num <= 0 && 0 <= max_segment_num)
    {
      // all axial_poss are addressed for segment 0
      max_axial_pos_num = proj_data_info_ptr->get_max_axial_pos_num(0);
    }
    else 
    {
      if (min_segment_num>0) // which implies max_segment_num>0
	max_axial_pos_num = proj_data_info_ptr->get_max_axial_pos_num(min_segment_num);      
      else // min_segment_num <= max_segment_num < 0      
	max_axial_pos_num = proj_data_info_ptr->get_max_axial_pos_num(max_segment_num);
    }
    
    const int start_axial_pos_num = 
      ask_num("Start axial_pos", 0, max_axial_pos_num, 0);
    const int end_axial_pos_num = 
      ask_num("End   axial_pos", start_axial_pos_num, max_axial_pos_num, max_axial_pos_num);
#else
    const int min_axial_pos_num = proj_data_info_ptr->get_min_axial_pos_num(segment_num);  
    const int max_axial_pos_num = proj_data_info_ptr->get_max_axial_pos_num(segment_num);  
    const int start_axial_pos_num = 
      ask_num("Start axial_pos", min_axial_pos_num, max_axial_pos_num, min_axial_pos_num);
    const int end_axial_pos_num = 
      ask_num("End   axial_pos", start_axial_pos_num, max_axial_pos_num, max_axial_pos_num);
#endif
    
    // TODO this message is symmetry specific
    const int nviews = proj_data_info_ptr->get_num_views();
    cerr << "Special views are at 0, "
      << nviews/4 <<", " << nviews/2 <<", " << nviews/4*3 << endl;
    
    int start_view = ask_num("Start view", 0, nviews-1, 0);
    int end_view = ask_num("End   view", 0, nviews-1, nviews-1);
    
    //const int start_tang_pos_num = 
    //  ask_num("Start tang_pos", proj_data_info_ptr->get_min_tangential_pos_num(), proj_data_info_ptr->get_max_tangential_pos_num(),proj_data_info_ptr->get_min_tangential_pos_num());
    const int end_tang_pos_num = 
      ask_num("Max  tang_pos", 0, proj_data_info_ptr->get_max_tangential_pos_num(),proj_data_info_ptr->get_max_tangential_pos_num());
    const int start_tang_pos_num = -end_tang_pos_num;

    if (!ask("Add this backprojection to image of previous run ?", false))
      image_sptr->fill(0);
    
    CPUTimer timer;
    timer.reset();
    timer.start();

    do_segments(*image_sptr, 
      *proj_data_ptr, 
      min_segment_num, max_segment_num,
      start_axial_pos_num,end_axial_pos_num,
      start_tang_pos_num,end_tang_pos_num,
      start_view, end_view,
      bck_projector_ptr,
      fill);  
    
    timer.stop();
    cerr << timer.value() << " s CPU time"<<endl;
    cerr << "min and max in image " << image_sptr->find_min()
      << ", " << image_sptr->find_max() << endl;
    
    if (disp)
      display(*image_sptr);
    
    if (save)
    {
      char* file = "bcktest_new";
      cerr <<"  - Saving " << file << endl;
      write_basic_interfile(file, *image_sptr);
      
    }

    if (save_profiles)
    {
      
      cerr << "Writing horizontal profiles to bcktest.prof" << endl;
      ofstream profile("bcktest.prof");
      if (!profile)
      { cerr << "Couldn't open " << "bcktest.prof"; }
      
      for (int z=vox_image_ptr->get_min_z(); z<= vox_image_ptr->get_max_z(); z++) 
      { 
	for (int x=vox_image_ptr->get_min_x(); x<= vox_image_ptr->get_max_x(); x++)
	  profile<<(*vox_image_ptr)[z][0][x]<<" ";
	profile << "\n";
      }
    }
    
  }
  while (ask("One more ?", true));

  return  EXIT_SUCCESS;
}


