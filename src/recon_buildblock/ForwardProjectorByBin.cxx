//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for ForwardProjectorByBin

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR


ForwardProjectorByBin::ForwardProjectorByBin()
{
}

ForwardProjectorByBin::~ForwardProjectorByBin()
{
}

void 
ForwardProjectorByBin::forward_project(ProjData* proj_data_ptr, 
		  const shared_ptr<DiscretisedDensity<3,float> > image_sptr)
{
  
  const VoxelsOnCartesianGrid<float> * vox_im_ptr = 0;
  vox_im_ptr = dynamic_cast<const VoxelsOnCartesianGrid<float> *> (image_sptr.get());
  VoxelsOnCartesianGrid<float> * vox_image_ptr = 
                   const_cast<VoxelsOnCartesianGrid<float>*>(vox_im_ptr);

  vox_image_ptr->set_origin(Coordinate3D<float>(0,0,0));
  // use shared_ptr such that it cleans up automatically
  
 // this->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
//			     image_sptr);
    

  do_segments(*vox_image_ptr, *proj_data_ptr,
    proj_data_ptr->get_min_segment_num(), proj_data_ptr->get_max_segment_num(), 
    proj_data_ptr->get_min_view_num(), 
    proj_data_ptr->get_max_view_num(),
    proj_data_ptr->get_min_tangential_pos_num(), 
    proj_data_ptr->get_max_tangential_pos_num());
  
  
}

void 
ForwardProjectorByBin::forward_project(RelatedViewgrams<float>& viewgrams, 
				 const DiscretisedDensity<3,float>& image)
{
  forward_project(viewgrams, image,
                  viewgrams.get_min_axial_pos_num(),
		  viewgrams.get_max_axial_pos_num(),
		  viewgrams.get_min_tangential_pos_num(),
		  viewgrams.get_max_tangential_pos_num());
}

void ForwardProjectorByBin::forward_project
  (RelatedViewgrams<float>& viewgrams, 
   const DiscretisedDensity<3,float>& image,
   const int min_axial_pos_num, 
   const int max_axial_pos_num)
{
  forward_project(viewgrams, image,
             min_axial_pos_num,
	     max_axial_pos_num,
	     viewgrams.get_min_tangential_pos_num(),
	     viewgrams.get_max_tangential_pos_num());
}

void 
ForwardProjectorByBin::
forward_project(RelatedViewgrams<float>& viewgrams, 
		     const DiscretisedDensity<3,float>& density,
		     const int min_axial_pos_num, const int max_axial_pos_num,
		     const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  start_timers();
  actual_forward_project(viewgrams, density,
             min_axial_pos_num,
	     max_axial_pos_num,
	     min_tangential_pos_num,
	     max_tangential_pos_num);
  stop_timers();
}

void
ForwardProjectorByBin::do_segments(const VoxelsOnCartesianGrid<float>& image, 
            ProjData& proj_data,
	    const int start_segment_num, const int end_segment_num,
	    const int start_view, const int end_view,
	    const int start_tangential_pos_num, const int end_tangential_pos_num)
{
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    this->get_symmetries_used()->clone();  
  
  list<ViewSegmentNumbers> already_processed;
  
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    for (int view= start_view; view<=end_view; view++)      
    {       
      ViewSegmentNumbers vs(view, segment_num);
      symmetries_sptr->find_basic_view_segment_numbers(vs);
      if (std::find(already_processed.begin(), already_processed.end(), vs)
        != already_processed.end())
        continue;
      
      already_processed.push_back(vs);
      
      cerr << "Processing view " << vs.view_num() 
        << " of segment " <<vs.segment_num()
        << endl;
      
      RelatedViewgrams<float> viewgrams = 
        proj_data.get_empty_related_viewgrams(vs, symmetries_sptr,false);
      this->forward_project(viewgrams, image,
        viewgrams.get_min_axial_pos_num(),
        viewgrams.get_max_axial_pos_num(),
        start_tangential_pos_num, end_tangential_pos_num);	  
      if (!(proj_data.set_related_viewgrams(viewgrams) == Succeeded::yes))
        error("Error set_related_viewgrams\n");            
    }   
}

END_NAMESPACE_STIR
