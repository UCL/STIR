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
ForwardProjectorByBin::forward_project(ProjData& proj_data, 
				       const DiscretisedDensity<3,float>& image)
{
  
 // this->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
//			     image_sptr);
    
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    this->get_symmetries_used()->clone();  
  
  for (int segment_num = proj_data.get_min_segment_num(); 
       segment_num <= proj_data.get_max_segment_num(); 
       ++segment_num)
    for (int view= proj_data.get_min_view_num(); 
	 view <= proj_data.get_max_view_num();
	 view++)      
    {       
      ViewSegmentNumbers vs(view, segment_num);
      if (!symmetries_sptr->is_basic(vs))
        continue;
      cerr << "Processing view " << vs.view_num() 
        << " of segment " <<vs.segment_num()
        << endl;
      
      RelatedViewgrams<float> viewgrams = 
        proj_data.get_empty_related_viewgrams(vs, symmetries_sptr);
      forward_project(viewgrams, image);	  
      if (!(proj_data.set_related_viewgrams(viewgrams) == Succeeded::yes))
        error("Error set_related_viewgrams in forward projecting\n");            
    }   
  
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


END_NAMESPACE_STIR
