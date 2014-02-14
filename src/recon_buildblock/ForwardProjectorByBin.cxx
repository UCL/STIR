//
//
/*!

  \file
  \ingroup projection

  \brief non-inline implementations for stir::ForwardProjectorByBin

  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013 Kris Thielemans
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

#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include <boost/format.hpp>
#include <iostream>

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
    
  shared_ptr<DataSymmetriesForViewSegmentNumbers> 
    symmetries_sptr(this->get_symmetries_used()->clone());
  
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
      info(boost::format("Processing view %1% of segment %2%") % vs.view_num() % vs.segment_num());
      
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
  if (viewgrams.get_num_viewgrams()==0)
    return;

  start_timers();

  // first check symmetries    
  {
    const ViewSegmentNumbers basic_vs = viewgrams.get_basic_view_segment_num();
        
    if (get_symmetries_used()->num_related_view_segment_numbers(basic_vs) !=
      viewgrams.get_num_viewgrams())
      error("ForwardProjectByBin: forward_project called with incorrect related_viewgrams. Problem with symmetries!\n");
    
    for (RelatedViewgrams<float>::const_iterator iter = viewgrams.begin();
	 iter != viewgrams.end();
	 ++iter)
      {
	ViewSegmentNumbers vs(iter->get_view_num(), iter->get_segment_num());
	get_symmetries_used()->find_basic_view_segment_numbers(vs);
	if (vs != basic_vs)
	  error("ForwardProjectByBin: forward_project called with incorrect related_viewgrams. Problem with symmetries!\n");
    }
  }
  actual_forward_project(viewgrams, density,
             min_axial_pos_num,
	     max_axial_pos_num,
	     min_tangential_pos_num,
	     max_tangential_pos_num);
  stop_timers();
}


END_NAMESPACE_STIR
