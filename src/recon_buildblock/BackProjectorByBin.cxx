//
//
/*!
  \file
  \ingroup projection

  \brief non-inline implementations for stir::BackProjectorByBin

  \author Kris Thielemans
  \author PARAPET project
  
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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


#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjData.h"

START_NAMESPACE_STIR

BackProjectorByBin::BackProjectorByBin()
{
}

BackProjectorByBin::~BackProjectorByBin()
{
}

void 
BackProjectorByBin::back_project(DiscretisedDensity<3,float>& image,
				 const ProjData& proj_data)
{
    
  shared_ptr<DataSymmetriesForViewSegmentNumbers> 
    symmetries_sptr(this->get_symmetries_used()->clone());  
  
  for (int segment_num = proj_data.get_min_segment_num(); 
       segment_num <= proj_data.get_max_segment_num(); 
       ++segment_num)
    for (int view_num= proj_data.get_min_view_num(); 
	 view_num <= proj_data.get_max_view_num();
	 ++view_num)      
    {       
      ViewSegmentNumbers vs(view_num, segment_num);
      if (!symmetries_sptr->is_basic(vs))
        continue;
      
      const RelatedViewgrams<float> viewgrams = 
        proj_data.get_related_viewgrams(vs, symmetries_sptr);
      back_project(image, viewgrams);	  
    }
}

void 
BackProjectorByBin::back_project( DiscretisedDensity<3,float>& image,
				  const RelatedViewgrams<float>& viewgrams)
{
  back_project(image,viewgrams,
                  viewgrams.get_min_axial_pos_num(),
		  viewgrams.get_max_axial_pos_num(),
		  viewgrams.get_min_tangential_pos_num(),
		  viewgrams.get_max_tangential_pos_num());
}

void BackProjectorByBin::back_project
  (DiscretisedDensity<3,float>& image,
   const RelatedViewgrams<float>& viewgrams,    
   const int min_axial_pos_num, 
   const int max_axial_pos_num)
{
  back_project(image,viewgrams,
             min_axial_pos_num,
	     max_axial_pos_num,
	     viewgrams.get_min_tangential_pos_num(),
	     viewgrams.get_max_tangential_pos_num());
}

void 
BackProjectorByBin::
back_project(DiscretisedDensity<3,float>& density,
	     const RelatedViewgrams<float>& viewgrams, 		    
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
      error("BackProjectorByBin::back_project called with incorrect related_viewgrams. Problem with symmetries!\n");
    
    for (RelatedViewgrams<float>::const_iterator iter = viewgrams.begin();
	 iter != viewgrams.end();
	 ++iter)
      {
	ViewSegmentNumbers vs(iter->get_view_num(), iter->get_segment_num());
	get_symmetries_used()->find_basic_view_segment_numbers(vs);
	if (vs != basic_vs)
	  error("BackProjectorByBin::back_project called with incorrect related_viewgrams. Problem with symmetries!\n");
    }
  }

  actual_back_project(density,viewgrams,
             min_axial_pos_num,
	     max_axial_pos_num,
	     min_tangential_pos_num,
	     max_tangential_pos_num);
  stop_timers();
}


END_NAMESPACE_STIR
