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
    Copyright (C) 2015, University College London
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
#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjData.h"
#include <vector>
#ifdef STIR_OPENMP
#include "stir/is_null_ptr.h"
#include "stir/DiscretisedDensity.h"
#include <omp.h>
#endif

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
  
  const std::vector<ViewSegmentNumbers> vs_nums_to_process = 
    detail::find_basic_vs_nums_in_subset(*proj_data.get_proj_data_info_ptr(), *symmetries_sptr,
                                         proj_data.get_min_segment_num(), proj_data.get_max_segment_num(),
                                         0, 1/*subset_num, num_subsets*/);

#ifdef STIR_OPENMP
  std::vector< shared_ptr<DiscretisedDensity<3,float> > > local_output_image_sptrs;
#pragma omp parallel shared(proj_data, symmetries_sptr, local_output_image_sptrs)
#endif
  { 
#ifdef STIR_OPENMP
#pragma omp single
    {
      local_output_image_sptrs.resize(omp_get_num_threads(), shared_ptr<DiscretisedDensity<3,float> >());
    }
#pragma omp for schedule(runtime)  
#endif
    // note: older versions of openmp need an int as loop
    for (int i=0; i<static_cast<int>(vs_nums_to_process.size()); ++i)
      {
        const ViewSegmentNumbers vs=vs_nums_to_process[i];

        for (int k=proj_data.get_proj_data_info_ptr()->get_min_tof_pos_num();
      		  k<=proj_data.get_proj_data_info_ptr()->get_max_tof_pos_num();
      		  ++k)
        {
#ifdef STIR_OPENMP
        RelatedViewgrams<float> viewgrams;
#pragma omp critical (BACKPROJECTORBYBIN_GETVIEWGRAMS)
        viewgrams = proj_data.get_related_viewgrams(vs, symmetries_sptr);
#else
        const RelatedViewgrams<float> viewgrams = 
          proj_data.get_related_viewgrams(vs, symmetries_sptr, false, k);
#endif
#ifdef STIR_OPENMP
        const int thread_num=omp_get_thread_num();
        if(is_null_ptr(local_output_image_sptrs[thread_num]))
          local_output_image_sptrs[thread_num].reset(image.get_empty_copy());
        
        back_project(*(local_output_image_sptrs[thread_num]), viewgrams);	  
#else            
        back_project(image, viewgrams);
#endif
        }
      }
  }
#ifdef STIR_OPENMP
  // "reduce" data constructed by threads
  {
    for (int i=0; i<static_cast<int>(local_output_image_sptrs.size()); ++i)
      if(!is_null_ptr(local_output_image_sptrs[i])) // only accumulate if a thread filled something in
        image += *(local_output_image_sptrs[i]);
  }
#endif
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

void
BackProjectorByBin::
back_project(DiscretisedDensity<3, float>& density,
         const Bin &bin)
{
    actual_back_project(density, bin);
}

END_NAMESPACE_STIR
