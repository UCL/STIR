//
// $Id$
//
/*!

  \file

  \brief Implementation of class PostsmoothingForwardProjectorByBin

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/recon_buildblock/PostsmoothingForwardProjectorByBin.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"

START_NAMESPACE_STIR
const char * const 
PostsmoothingForwardProjectorByBin::registered_name =
  "Post Smoothing";


void
PostsmoothingForwardProjectorByBin::
set_defaults()
{
  tang_kernel_double.resize(0);
  ax_kernel_double.resize(0);

  original_forward_projector_ptr = 0;
  tang_kernel = VectorWithOffset<float>();
  ax_kernel = VectorWithOffset<float>();
  smooth_segment_0_axially= false;
}

void
PostsmoothingForwardProjectorByBin::
initialise_keymap()
{
  parser.add_start_key("Post Smoothing Forward Projector");
  parser.add_stop_key("End Post Smoothing Forward Projector");
  parser.add_parsing_key("Original Forward projector type", &original_forward_projector_ptr);
  parser.add_key("Forward projector postsmoothing kernel", &tang_kernel_double);
  parser.add_key("Forward projector postsmoothing tangential kernel", &tang_kernel_double);
  parser.add_key("Forward projector postsmoothing axial kernel", &ax_kernel_double);
  parser.add_key("Forward projector postsmoothing smooth segment 0 axially", &smooth_segment_0_axially);
}

bool
PostsmoothingForwardProjectorByBin::
post_processing()
{
  if (is_null_ptr(original_forward_projector_ptr))
  {
    warning("Post Smoothing Forward Projector: original forward projector needs to be set");
    return true;
  }
  if (tang_kernel_double.size()>0)
  {
    const int max_kernel_num = tang_kernel_double.size()/2;
    const int min_kernel_num = max_kernel_num - tang_kernel_double.size()+1;
    tang_kernel.grow(min_kernel_num, max_kernel_num);

    int i=min_kernel_num;
    int j=0;
    while(i<=max_kernel_num)
         {
           tang_kernel[i++] =
             static_cast<float>(tang_kernel_double[j++]);
         }
  }
  if (ax_kernel_double.size()>0)
  {
    const int max_kernel_num = ax_kernel_double.size()/2;
    const int min_kernel_num = max_kernel_num - ax_kernel_double.size()+1;
    ax_kernel.grow(min_kernel_num, max_kernel_num);

    int i=min_kernel_num;
    int j=0;
    while(i<=max_kernel_num)
         {
           ax_kernel[i++] =
             static_cast<float>(ax_kernel_double[j++]);
         }
  }
  return false;
}

PostsmoothingForwardProjectorByBin::
  PostsmoothingForwardProjectorByBin()
{
  set_defaults();
}

PostsmoothingForwardProjectorByBin::
PostsmoothingForwardProjectorByBin(
                       const shared_ptr<ForwardProjectorByBin>& original_forward_projector_ptr,
		       const VectorWithOffset<float>& tangential_kernel,
		       const VectorWithOffset<float>& axial_kernel,
		       const bool smooth_segment_0_axially)
                       : original_forward_projector_ptr(original_forward_projector_ptr),
                         tang_kernel(tangential_kernel),
			 ax_kernel(axial_kernel),
			 smooth_segment_0_axially(smooth_segment_0_axially)
{}

void
PostsmoothingForwardProjectorByBin::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)
{
  original_forward_projector_ptr->set_up(proj_data_info_ptr, image_info_ptr);
}

const DataSymmetriesForViewSegmentNumbers * 
PostsmoothingForwardProjectorByBin::
get_symmetries_used() const
{
  return original_forward_projector_ptr->get_symmetries_used();
}

void 
PostsmoothingForwardProjectorByBin::
actual_forward_project(RelatedViewgrams<float>& viewgrams, 
		  const DiscretisedDensity<3,float>& density,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  original_forward_projector_ptr->forward_project(viewgrams, density,
      min_axial_pos_num, max_axial_pos_num,
      min_tangential_pos_num, max_tangential_pos_num);
  for(RelatedViewgrams<float>::iterator iter = viewgrams.begin();
      iter != viewgrams.end();
      ++iter)
      {
        smooth(*iter, 
               min_axial_pos_num, max_axial_pos_num,
               min_tangential_pos_num, max_tangential_pos_num);
      }

}
 
void 
PostsmoothingForwardProjectorByBin::
smooth(Viewgram<float>& v,
       const int min_axial_pos_num, const int max_axial_pos_num,
       const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  // first do tangentially
  {
    VectorWithOffset<float> new_row(min_tangential_pos_num, max_tangential_pos_num);
    
    for (int ax_pos = min_axial_pos_num; ax_pos<= max_axial_pos_num; ++ax_pos)
      {
	for (int tang_pos = min_tangential_pos_num; tang_pos<= max_tangential_pos_num; ++tang_pos)
	  {
	    new_row[tang_pos] = 0;
	    for (int i=max(tang_pos - v.get_max_tangential_pos_num(), tang_kernel.get_min_index()); 
		 i<= tang_kernel.get_max_index() && i<= tang_pos - v.get_min_tangential_pos_num(); ++i)
	      new_row[tang_pos] += tang_kernel[i] * v[ax_pos][tang_pos-i];
	  }
	// copy new_row to old_row
	for (int tang_pos = min_tangential_pos_num; tang_pos<= max_tangential_pos_num; ++tang_pos)
	  {
	    v[ax_pos][tang_pos] = new_row[tang_pos];
	  }
      }
  }
  // now do axially
  if (!v.get_segment_num()==0 || smooth_segment_0_axially)
  {
    VectorWithOffset<float> new_column(min_axial_pos_num, max_axial_pos_num);
    
    for (int tang_pos = min_tangential_pos_num; tang_pos<= max_tangential_pos_num; ++tang_pos)
      {
	for (int ax_pos = min_axial_pos_num; ax_pos<= max_axial_pos_num; ++ax_pos)
	  {
	    new_column[ax_pos] = 0;
	    for (int i=max(ax_pos - v.get_max_axial_pos_num(), ax_kernel.get_min_index()); 
		 i<= ax_kernel.get_max_index() && i<= ax_pos - v.get_min_axial_pos_num(); ++i)
	      new_column[ax_pos] += ax_kernel[i] * v[ax_pos][tang_pos-i];
	  }
	// copy new_column to old_column
	for (int ax_pos = min_axial_pos_num; ax_pos<= max_axial_pos_num; ++ax_pos)
	  {
	    v[ax_pos][tang_pos] = new_column[ax_pos];
	  }
      }
  }

}



END_NAMESPACE_STIR
