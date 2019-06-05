/*!
  \file

  \brief Implementation of class stir::PostsmoothingBackProjectorByBin

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2012, Hammersmith Imanet

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

#include "stir/recon_buildblock/PostsmoothingBackProjectorByBin.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR
const char * const 
PostsmoothingBackProjectorByBin::registered_name =
  "Post Smoothing";


void
PostsmoothingBackProjectorByBin::
set_defaults()
{
  original_back_projector_ptr.reset();
  image_processor_ptr.reset();
}

void
PostsmoothingBackProjectorByBin::
initialise_keymap()
{
  parser.add_start_key("Post Smoothing Back Projector Parameters");
  parser.add_stop_key("End Post Smoothing Back Projector Parameters");
  parser.add_parsing_key("Original Back projector type", &original_back_projector_ptr);
  parser.add_parsing_key("filter type", &image_processor_ptr);
}

bool
PostsmoothingBackProjectorByBin::
post_processing()
{
  if (is_null_ptr(original_back_projector_ptr))
  {
    warning("Pre Smoothing Back Projector: original back projector needs to be set");
    return true;
  }
  return false;
}

PostsmoothingBackProjectorByBin::
  PostsmoothingBackProjectorByBin()
{
  set_defaults();
}

void PostsmoothingBackProjectorByBin::
update_filtered_density_image(DiscretisedDensity<3, float> &density)
{
        image_processor_ptr->apply(*filtered_density_sptr);
        density += *filtered_density_sptr;
        filtered_density_sptr->fill(0.f);
}

BackProjectorByBin*
PostsmoothingBackProjectorByBin::get_original_back_projector_ptr() const
{
    return original_back_projector_ptr.get();
}

PostsmoothingBackProjectorByBin*
PostsmoothingBackProjectorByBin::clone() const
{
    PostsmoothingBackProjectorByBin* sptr(new PostsmoothingBackProjectorByBin(*this));
    sptr->original_back_projector_ptr.reset(this->original_back_projector_ptr->clone());
    return sptr;
}

void PostsmoothingBackProjectorByBin::
init_filtered_density_image(DiscretisedDensity<3, float> &density)
{
    filtered_density_sptr.reset(
                density.get_empty_discretised_density());
    assert(density.get_index_range() == filtered_density_sptr->get_index_range());
}

PostsmoothingBackProjectorByBin::
PostsmoothingBackProjectorByBin(
                       const shared_ptr<BackProjectorByBin>& original_back_projector_ptr,
                       const shared_ptr<DataProcessor<DiscretisedDensity<3,float> > >& image_processor_ptr)
                       : original_back_projector_ptr(original_back_projector_ptr),
                         image_processor_ptr(image_processor_ptr)
{}

PostsmoothingBackProjectorByBin::
~PostsmoothingBackProjectorByBin()
{}

void
PostsmoothingBackProjectorByBin::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)
{
  BackProjectorByBin::set_up(proj_data_info_ptr, image_info_ptr);
  original_back_projector_ptr->set_up(proj_data_info_ptr, image_info_ptr);
  // don't do set_up as image sizes might change
  //if (!is_null_ptr(image_processor_ptr))
  //   image_processor_ptr->set_up(*image_info_ptr);
}

const DataSymmetriesForViewSegmentNumbers * 
PostsmoothingBackProjectorByBin::
get_symmetries_used() const
{
  return original_back_projector_ptr->get_symmetries_used();
}

void 
PostsmoothingBackProjectorByBin::
actual_back_project(DiscretisedDensity<3,float>& density,
                    const RelatedViewgrams<float>& viewgrams, 
                    const int min_axial_pos_num, const int max_axial_pos_num,
                    const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  if (!is_null_ptr(image_processor_ptr))
    {
      shared_ptr<DiscretisedDensity<3,float> > filtered_density_ptr
        (density.get_empty_discretised_density());
      assert(density.get_index_range() == filtered_density_ptr->get_index_range());
      original_back_projector_ptr->back_project(*filtered_density_ptr, viewgrams, 
                                                min_axial_pos_num, max_axial_pos_num,
                                                min_tangential_pos_num, max_tangential_pos_num);
      image_processor_ptr->apply(*filtered_density_ptr);
      density += *filtered_density_ptr;
    }
  else
    {
      original_back_projector_ptr->back_project(density, viewgrams, 
                                                min_axial_pos_num, max_axial_pos_num,
                                                min_tangential_pos_num, max_tangential_pos_num);
    }
}

void
PostsmoothingBackProjectorByBin::
actual_back_project(DiscretisedDensity<3, float> &density,
                    const Bin& bin)
{
    if (!is_null_ptr(image_processor_ptr))
      {
//
        original_back_projector_ptr->back_project(*filtered_density_sptr, bin);

      }
    else
      {
        original_back_projector_ptr->back_project(density, bin);
      }
}

END_NAMESPACE_STIR
