//
//
/*!

  \file

  \brief Implementation of class stir::PresmoothingForwardProjectorByBin

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2012, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/recon_buildblock/PresmoothingForwardProjectorByBin.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR
const char * const 
PresmoothingForwardProjectorByBin::registered_name =
  "Pre Smoothing";


void
PresmoothingForwardProjectorByBin::
set_defaults()
{
  original_forward_projector_ptr.reset();
  image_processor_ptr.reset();
}

void
PresmoothingForwardProjectorByBin::
initialise_keymap()
{
  parser.add_start_key("Pre Smoothing Forward Projector Parameters");
  parser.add_stop_key("End Pre Smoothing Forward Projector Parameters");
  parser.add_parsing_key("Original Forward projector type", &original_forward_projector_ptr);
  parser.add_parsing_key("filter type", &image_processor_ptr);
}

bool
PresmoothingForwardProjectorByBin::
post_processing()
{
  if (is_null_ptr(original_forward_projector_ptr))
  {
    warning("Pre Smoothing Forward Projector: original forward projector needs to be set\n");
    return true;
  }
  return false;
}

PresmoothingForwardProjectorByBin::
  PresmoothingForwardProjectorByBin()
{
  set_defaults();
}

PresmoothingForwardProjectorByBin::
PresmoothingForwardProjectorByBin(
                       const shared_ptr<ForwardProjectorByBin>& original_forward_projector_ptr,
		       const shared_ptr<DataProcessor<DiscretisedDensity<3,float> > >& image_processor_ptr)
                       : original_forward_projector_ptr(original_forward_projector_ptr),
			 image_processor_ptr(image_processor_ptr)
{}

PresmoothingForwardProjectorByBin::
~PresmoothingForwardProjectorByBin()
{}

void
PresmoothingForwardProjectorByBin::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)
{
  original_forward_projector_ptr->set_up(proj_data_info_ptr, image_info_ptr);
  if (!is_null_ptr(image_processor_ptr))
    image_processor_ptr->set_up(*image_info_ptr);
}

const DataSymmetriesForViewSegmentNumbers * 
PresmoothingForwardProjectorByBin::
get_symmetries_used() const
{
  return original_forward_projector_ptr->get_symmetries_used();
}

void 
PresmoothingForwardProjectorByBin::
actual_forward_project(RelatedViewgrams<float>& viewgrams, 
		  const DiscretisedDensity<3,float>& density,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  if (!is_null_ptr(image_processor_ptr))
    {
      shared_ptr<DiscretisedDensity<3,float> > filtered_density_ptr(density.get_empty_discretised_density());
      image_processor_ptr->apply(*filtered_density_ptr, density);
      assert(density.get_index_range() == filtered_density_ptr->get_index_range());
      original_forward_projector_ptr->forward_project(viewgrams, *filtered_density_ptr,
						      min_axial_pos_num, max_axial_pos_num,
						      min_tangential_pos_num, max_tangential_pos_num);
    }
  else
    {
      original_forward_projector_ptr->forward_project(viewgrams, density,
						      min_axial_pos_num, max_axial_pos_num,
						      min_tangential_pos_num, max_tangential_pos_num);
    }
}
 


END_NAMESPACE_STIR
