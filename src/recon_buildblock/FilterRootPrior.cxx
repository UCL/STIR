//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief  implementation of the FilterRootPrior class 
    
  \author Kris Thielemans
  \author Sanida Mustafovic      
  $Date$        
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/FilterRootPrior.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ImageProcessor.h"

START_NAMESPACE_STIR

template <typename elemT>
FilterRootPrior<elemT>::FilterRootPrior()
{
  set_defaults();
}


template <typename elemT>
FilterRootPrior<elemT>::FilterRootPrior(ImageProcessor<3,elemT>* filter, float penalisation_factor_v)
:  filter_ptr(filter)
{
  penalisation_factor = penalisation_factor_v;
}


template <typename elemT>
void 
FilterRootPrior<elemT>::compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
						     const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  assert(  prior_gradient.get_index_range() == current_image_estimate.get_index_range());  
  if (penalisation_factor==0 || filter_ptr==0)
  {
    prior_gradient.fill(0);
    return;
  }
  
  // TODO we don't need the cast stuff here
  
  const VoxelsOnCartesianGrid<float>& current_image_cast =
    dynamic_cast< const VoxelsOnCartesianGrid<float> &>(current_image_estimate);
  
  VoxelsOnCartesianGrid<float>& prior_gradient_cast =
    dynamic_cast<VoxelsOnCartesianGrid<float> &>(prior_gradient);
  
  
  VoxelsOnCartesianGrid<float> filtered_image = current_image_cast;
  
  
  filter_ptr->apply(filtered_image,current_image_estimate);  
  
  
  for (int z=prior_gradient_cast.get_min_z();z<= prior_gradient_cast.get_max_z();z++)
    for (int y=prior_gradient_cast.get_min_y();y<= prior_gradient_cast.get_max_y();y++)
      for (int x=prior_gradient_cast.get_min_x();x<= prior_gradient_cast.get_max_x();x++)       
      {
	prior_gradient_cast[z][y][x]=
	  penalisation_factor * 
	  (current_image_cast[z][y][x]/filtered_image[z][y][x] - 1);	        
      }  
      
      
}


template <typename elemT>
void 
FilterRootPrior<elemT>::initialise_keymap()
{
  parser.add_start_key("FilterRootPrior Parameters");
  parser.add_key("penalisation_factor", &penalisation_factor);
  parser.add_parsing_key("Filter type", &filter_ptr); 
  parser.add_stop_key("END FilterRootPrior Parameters");
}


template <typename elemT>
void
FilterRootPrior<elemT>::set_defaults()
{
  filter_ptr = 0;
  penalisation_factor = 0;
  
}


const char * const 
FilterRootPrior<float>::registered_name =
  "FilterRootPrior";

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


#if 0
// registration stuff moved to recon_buildblock_registries.cxx

// Register this class in the ImageProcessor registry
static FilterRootPrior<float>::RegisterIt dummy;
#endif

template FilterRootPrior<float>;

END_NAMESPACE_STIR

