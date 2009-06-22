//
// $Id$
//
/*!

  \file
  \ingroup ImageProcessor
  \brief Implementations for class stir::SeparableCartesianMetzImageFilter

  \author Sanida Mustafovic
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
#include "stir/SeparableCartesianMetzImageFilter.h"
#include "stir/VoxelsOnCartesianGrid.h"


START_NAMESPACE_STIR

  
template <typename elemT>
Succeeded
SeparableCartesianMetzImageFilter<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)

{
/*  if (consistency_check(density) == Succeeded::no)
    return Succeeded::no;
  */
  const VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(density);

  metz_filter = 
    SeparableMetzArrayFilter<3,elemT>(get_metz_fwhms(),
				      get_metz_powers(),
				      image.get_voxel_size(), 
				      get_max_kernel_sizes());
  
  return Succeeded::yes;
  
}


template <typename elemT>
void
SeparableCartesianMetzImageFilter<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& density) const

{     
  //assert(consistency_check(density) == Succeeded::yes);
  metz_filter(density);  
}


template <typename elemT>
void
SeparableCartesianMetzImageFilter<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& out_density, 
	  const DiscretisedDensity<3,elemT>& in_density) const
{
  //assert(consistency_check(in_density) == Succeeded::yes);
  metz_filter(out_density,in_density);
}

#if 0

template <typename elemT>
Succeeded
SeparableCartesianMetzImageFilter<elemT>:: 
consistency_check( const DiscretisedDensity<3, elemT>& image) const
{
  
  //TODO?
  VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(density);
  
  CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
  
  // checks if metz_powers >= 0, also checks if FWHM of the filter
  // is smaller than a sampling interval(to prevent bandwidth of the filter
  // exceed NF of the image)
  if ( metz_powers[0]>=0 && metz_powers[1]>=0 &&metz_powers[1]>=0&& metz_filter.fwhms[0] <=voxel_size.x() &&  metz_filter.fwhms[1] <=voxel_size.y()&&  metz_filter.fwhms[1] <=voxel_size.z())
    
    warning("Filter's fwhm is smaller than a sampling distance in the image\n");
  return Succeeded::yes;
  //else
  //return Succeeded::no;
}
#endif

template <typename elemT>
SeparableCartesianMetzImageFilter<elemT>::
SeparableCartesianMetzImageFilter()
: fwhms(VectorWithOffset<float>(1,3)),
  metz_powers(VectorWithOffset<float>(1,3)),
  max_kernel_sizes(VectorWithOffset<int>(1,3))
{
  set_defaults();
}

template <typename elemT>
VectorWithOffset<float>
SeparableCartesianMetzImageFilter<elemT>:: 
get_metz_fwhms() const
{  return fwhms;}

template <typename elemT>
VectorWithOffset<float> 
SeparableCartesianMetzImageFilter<elemT>::
get_metz_powers() const
{  return metz_powers;}


template <typename elemT>
VectorWithOffset<int> 
SeparableCartesianMetzImageFilter<elemT>::
get_max_kernel_sizes() const
{  return max_kernel_sizes;}

template <typename elemT>
void
SeparableCartesianMetzImageFilter<elemT>::
set_defaults()
{
  base_type::set_defaults();
  fwhms.fill(0);
  metz_powers.fill(0);  
  max_kernel_sizes.fill(-1);
}

template <typename elemT>
void 
SeparableCartesianMetzImageFilter<elemT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Separable Cartesian Metz Filter Parameters");
  this->parser.add_key("x-dir filter FWHM (in mm)", &fwhms[3]);
  this->parser.add_key("y-dir filter FWHM (in mm)", &fwhms[2]);
  this->parser.add_key("z-dir filter FWHM (in mm)", &fwhms[1]);
  this->parser.add_key("x-dir filter Metz power", &metz_powers[3]);
  this->parser.add_key("y-dir filter Metz power", &metz_powers[2]);
  this->parser.add_key("z-dir filter Metz power", &metz_powers[1]);   
  this->parser.add_key("x-dir maximum kernel size", &max_kernel_sizes[3]);
  this->parser.add_key("y-dir maximum kernel size", &max_kernel_sizes[2]);
  this->parser.add_key("z-dir maximum kernel size", &max_kernel_sizes[1]);
  this->parser.add_stop_key("END Separable Cartesian Metz Filter Parameters");
}


template <>
const char * const 
SeparableCartesianMetzImageFilter<float>::registered_name =
  "Separable Cartesian Metz";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template class SeparableCartesianMetzImageFilter<float>;

END_NAMESPACE_STIR



