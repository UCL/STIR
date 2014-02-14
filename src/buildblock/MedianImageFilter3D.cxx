//
//
/*!
  \file
  \ingroup ImageProcessor
  \brief Implementations for class stir::MedianImageFilter3D

  \author Sanida Mustafovic
  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

#include "stir/MedianImageFilter3D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/DiscretisedDensity.h"


START_NAMESPACE_STIR

template <typename elemT>
MedianImageFilter3D<elemT>:: MedianImageFilter3D(const CartesianCoordinate3D<int>& mask_radius)
{
  mask_radius_x = mask_radius.x();
  mask_radius_y = mask_radius.y();
  mask_radius_z = mask_radius.z();
}

template <typename elemT>
MedianImageFilter3D<elemT>:: MedianImageFilter3D()
{
  set_defaults();
}

template <typename elemT>
Succeeded
MedianImageFilter3D<elemT>::virtual_set_up (const DiscretisedDensity<3,elemT>& density)
{

/*   if (consistency_check(density) == Succeeded::no)
      return Succeeded::no;*/
   median_filter = 
     MedianArrayFilter3D<elemT>(Coordinate3D<int>
     (mask_radius_z, mask_radius_y, mask_radius_x));

   return Succeeded::yes;
}

template <typename elemT>
void
MedianImageFilter3D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
  //assert(consistency_check(density) == Succeeded::yes);
  median_filter(density);   
}

template <typename elemT>
void
MedianImageFilter3D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& out_density, const DiscretisedDensity<3, elemT>& in_density) const
{
  //assert(consistency_check(in_density) == Succeeded::yes);
  median_filter(out_density,in_density);   
}

template <typename elemT>
void
MedianImageFilter3D<elemT>::set_defaults()
{
  base_type::set_defaults();

  mask_radius_x = 0;
  mask_radius_y = 0;
  mask_radius_z = 0;
}

template <typename elemT>
void 
MedianImageFilter3D<elemT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Median Filter Parameters");
  this->parser.add_key("mask radius x", &mask_radius_x);
  this->parser.add_key("mask radius y", &mask_radius_y);
  this->parser.add_key("mask radius z", &mask_radius_z);
  this->parser.add_stop_key("END Median Filter Parameters");
}

template <>
const char * const 
MedianImageFilter3D<float>::registered_name =
  "Median";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


// Register this class in the ImageProcessor registry
//static MedianImageFilter3D<float>::RegisterIt dummy;

template class MedianImageFilter3D<float>;

END_NAMESPACE_STIR
