//
//
/*
    Copyright (C) 2006 - 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup ImageProcessor
  \brief Implementations for class MinimalImageFilter3D

  \author Kris Thielemans
  \author Charalampos Tsoumpas

*/

#include "stir/MinimalImageFilter3D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/DiscretisedDensity.h"


START_NAMESPACE_STIR

template <typename elemT>
MinimalImageFilter3D<elemT>:: MinimalImageFilter3D(const CartesianCoordinate3D<int>& mask_radius)
{
  mask_radius_x = mask_radius.x();
  mask_radius_y = mask_radius.y();
  mask_radius_z = mask_radius.z();
}

template <typename elemT>
MinimalImageFilter3D<elemT>:: MinimalImageFilter3D()
{
  set_defaults();
}

template <typename elemT>
Succeeded
MinimalImageFilter3D<elemT>::virtual_set_up (const DiscretisedDensity<3,elemT>& density)
{

/*   if (consistency_check(density) == Succeeded::no)
      return Succeeded::no;*/
   minimal_filter = 
     MinimalArrayFilter3D<elemT>(Coordinate3D<int>
     (mask_radius_z, mask_radius_y, mask_radius_x));

   return Succeeded::yes;
}

template <typename elemT>
void
MinimalImageFilter3D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
  //assert(consistency_check(density) == Succeeded::yes);
  minimal_filter(density);   
}

template <typename elemT>
void
MinimalImageFilter3D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& out_density, const DiscretisedDensity<3, elemT>& in_density) const
{
  //assert(consistency_check(in_density) == Succeeded::yes);
  minimal_filter(out_density,in_density);   
}

template <typename elemT>
void
MinimalImageFilter3D<elemT>::set_defaults()
{
  base_type::set_defaults();

  mask_radius_x = 0;
  mask_radius_y = 0;
  mask_radius_z = 0;
}

template <typename elemT>
void 
MinimalImageFilter3D<elemT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Minimal Filter Parameters");
  this->parser.add_key("mask radius x", &mask_radius_x);
  this->parser.add_key("mask radius y", &mask_radius_y);
  this->parser.add_key("mask radius z", &mask_radius_z);
  this->parser.add_stop_key("END Minimal Filter Parameters");
}

template <>
const char * const 
MinimalImageFilter3D<float>::registered_name =
  "Minimal";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

template class MinimalImageFilter3D<float>;

END_NAMESPACE_STIR
