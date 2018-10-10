//
//
/*!

  \file

  \brief 

  \author Sanida Mustafovic
  \author Kris Thielemans
  
*/
/*
    Copyright (C) 2000- 2001, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir_experimental/DAVImageFilter3D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/DiscretisedDensity.h"

// TODO REMOVE
#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ifstream;
using std::ofstream;
using std::fstream;
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

template <typename elemT>
DAVImageFilter3D<elemT>:: DAVImageFilter3D(const CartesianCoordinate3D<int>& mask_radius)
{
  mask_radius_x = mask_radius.x();
  mask_radius_y = mask_radius.y();
  mask_radius_z = mask_radius.z();
}

template <typename elemT>
DAVImageFilter3D<elemT>:: DAVImageFilter3D()
{
  set_defaults();
}

template <typename elemT>
Succeeded
DAVImageFilter3D<elemT>::virtual_set_up (const DiscretisedDensity<3,elemT>& density)
{

  //if (consistency_check(density) == Succeeded::no)
  //    return Succeeded::no;
   dav_filter = 
     DAVArrayFilter3D<elemT>(Coordinate3D<int>
     (mask_radius_z, mask_radius_y, mask_radius_x));

   return Succeeded::yes;
}

template <typename elemT>
void
DAVImageFilter3D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
  //assert(consistency_check(density) == Succeeded::yes);
  dav_filter(density);
   
}

template <typename elemT>
void
DAVImageFilter3D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& out_density, const DiscretisedDensity<3, elemT>& in_density) const
{
  //assert(consistency_check(in_density) == Succeeded::yes);
 // cerr << mask_radius_x << "   "  << mask_radius_y << endl;
  dav_filter(out_density,in_density);
   
}

template <typename elemT>
void
DAVImageFilter3D<elemT>::set_defaults()
{
  mask_radius_x = 0;
  mask_radius_y = 0;
  mask_radius_z = 0;
}

template <typename elemT>
void 
DAVImageFilter3D<elemT>::initialise_keymap()
{
  parser.add_start_key("DAV Filter Parameters");
  parser.add_key("mask radius x", &mask_radius_x);
  parser.add_key("mask radius y", &mask_radius_y);
  parser.add_key("mask radius z", &mask_radius_z);
  parser.add_stop_key("END DAV Filter Parameters");
}


const char * const 
DAVImageFilter3D<float>::registered_name =
  "DAV";

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

#if 0
// registration business moved to local_buildblock_registries.cxx

// Register this class in the ImageProcessor registry
// At the same time, the compiler will instantiate DAVImageFilter3D<float>
static DAVImageFilter3D<float>::RegisterIt dummy;
#endif

template DAVImageFilter3D<float>;


END_NAMESPACE_STIR

