//
//
/*!
  \file
  \ingroup ImageProcessor
  \brief Implementations for class stir::WienerImageFilter2D

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/WienerImageFilter2D.h"
#include "stir/CartesianCoordinate2D.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

template <typename elemT>
WienerImageFilter2D<elemT>::WienerImageFilter2D()
{
  std::cout << "Wiener filter start" << std::endl;

  set_defaults();
}

template <typename elemT>
Succeeded
WienerImageFilter2D<elemT>::virtual_set_up(const DiscretisedDensity<3, elemT>& density)
{

  /*   if (consistency_check(density) == Succeeded::no)
        return Succeeded::no;*/
  wiener_filter = WienerArrayFilter2D<elemT>();

  return Succeeded::yes;
}

template <typename elemT>
void
WienerImageFilter2D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
  // assert(consistency_check(density) == Succeeded::yes);
  wiener_filter(density);
}

template <typename elemT>
void
WienerImageFilter2D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& out_density,
                                          const DiscretisedDensity<3, elemT>& in_density) const
{
  // assert(consistency_check(in_density) == Succeeded::yes);
  wiener_filter(out_density, in_density);
}

template <typename elemT>
void
WienerImageFilter2D<elemT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename elemT>
void
WienerImageFilter2D<elemT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Wiener Filter Parameters");
  this->parser.add_stop_key("END Wiener Filter Parameters");
}

template <>
const char* const WienerImageFilter2D<float>::registered_name = "Wiener";

#ifdef _MSC_VER
// prevent warning message on reinstantiation,
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable : 4660)
#endif

// Register this class in the ImageProcessor registry
// static WienerImageFilter2D<float>::RegisterIt dummyWiener;

template class WienerImageFilter2D<float>;

END_NAMESPACE_STIR
