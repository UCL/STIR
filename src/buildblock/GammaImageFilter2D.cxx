//
//
/*!
  \file
  \ingroup ImageProcessor
  \brief Implementations for class stir::GammaImageFilter2D

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/GammaImageFilter2D.h"
#include "stir/DiscretisedDensity.h"
#include <iostream>

START_NAMESPACE_STIR

template <typename elemT>
GammaImageFilter2D<elemT>::GammaImageFilter2D()
{
  std::cout << "Gamma filter initialization" << std::endl;
  set_defaults();
}

template <typename elemT>
Succeeded
GammaImageFilter2D<elemT>::virtual_set_up(const DiscretisedDensity<3, elemT>& density)
{
  gamma_filter = GammaArrayFilter2D<elemT>();
  return Succeeded::yes;
}

template <typename elemT>
void
GammaImageFilter2D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
  gamma_filter(density);
}

template <typename elemT>
void
GammaImageFilter2D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& out_density,
                                         const DiscretisedDensity<3, elemT>& in_density) const
{
  gamma_filter(out_density, in_density);
}

template <typename elemT>
void
GammaImageFilter2D<elemT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename elemT>
void
GammaImageFilter2D<elemT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Gamma Filter Parameters");
  this->parser.add_stop_key("END Gamma Filter Parameters");
}

template <>
const char* const GammaImageFilter2D<float>::registered_name = "Gamma";

#ifdef _MSC_VER
#  pragma warning(disable : 4660)
#endif

// Explicit instantiation
template class GammaImageFilter2D<float>;

END_NAMESPACE_STIR
