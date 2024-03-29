//
//
/*
    Copyright (C) 2005- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup ImageProcessor
  \brief Implementations for class stir::TruncateToCylindricalFOVImageProcessor
  \author Kris Thielemans

*/
#include "stir/TruncateToCylindricalFOVImageProcessor.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/recon_array_functions.h"

START_NAMESPACE_STIR

template <>
const char* const TruncateToCylindricalFOVImageProcessor<float>::registered_name = "Truncate To Cylindrical FOV";

template <typename elemT>
void
TruncateToCylindricalFOVImageProcessor<elemT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Truncate To Cylindrical FOV Parameters");
  this->parser.add_key("strictly_less_than_radius", &_strictly_less_than_radius);
  this->parser.add_stop_key("END Truncate To Cylindrical FOV Parameters");
}

template <typename elemT>
void
TruncateToCylindricalFOVImageProcessor<elemT>::set_defaults()
{
  base_type::set_defaults();
  this->_strictly_less_than_radius = true;
  this->_truncate_rim = 0;
}

template <typename elemT>
Succeeded
TruncateToCylindricalFOVImageProcessor<elemT>::virtual_set_up(const DiscretisedDensity<3, elemT>& density)

{
  if (dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, elemT>*>(&density) == 0)
    return Succeeded::no;
  else
    return Succeeded::yes;
}

template <typename elemT>
void
TruncateToCylindricalFOVImageProcessor<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const

{
  truncate_rim(density, this->_truncate_rim, this->_strictly_less_than_radius);
}

template <typename elemT>
void
TruncateToCylindricalFOVImageProcessor<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& out_density,
                                                             const DiscretisedDensity<3, elemT>& in_density) const
{
  out_density = in_density;
  this->virtual_apply(out_density);
}

template <typename elemT>
TruncateToCylindricalFOVImageProcessor<elemT>::TruncateToCylindricalFOVImageProcessor()
{
  this->set_defaults();
}

#ifdef _MSC_VER
// prevent warning message on reinstantiation,
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable : 4660)
#endif

// Register this class in the ImageProcessor registry
// static TruncateToCylindricalFOVImageProcessor<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template class TruncateToCylindricalFOVImageProcessor<float>;

END_NAMESPACE_STIR
