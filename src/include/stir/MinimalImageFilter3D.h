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

#ifndef __stir_MinimalImageFilter3D_H__
#define __stir_MinimalImageFilter3D_H__

#include "stir/DataProcessor.h"
#include "stir/MinimalArrayFilter3D.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <typename coordT>
class CartesianCoordinate3D;

/*!
  \ingroup ImageProcessor
  \brief A class in the ImageProcessor hierarchy that implements minimal
  filtering.

  As it is derived from RegisteredParsingObject, it implements all the
  necessary things to parse parameter files etc.
 */
template <typename elemT>
class MinimalImageFilter3D : public RegisteredParsingObject<MinimalImageFilter3D<elemT>,
                                                            DataProcessor<DiscretisedDensity<3, elemT>>,
                                                            DataProcessor<DiscretisedDensity<3, elemT>>>
{
private:
  typedef RegisteredParsingObject<MinimalImageFilter3D<elemT>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>>
      base_type;

public:
  static const char* const registered_name;

  MinimalImageFilter3D();

  MinimalImageFilter3D(const CartesianCoordinate3D<int>& mask_radius);

private:
  MinimalArrayFilter3D<elemT> minimal_filter;
  int mask_radius_x;
  int mask_radius_y;
  int mask_radius_z;

  void set_defaults() override;
  void initialise_keymap() override;

  Succeeded virtual_set_up(const DiscretisedDensity<3, elemT>& density) override;
  void virtual_apply(DiscretisedDensity<3, elemT>& density, const DiscretisedDensity<3, elemT>& in_density) const override;
  void virtual_apply(DiscretisedDensity<3, elemT>& density) const override;
};

END_NAMESPACE_STIR

#endif // MinimalImageFilter3D
