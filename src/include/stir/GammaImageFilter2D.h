//
//
/*!
  \file
  \ingroup ImageProcessor
  \brief Declaration of class stir::GammaImageFilter2D

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_GammaImageFilter2D_H__
#define __stir_GammaImageFilter2D_H__

#include "stir/DataProcessor.h"
#include "stir/GammaArrayFilter2D.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup ImageProcessor
  \brief A class in the ImageProcessor hierarchy that implements gamma correction filtering.

  This class applies a gamma correction filter to a 3D volume using the GammaArrayFilter3D.
  It is derived from RegisteredParsingObject, allowing it to be configured through parameter files.

  \tparam elemT The element type of the density (e.g., `float`, `double`).
*/
template <typename elemT>
class GammaImageFilter2D : public RegisteredParsingObject<GammaImageFilter2D<elemT>,
                                                          DataProcessor<DiscretisedDensity<3, elemT>>,
                                                          DataProcessor<DiscretisedDensity<3, elemT>>>
{
private:
  typedef RegisteredParsingObject<GammaImageFilter2D<elemT>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>>
      base_type;

public:
  static const char* const registered_name;

  GammaImageFilter2D();

private:
  GammaArrayFilter2D<elemT> gamma_filter;

  void set_defaults() override;
  void initialise_keymap() override;

  Succeeded virtual_set_up(const DiscretisedDensity<3, elemT>& density) override;
  void virtual_apply(DiscretisedDensity<3, elemT>& density, const DiscretisedDensity<3, elemT>& in_density) const override;
  void virtual_apply(DiscretisedDensity<3, elemT>& density) const override;
};

END_NAMESPACE_STIR

#endif
