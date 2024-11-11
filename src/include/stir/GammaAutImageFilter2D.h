//
//
/*!

  \file
  \ingroup ImageProcessor
  \brief Declaration of class stir::GammaAutImageFilter2D.h

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_GammaCorrectionImageFilter2D_H__
#define __stir_GammaCorrectionImageFilter2D_H__

#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup ImageProcessor
  \brief A class in the DataProcessor hierarchy that implements Gamma correction in 2D.
 */
template <typename elemT>
class GammaCorrectionImageFilter2D : public RegisteredParsingObject<GammaCorrectionImageFilter2D<elemT>,
                                                                   DataProcessor<DiscretisedDensity<3, elemT>>,
                                                                   DataProcessor<DiscretisedDensity<3, elemT>>>
{
private:
  typedef RegisteredParsingObject<GammaCorrectionImageFilter2D<elemT>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>>
      base_type;

public:
  static const char* const registered_name;

  //! Default constructor
  GammaCorrectionImageFilter2D();

  //! Set default parameters
  void set_defaults() override;

  //! Applies the Gamma correction filter to the input density
  void virtual_apply(DiscretisedDensity<3, elemT>& density) const override;

protected:
  Succeeded virtual_set_up(const DiscretisedDensity<3, elemT>& density) override;

private:
  void apply_gamma_correction(VoxelsOnCartesianGrid<elemT>& image, int sx, int sy, int sa) const;
};

END_NAMESPACE_STIR

#endif // __stir_GammaCorrectionImageFilter2D_H__

