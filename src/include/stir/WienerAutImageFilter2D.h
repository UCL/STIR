//
//
/*!

  \file
  \ingroup ImageProcessor
  \brief Declaration of class stir::WienerAutImageFilter2D.h

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#ifndef __stir_WienerImageFilter2D_H__
#define __stir_WienerImageFilter2D_H__

#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup ImageProcessor
  \brief A class in the DataProcessor hierarchy that implements the Wiener filter in 2D.
 */
template <typename elemT>
class WienerImageFilter2D : public RegisteredParsingObject<WienerImageFilter2D<elemT>,
                                                           DataProcessor<DiscretisedDensity<3, elemT>>,
                                                           DataProcessor<DiscretisedDensity<3, elemT>>>
{
private:
  typedef RegisteredParsingObject<WienerImageFilter2D<elemT>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>>
      base_type;

public:
  static const char* const registered_name;

  //! Default constructor
  WienerImageFilter2D();

  //! Set default parameters
  void set_defaults() override;

  //! Applies the Wiener filter to the input density
  void virtual_apply(DiscretisedDensity<3, elemT>& density) const override;

protected:
  Succeeded virtual_set_up(const DiscretisedDensity<3, elemT>& density) override;

private:
  void apply_wiener_filter(VoxelsOnCartesianGrid<elemT>& image, int sx, int sy, int sa) const;
};

END_NAMESPACE_STIR

#endif // __stir_WienerImageFilter2D_H__

