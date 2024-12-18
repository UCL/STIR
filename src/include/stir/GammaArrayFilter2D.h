//
//
/*!
  \file
  \ingroup Array
  \brief Declaration of class stir::GammaArrayFilter2D

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_GammaArrayFilter2D_H__
#define __stir_GammaArrayFilter2D_H__

#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"

START_NAMESPACE_STIR

/*!
  \ingroup Array
  \brief Gamma correction filter for 2D slices in a 3D volume.

  This filter enhances image contrast by adjusting pixel intensities using gamma correction.
  The algorithm operates on each 2D slice (axial direction) independently and involves:

  1. **Normalization**: Each pixel value in the slice is normalized to [0, 1]:
     \f[
     \text{normalized}(i, j) = \frac{\text{input}(i, j) - \text{min\_val}}{\text{max\_val} - \text{min\_val}}
     \f]
     where \c min_val and \c max_val are the minimum and maximum pixel values in the slice.

  2. **Average Pixel Value Calculation**: The filter computes the average pixel value across all pixels in the normalized slice
 that have absolute values greater than 0.1. This average value is used to determine the gamma exponent.

  3. **Gamma Calculation**: Determines the gamma exponent using:
     \f[
     \gamma = \frac{\log(0.25)}{\log(\text{averagePixelValue})}
     \f]
     where \c 0.25 is the target average intensity level for contrast adjustment.

  4. **Gamma Correction**: Adjusts pixel values using:
     \f[
     \text{corrected}(i, j) = \text{normalized}(i, j)^{\gamma}
     \f]

  5. **Rescaling**: Converts the corrected values back to their original range:
     \f[
     \text{output}(i, j) = \text{corrected}(i, j) \times (\text{max\_val} - \text{min\_val}) + \text{min\_val}
     \f]

  ### Edge Handling
  - The filter processes each 2D slice independently and does not apply padding. Therefore, the edges of the slices are fully
 processed without boundary effects.

 ### Gamma Filter Configuration
  This filter is fully automated and does not require any parameters.
  To enable it in the reconstruction process, include the following in the parameter file:
  \code
  post-filter type := Gamma
  Gamma Filter Parameters :=
  End Gamma Filter Parameters :=
  \endcode

  \note This filter operates on each axial slice independently, and does not take into account neighboring slices. It is
 effectively a 2D filter applied slice-by-slice on a 3D volume.

 */

template <typename elemT>
class GammaArrayFilter2D : public ArrayFunctionObject_2ArgumentImplementation<3, elemT>
{
public:
  explicit GammaArrayFilter2D();
  bool is_trivial() const override;

private:
  void do_it(Array<3, elemT>& out_array, const Array<3, elemT>& in_array) const override;
};

END_NAMESPACE_STIR

#endif
