//
//
/*!

  \file
  \ingroup Array
  \brief Declaration of class stir::WienerArrayFilter2D

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_WienerArrayFilter2D_H__
#define __stir_WienerArrayFilter2D_H__

#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"

START_NAMESPACE_STIR

/*!
  \ingroup Array
  \brief Applies a 2D Wiener filter on a 3D input array, slice by slice.

  This function applies a Wiener filter on each 2D slice of a 3D volume independently, using a fixed 3x3 window.
  For each pixel in the slice, the filter estimates the local mean and variance, and uses these values, along with
  the noise variance estimated from the entire slice, to reduce noise while preserving details. The filtered
  output is stored in the `out_array`.

  The formula used for the Wiener filter is:
  \f[
  \text{output}(i, j) = \left(\frac{\text{input}(i, j) - \text{localMean}(i, j)}{\max(\text{localVar}(i, j), \text{noise})}\right)
  \cdot \max(\text{localVar}(i, j) - \text{noise}, 0) + \text{localMean}(i, j)
  \f]

  - `localMean[i][j]` is the mean of the 3x3 neighborhood around pixel `(i, j)`.
  - `localVar[i][j]` is the variance of the 3x3 neighborhood around pixel `(i, j)`.
  - `noise` is the average noise variance estimated over the entire slice.


 \warning The edges of each 2D slice are not processed, as the filter does not have sufficient neighboring pixels for the 3x3
 window.

  ### Wiener Filter Configuration
  This filter is fully automated and does not require any parameters.
  To enable it in the reconstruction process, include the following in the parameter file:
  \code
  post-filter type := Wiener
  Wiener Filter Parameters :=
  End Wiener Filter Parameters :=
  \endcode

  \note This filter operates on each axial slice independently, and does not take into account neighboring slices. It is
 effectively a 2D filter applied slice-by-slice on a 3D volume.

 */

template <typename elemT>
class WienerArrayFilter2D : public ArrayFunctionObject_2ArgumentImplementation<3, elemT>
{
public:
  WienerArrayFilter2D();
  bool is_trivial() const override;

private:
  void do_it(Array<3, elemT>& out_array, const Array<3, elemT>& in_array) const override;
};

END_NAMESPACE_STIR

#endif
