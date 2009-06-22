//
// $Id$
//
/*!

  \file
  \ingroup Array
  \brief Declaration of class stir::SeparableMetzArrayFilter

  \author Sanida Mustafovic
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SeparableMetzArrayFilter_H__
#define __stir_SeparableMetzArrayFilter_H__

#include "stir/SeparableArrayFunctionObject.h"
#include "stir/BasicCoordinate.h"
#include "stir/VectorWithOffset.h"

START_NAMESPACE_STIR



/*!
  \ingroup Array
  \brief Separable Metz filtering in \c n - dimensions
 
  The Metz filter is easiest defined in frequency space. For a \c fwhm \c s and
  power \c P, its (continuous) Fourier transform is given by
  \f[ 
  M(k,s,P) =
  (1 - (1 - G(k, s)^2)^{(P + 1)})/ G(k, s)
  \f]
  where \f$ G(k,s) \f$ is the Fourier transform of a Gaussian with FWHM \c s, 
  normalised such that \f$G(0,s) = 1\f$. 

  For power 0, the Metz filter is just a Gaussian. For higher power, mid-range 
  frequencies are more and more amplified. The first figure shows the FT of the
  Metz filter with \c fwhm 1, for powers 0, 0.5, 1, ... 3 (lowest curve is Gaussian).
  \image html FTMetz.jpg
  \image latex FTMetz.eps width=10cm

  Spatially, the Metz filter has negative lobes. The 2nd figure shows the Metz kernel 
  in space, again with \c fwhm 1, powers 0 (long dashes),1,2,3 (no dashes)
  \image html Metz.jpg
  \image latex Metz.eps width=10cm
  Note that from the definition it follows that 
  \f[ Metz(x,s,P) = Metz(x/s, 1 ,P)/s \f]
  The final figure illustrates the relation between the actual FWHM of the Metz 
  filter and the FWHM of the underlying Gaussian.
  \image html MetzFWHM.jpg
  \image latex MetzFWHM.eps width=10cm

  This implementation discretises the Metz filter currently in the following way.
  it assumes that the input data are band-limited. For such data, it is possible
  to compute the filtering with the continuous Metz filter exactly. This is
  done with linear convolution of the sampled data with samples of the 
  spatial Metz cut off at the same frequency as the input data.

  \warning Currently, this implements a Metz filter cut off at 1/\c sampling_distance.
  \warning The Metz filter does \e not preserve positivity.
 */
template <int num_dimensions, typename elemT>
class SeparableMetzArrayFilter: public SeparableArrayFunctionObject <num_dimensions, elemT>
{
public:
  /*!
  \brief Default constructor
  \warning This currently does not set things properly for a trivial filter.
  */
  SeparableMetzArrayFilter() {}
  
  //! Constructor
  /*! 
  \param fwhms the FWHM of the underlying Gauss 1D filters (in mm)
  \param metz_powers the powers of the 1D Metz filters
  \param sampling_distances in each dimensions (in mm)
  \param max_kernel_sizes maximum number of elements in the kernels.
          -1 means unrestricted

  For each of these parameters, the index range should be from 1 to num_dimensions, 
  with 1 corresponding to the 1st (i.e. slowest) index.

  \warning the fwhms parameter does \c not give the FWHM of the Metz filter, but of
  the underlying Gauss.
  */
  SeparableMetzArrayFilter(const VectorWithOffset<float>& fwhms,
			   const VectorWithOffset<float>& metz_powers,
			   const BasicCoordinate<num_dimensions, float>& sampling_distances,
			   const VectorWithOffset<int>& max_kernel_sizes);
  
  
  
private:
  VectorWithOffset<float> fwhms;
  VectorWithOffset<float> metz_powers;
  BasicCoordinate<num_dimensions, float> sampling_distances;
  VectorWithOffset<int> max_kernel_sizes;
};


END_NAMESPACE_STIR

#endif // SeparableMetzArrayFilter


