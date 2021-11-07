/*!

  \file
  \ingroup projdata

  \brief Declaration of stir::multiply_crystal_factors

  \author Kris Thielemans

*/
/*
  Copyright (C) 2021, University Copyright London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

#include "stir/common.h"

START_NAMESPACE_STIR

class ProjData;
template <int num_dimensions, typename elemT> class Array;

/*!
  \ingroup projdata

  \brief Construct proj-data as a multiple of crystal efficiencies (or singles)

  \param[in,out] proj_data projection data to write output (potentially to read first). This needs
     to be an existing object as geometry will be obtained from it.
  \param[in] efficiencies array of factors, one per crystal
  \param[in] global_factor global additional factor to use

  Sets \c proj_data(bin) to the product  \c global_factor times
  the sum of \c efficiencies[c1]*efficiencies[c2] (with \c c1,c2 the crystals in the bin).

  This is useful for normalisation, but also for randoms from singles.
*/
void multiply_crystal_factors(ProjData& proj_data, const Array<2,float>& efficiencies, const float global_factor);

END_NAMESPACE_STIR
