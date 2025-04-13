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

#include "stir/ArrayFwd.h"

START_NAMESPACE_STIR

class ProjData;

/*!
  \ingroup projdata

  \brief Construct proj-data as a multiple of crystal efficiencies (or singles)

  \param[in,out] proj_data projection data to write output. This needs
     to be an existing object as geometry will be obtained from it.
  \param[in] efficiencies array of factors, one per crystal
  \param[in] global_factor global additional factor to use

  Sets \c proj_data(bin) to the product  \c global_factor times
  the sum of \c efficiencies[c1]*efficiencies[c2] (with \c c1,c2 the crystals in the bin).

  This is useful for normalisation, but also for randoms from singles.

  \warning If TOF data is used, each TOF bin will be set to 1/num_tof_bins the non-TOF value.
  This is appropriate for RFS, but would be confusing when using for normalisation.

  \warning, the name is a bit misleading. This function does currently <strong>not</strong> multiply
  the existing data with the efficiencies, but overwrites it.

*/
void multiply_crystal_factors(ProjData& proj_data, const ArrayType<2, float>& efficiencies, const float global_factor);

END_NAMESPACE_STIR
