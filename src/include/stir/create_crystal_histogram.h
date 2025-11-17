/*!

  \file
  \ingroup projdata

  \brief Declaration of stir::create_prompt_histogram

  \author Markus Jehl

*/
/*
  Copyright (C) 2025, Positrigo
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

#include "stir/ArrayFwd.h"

START_NAMESPACE_STIR

class ProjData;

/*!
  \ingroup projdata

  \brief Construct crystal histogram from projdata

  \param[in,out] histogram array of coincidence counts, one entry per crystal
  \param[in] proj_data projection data

  Sets \c efficiencies[c1] and \c efficiencies[c2] to the sum of \c proj_data(bin) the crystals \c c1 and \c c2 have
  contributed to. This is useful for the singles prompt method for creating randoms from singles (see
  Oliver, Josep F., and M. Rafecas. "Modelling random coincidences in positron emission tomography by using singles
  and prompts: a comparison study." PloS one 11.9 (2016): e0162096).

*/
void create_crystal_histogram(ArrayType<2, float>& histogram, const ProjData& proj_data);

END_NAMESPACE_STIR
