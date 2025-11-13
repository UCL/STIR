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
#include "stir/ProjDataInMemory.h"

START_NAMESPACE_STIR

/*!
  \ingroup projdata

  \brief Construct crystal histogram from prompt projdata

  \param[in,out] prompt_histogram array of coincidence counts, one per crystal
  \param[in] prompt_proj_data projection data created from the listmode prompt events.

  Sets \c efficiencies[c1] and \c efficiencies[c2] to the sum of \c proj_data(bin) the crystals \c c1,c2 have 
  contributed to. This is useful for the singles prompt method for creating randoms from singles (see 
  Oliver, Josep F., and M. Rafecas. "Modelling random coincidences in positron emission tomography by using singles 
  and prompts: a comparison study." PloS one 11.9 (2016): e0162096).

*/
void create_prompt_histogram(ArrayType<2, float>& prompt_histogram, const ProjDataInMemory& prompt_proj_data);

END_NAMESPACE_STIR
