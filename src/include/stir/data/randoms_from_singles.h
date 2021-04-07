/*!

  \file
  \ingroup singles_buildblock

  \brief declare stir:randoms_from_singles

  \author Kris Thielemans

*/
/*
  Copyright (C) 2021, University Copyright London
  This file is part of STIR.

  This file is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2.0 of the License, or
  (at your option) any later version.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  See STIR/LICENSE.txt for details
*/

#include "stir/common.h"


START_NAMESPACE_STIR

class ProjData;
class SinglesRates;

/*!
  \ingroup singles_buildblock
  \brief Estimate randoms from singles

  This uses the formula \f$ R_{ij}= \tau S_i S_j \f$ (with \f$\tau\f$ the \c coincidence_time_window)
  for finding the randoms-rate in terms of the
  singles-rates. The function then takes duration properly into account.

  \todo Dead-time is currently completely ignored.
  \todo The function currently assumes F-18 half-life.
  \todo The SinglesRates class actually gives total singles, not rates!
*/
void randoms_from_singles(ProjData& proj_data, const SinglesRates& singles, const float coincidence_time_window);

END_NAMESPACE_STIR
