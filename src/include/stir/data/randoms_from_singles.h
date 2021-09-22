/*!

  \file
  \ingroup singles_buildblock

  \brief declare stir:randoms_from_singles

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
class SinglesRates;

/*!
  \ingroup singles_buildblock
  \brief Estimate randoms from singles

  This uses the formula \f$ R_{ij}= \tau S_i S_j \f$ (with \f$\tau\f$ the \c coincidence_time_window)
  for finding the randoms-rate in terms of the
  singles-rates. The function then takes duration properly into account using the following
  procedure.

  We actually have total counts in the singles for sinograms and need total counts in the randoms.
  Assuming the radioactivity distribution is stationary and only decays, and dropping indices \f$i,j\f$,
  we get
  \f[
     R_{ij}(t) = \tau * S_i(0) S_j(0) exp (-2 \lambda t)
  \f]
  Integrating over time, and writing $r$ for the integral of $R$ etc
  \f[
     r_{ij} = \tau S_i(0) S_j(0) \int_{t_1}^{t_2} dt\,exp (-2 \lambda t)
  \f]
  converting that to total singles $s$ in the time frame, we get
  \f[
     r_{ij} = \tau s_i s_j \frac{ \int_{t_1}^{t_2} dt\,exp (-2 \lambda t)}{\left(\int_t1^t2 dt\,exp (-\lambda t)\right)^2}
  \f]

  \todo Dead-time is currently completely ignored.
  \todo The function currently assumes F-18 half-life.
*/
void randoms_from_singles(ProjData& proj_data, const SinglesRates& singles, const float coincidence_time_window);

END_NAMESPACE_STIR
