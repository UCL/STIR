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
  \brief Estimate randoms from singles (RFS)

  \param[in,out] proj_data
     Projection data to store output. It needs to be properly initialised with sizes etc.
     If \a coincidence_time_window or
     \a radionuclide_halflife are invalid, they will be determined from the \a proj_data.
  \param[in] singles
     Input value for RFS
  \param[in] coincidence_time_window Scanner coincidence window (in secs). Deprecated.
  \param[in] radionuclide_halflife half-life. Deprecated.

  This uses the formula \f$ R_{ij}= \tau S_i S_j \f$ (with \f$\tau\f$ the \c coincidence_time_window)
  for finding the randoms-rate in terms of the
  singles-rates. The function then takes duration properly into account using the following
  procedure.
  This uses \c isotope_halflife in the computation of \f$\lambda\f$. Default value is -1 and the function will
  extract \c isotope_halflife from \c proj_data. If set, will use the passed value.

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
     r_{ij} = \tau s_i s_j \frac{ \int_{t_1}^{t_2} dt\,exp (-2 \lambda t)}{\left(\int_{t1}^{t2} dt\,exp (-\lambda t)\right)^2}
  \f]

  For more details, see:
  Stearns, C. W., McDaniel, D. L., Kohlmyer, S. G., Arul, P. R., Geiser, B. P., & Shanmugam, V. (2003).
  Random coincidence estimation from single event rates on the Discovery ST PET/CT scanner.
  2003 IEEE Nuclear Science Symposium. Conference Record (IEEE Cat. No.03CH37515), 5, 3067-3069.
  https://doi.org/10.1109/NSSMIC.2003.1352545

  \todo Dead-time is currently completely ignored.
*/
void randoms_from_singles(ProjData& proj_data,
                          const SinglesRates& singles,
                          float coincidence_time_window = -1.F,
                          float radionuclide_halflife = -1.F);

END_NAMESPACE_STIR
