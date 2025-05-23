//
//
/*!
  \file
  \ingroup Parallelproj

  \brief Defines stir::detail::ParallelprojHelper

  \author Kris Thielemans

*/
/*
    Copyright (C) 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_ParallelprojHelper_h__
#define __stir_recon_buildblock_ParallelprojHelper_h__

#include "stir/common.h"
#include <vector>
#include <array>

START_NAMESPACE_STIR

template <int num_dimensions, class elemT> class DiscretisedDensity;
class ProjDataInfo;

namespace detail
{
  /*!
    \ingroup projection
    \ingroup Parallelproj
    \brief Helper class for Parallelproj's projectors
  */
  class ParallelprojHelper
  { 
  public:

    ~ParallelprojHelper();
    ParallelprojHelper(const ProjDataInfo& p_info, const DiscretisedDensity<3,float> &density);

    // parallelproj arrays
    std::array<float,3> voxsize;
    std::array<int,3> imgdim;
    std::array<float,3> origin;
    std::vector<float> xstart;
    std::vector<float> xend;
  };

}

END_NAMESPACE_STIR


#endif // __stir_recon_buildblock_ParallelprojHelper_h__
