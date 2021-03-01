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
