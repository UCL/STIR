//
//
/*
    Copyright (C) 2017, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief Implementations of inline functions for TOF time to mm

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/common.h"
START_NAMESPACE_STIR

inline double
mm_to_tof_delta_time(const float dist)
{
  return dist / speed_of_light_in_mm_per_ps_div2;
}

inline float
tof_delta_time_to_mm(const double delta_time)
{
  return static_cast<float>(delta_time * speed_of_light_in_mm_per_ps_div2);
}

END_NAMESPACE_STIR
