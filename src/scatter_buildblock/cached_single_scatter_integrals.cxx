/*
  Copyright (C) 2004-2009, Hammersmith Imanet Ltd
  Copyright (C) 2013 University College London
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
/*!
  \file
  \ingroup scatter
  \brief Implementations of functions defined in stir::ScatterEstimationByBin

  Functions calculate the integral along LOR in an image (attenuation or emission).
  (from scatter point to detector coordinate).

  \author Charalampos Tsoumpas
  \author Nikolaos Dikaios
  \author Kris Thielemans
*/
#include "stir/scatter/ScatterSimulation.h"
#include "stir/IndexRange.h"
#include "stir/Coordinate2D.h"

START_NAMESPACE_STIR

const float cache_init_value = -1234567.89E10F; // an arbitrary value that should never occur

void
ScatterSimulation::
remove_cache_for_integrals_over_attenuation()
{
  this->cached_attenuation_integral_scattpoint_det.recycle();
}

void
ScatterSimulation::
remove_cache_for_integrals_over_activity()
{
  this->cached_activity_integral_scattpoint_det.recycle();
}


void
ScatterSimulation::
initialise_cache_for_scattpoint_det_integrals_over_attenuation()
{
  if (!this->use_cache)
    return;

  const IndexRange<2> range (Coordinate2D<int> (0,0),
                             Coordinate2D<int> (static_cast<int>(this->scatt_points_vector.size()-1),
                                                this->total_detectors-1));
  if (this->cached_attenuation_integral_scattpoint_det.get_index_range() == range)
    return;  // keep cache if correct size

  this->cached_attenuation_integral_scattpoint_det.resize(range);
  this->cached_attenuation_integral_scattpoint_det.fill(cache_init_value);
}

void
ScatterSimulation::
initialise_cache_for_scattpoint_det_integrals_over_activity()
{
  if (!this->use_cache)
    return;

  const IndexRange<2> range (Coordinate2D<int> (0,0),
                             Coordinate2D<int> (static_cast<int>(this->scatt_points_vector.size()-1),
                                                this->total_detectors-1));

  if (this->cached_activity_integral_scattpoint_det.get_index_range() == range)
    return; // keep cache if correct size

  this->cached_activity_integral_scattpoint_det.resize(range);
  this->cached_activity_integral_scattpoint_det.fill(cache_init_value);
}

float
ScatterSimulation::
cached_integral_over_activity_image_between_scattpoint_det(const unsigned scatter_point_num,
                                                           const unsigned det_num)
{
  float * location_in_cache =
    this->use_cache
    ? &cached_activity_integral_scattpoint_det[scatter_point_num][det_num]
    : 0;

  /* OPENMP note:
     We use atomic read/write to get at the cache. This should ensure validity.
     Probably we could have 2 threads computing the same value that will be
     cached later, but this might be better than locking (and it's simpler to write).
     Sadly, this is only supported from OpenMP 3.1, so we need to add some extra checks.
  */
  float value;
#if defined(STIR_OPENMP)
# if _OPENMP >=201012
#  pragma omp atomic read
# else
#  pragma omp critical(STIRSCATTERESTIMATIONREADCACHE)
  {
# endif
#endif
  value = *location_in_cache;
#if defined(STIR_OPENMP) && (_OPENMP <201012)
  }
#endif

  if (this->use_cache && value!=cache_init_value)
    {
      return value;
    }
  else
    {
      const float result =
        integral_over_activity_image_between_scattpoint_det
        (scatt_points_vector[scatter_point_num].coord,
         detection_points_vector[det_num]
         );
      if (this->use_cache)
#ifdef STIR_OPENMP
# if _OPENMP >=201012
#  pragma omp atomic write
# else
#  pragma omp critical(STIRSCATTERESTIMATIONWRITECACHE)
        {
# endif
#endif
        *location_in_cache=result;
#if defined(STIR_OPENMP) && (_OPENMP <201012)
        }
#endif
      return result;
    }
}

float
ScatterSimulation::
cached_exp_integral_over_attenuation_image_between_scattpoint_det(const unsigned scatter_point_num,
                                                                  const unsigned det_num)
{
  float * location_in_cache =
    this->use_cache
    ? &cached_attenuation_integral_scattpoint_det[scatter_point_num][det_num]
    : 0;

  float value;
#if defined(STIR_OPENMP)
# if _OPENMP >=201012
#  pragma omp atomic read
# else
#  pragma omp critical(STIRSCATTERESTIMATIONREADCACHEATTENINT)
  {
# endif
#endif
  value = *location_in_cache;
#if defined(STIR_OPENMP) && (_OPENMP <201012)
  }
#endif

  if (this->use_cache && value!=cache_init_value)
    {
      return *location_in_cache;
    }
  else
    {
      const float result =
        exp_integral_over_attenuation_image_between_scattpoint_det
        (scatt_points_vector[scatter_point_num].coord,
         detection_points_vector[det_num]
         );
      if (this->use_cache)
#ifdef STIR_OPENMP
# if _OPENMP >=201012
#  pragma omp atomic write
# else
#  pragma omp critical(STIRSCATTERESTIMATIONREADCACHEATTENINT)
        {
# endif
#endif
        *location_in_cache=result;
#if defined(STIR_OPENMP) && (_OPENMP <201012)
        }
#endif
      return result;
    }
}

END_NAMESPACE_STIR
