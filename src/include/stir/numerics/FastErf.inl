/*!
  \file
  \ingroup numerics
  \brief Implementation of an erf interpolation

  \author Robert Twyman
  \author Alexander Whitehead
*/
/*
    Copyright (C) 2022, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/

#include "stir/numerics/BSplines1DRegularGrid.h"
#include "stir/numerics/erf.h"

START_NAMESPACE_STIR

inline void
FastErf::
set_up()
{
  this->_sampling_period = (2 * this->_maximum_sample_value) / this->get_num_samples();
  
  // Add two samples for end cases and compute a vector of erf values
  std::vector<double> erf_values(this->get_num_samples() + 2);
  for (int i=0; i<this->get_num_samples() + 2; ++i)
    erf_values[i] = erf(i * this->_sampling_period - this->_maximum_sample_value);

  // Setup BSplines
  BSpline::BSplines1DRegularGrid<double, double> spline(erf_values, BSpline::linear);
  this->_spline = spline;

  erf_values_vec = erf_values;
//  this->_is_setup = true;
}

inline double
FastErf::get_erf_BSplines_interpolation(double xp) const
{
#if 0
    xp = std::clamp(xp,-this->_maximum_sample_value, this->_maximum_sample_value);
#else
  xp = std::max(std::min(this->_maximum_sample_value, xp), -this->_maximum_sample_value);
#endif
  return this->_spline.BSplines((xp + this->_maximum_sample_value) / this->_sampling_period);
}


inline double
FastErf::
get_erf_linear_interpolation(double xp) const
{
#if 0
    xp = std::clamp(xp,-this->_maximum_sample_value, this->_maximum_sample_value);
#else
  xp = std::max(std::min(this->_maximum_sample_value, xp), -this->_maximum_sample_value);
#endif
  // Find xp in index sequence
  double xp_in_index = ((xp + this->_maximum_sample_value) / this->_sampling_period);

  // Find lower integer in index space
  int lower = static_cast<int>(floor(xp_in_index));

  // Linear interpolation of xp between vec[lower] and vec[lower + 1]
  return erf_values_vec[lower] + (xp_in_index - lower) * (erf_values_vec[lower + 1] - erf_values_vec[lower]);
}

inline double
FastErf::get_erf_nearest_neighbour_interpolation(double xp) const
{
#if 0
  xp = std::clamp(xp,-this->_maximum_sample_value, this->_maximum_sample_value);
#else
  xp = std::max(std::min(this->_maximum_sample_value, xp), -this->_maximum_sample_value);
#endif
  // Selects index of the nearest neighbour via rounding
    return erf_values_vec[static_cast<int>(std::round((xp + this->_maximum_sample_value) / this->_sampling_period))];
}

inline void
FastErf::set_num_samples(const int num_samples)
{
  this->_num_samples = num_samples;
}

inline
int
FastErf::get_num_samples() const
{
  return this->_num_samples;
}

int
FastErf::get_maximum_sample_value() const
{
  return this->_maximum_sample_value;
}

void
FastErf::set_maximum_sample_value(double maximum_sample_value)
{
  this->_maximum_sample_value = maximum_sample_value;
}

const double
FastErf::operator() (const double xp) const
{
  return get_erf_linear_interpolation(xp);
}

END_NAMESPACE_STIR
