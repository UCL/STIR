//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup singles_buildblock
  \brief Implementation of stir::FrameSinglesRates and stir::SinglesRates

  \author Kris Thielemans
  \author Tim Borgeaud
*/

#include "stir/data/SinglesRates.h"
using std::vector;

START_NAMESPACE_STIR

/*
 *! FrameSinglesRates constructor.
 */
FrameSinglesRates::FrameSinglesRates(vector<float>& avg_singles_rates,
                                     double start_time,
                                     double end_time,
                                     shared_ptr<Scanner> scanner_sptr)
    : _start_time(start_time),
      _end_time(end_time),
      _singles(avg_singles_rates),
      _scanner_sptr(scanner_sptr)
{
  assert(avg_singles_rates.size() == static_cast<std::size_t>(scanner_sptr->get_num_singles_units()));
}

float
FrameSinglesRates::get_singles_rate(int singles_bin_index) const
{
  return (_singles[singles_bin_index]);
}

float
FrameSinglesRates::get_singles_rate(const DetectionPosition<>& det_pos) const
{
  int singles_bin_index = _scanner_sptr->get_singles_bin_index(det_pos);
  return (get_singles_rate(singles_bin_index));
}

double
FrameSinglesRates::get_start_time() const
{
  return (_start_time);
}

double
FrameSinglesRates::get_end_time() const
{
  return (_end_time);
}

// Get the average singles rate for a particular bin.
float
SinglesRates::get_singles_rate(const int singles_bin_index, const double start_time, const double end_time) const
{
  return static_cast<float>(get_singles(singles_bin_index, start_time, end_time) / (end_time - start_time));
}

float
SinglesRates::get_singles_rate(const DetectionPosition<>& det_pos, const double start_time, const double end_time) const
{
  const int singles_bin_index = scanner_sptr->get_singles_bin_index(det_pos);
  return (get_singles_rate(singles_bin_index, start_time, end_time));
}

float
SinglesRates::get_singles(const DetectionPosition<>& det_pos, const double start_time, const double end_time) const
{
  const int singles_bin_index = scanner_sptr->get_singles_bin_index(det_pos);
  return (get_singles(singles_bin_index, start_time, end_time));
}

END_NAMESPACE_STIR
