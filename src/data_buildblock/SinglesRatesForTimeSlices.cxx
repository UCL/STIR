/*
    Copyright (C) 2003-2007, Hammersmith Imanet Ltd
    Copyright (C) 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup singles_buildblock
  \brief Implementation of stir::SinglesRatesForTimeSlices

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Tim Borgeaud
*/

#include "stir/IndexRange.h"
#include "stir/IndexRange2D.h"
#include "stir/data/SinglesRatesForTimeSlices.h"
#include "stir/round.h"
#include "stir/error.h"
#include "stir/format.h"
#include <vector>
#include <utility>
#include <algorithm>

START_NAMESPACE_STIR

const double MAX_INTERVAL_DIFFERENCE = 0.05; // 5% max difference.

// Constructor
SinglesRatesForTimeSlices::SinglesRatesForTimeSlices()
{}

// Generate a FramesSinglesRate - containing the average rates
// for a frame begining at start_time and ending at end_time.
FrameSinglesRates
SinglesRatesForTimeSlices::get_rates_for_frame(double start_time, double end_time) const
{

  int num_singles_units = scanner_sptr->get_num_singles_units();

  // Create a temporary vector
  std::vector<float> average_singles_rates(num_singles_units);

  // Loop over all bins.
  for (int singles_bin = 0; singles_bin < num_singles_units; ++singles_bin)
    {
      average_singles_rates[singles_bin] = get_singles_rate(singles_bin, start_time, end_time);
    }

  // Determine that start and end slice indices.
  int start_slice = get_start_time_slice_index(start_time);
  int end_slice = get_end_time_slice_index(end_time);

  double frame_start_time = get_slice_start_time(start_slice);
  double frame_end_time = _times[end_slice];

  // Create temp FrameSinglesRate object
  FrameSinglesRates frame_rates(average_singles_rates, frame_start_time, frame_end_time, scanner_sptr);

  return (frame_rates);
}

// Get time slice index.
// Returns the index of the slice that contains the specified time.
int
SinglesRatesForTimeSlices::get_end_time_slice_index(double t) const
{

  int slice_index = 0;

  // Start with an initial estimate.
  if (_singles_time_interval != 0)
    {
      slice_index = static_cast<int>(floor(t / _singles_time_interval));
    }

  if (slice_index >= _num_time_slices)
    {
      slice_index = _num_time_slices - 1;
    }

  // Check estimate and determine whether to look further or backwards.
  // Note that we could just move fowards first and then backwards but this
  // method is more intuitive.

  if (_times[slice_index] < t)
    {

      // Check forwards.
      while (slice_index < _num_time_slices - 1 && _times[slice_index] < t)
        {
          slice_index++;
        }
    }
  else
    {

      // Check backwards.
      while (slice_index > 0 && _times[slice_index - 1] >= t)
        {
          slice_index--;
        }
    }

  return (slice_index);
}

// Get time slice index.
// Returns first slice ending _after_ t.
int
SinglesRatesForTimeSlices::get_start_time_slice_index(double t) const
{

  int slice_index = 0;

  // Start with an initial estimate.
  if (_singles_time_interval != 0)
    {
      slice_index = std::max(0, static_cast<int>(floor((t - _times[0]) / _singles_time_interval)));
    }

  if (slice_index >= _num_time_slices)
    {
      slice_index = _num_time_slices - 1;
    }

  // Check estimate and determine whether to look further or backwards.
  // Note that we could just move fowards first and then backwards but this
  // method is more intuitive.

  if (_times[slice_index] < t)
    {

      // Check forwards.
      while (slice_index < _num_time_slices - 1 && _times[slice_index] <= t)
        {
          slice_index++;
        }
    }
  else
    {

      // Check backwards.
      while (slice_index > 0 && _times[slice_index - 1] > t)
        {
          slice_index--;
        }
    }

  return (slice_index);
}

#if 0
// Get rates using time slice and singles bin indices.
int 
SinglesRatesForTimeSlices::
get_singles_rate(int singles_bin_index, int time_slice) const {
  
  // Check ranges.
  int total_singles_units = scanner_sptr->get_num_singles_units();
  
  if ( singles_bin_index < 0 || singles_bin_index >= total_singles_units ||
       time_slice < 0 || time_slice >= _num_time_slices ) {
    return(0);
  } else {
    return _singles[time_slice][singles_bin_index];
  }

}
#endif

// Set a singles by bin index and time slice.
void
SinglesRatesForTimeSlices::set_singles(int singles_bin_index, int time_slice, int new_rate)
{

  int total_singles_units = scanner_sptr->get_num_singles_units();

  if (singles_bin_index >= 0 && singles_bin_index < total_singles_units && time_slice >= 0 && time_slice < _num_time_slices)
    {
      _singles[time_slice][singles_bin_index] = new_rate;
    }
}

int
SinglesRatesForTimeSlices::rebin(std::vector<double>& new_end_times)
{

  const int num_new_slices = new_end_times.size();
  const int total_singles_units = scanner_sptr->get_num_singles_units();

  // Create the new array of singles data.
  Array<2, int> new_singles = Array<2, int>(IndexRange2D(0, num_new_slices - 1, 0, total_singles_units - 1));

  // Sort the set of new time slices.
  std::sort(new_end_times.begin(), new_end_times.end());

  double start_time = get_slice_start_time(0);

  // Loop over new time slices.
  for (unsigned int new_slice = 0; new_slice < new_end_times.size(); ++new_slice)
    {

      // End time for the new time slice.
      double end_time = new_end_times[new_slice];

      // If start time is beyond last end time in original data, then use zeros.
      if (start_time > _times[_num_time_slices - 1])
        {
          for (int singles_bin = 0; singles_bin < total_singles_units; ++singles_bin)
            {
              new_singles[new_slice][singles_bin] = 0;
            }
        }
      else
        {

          // Get the singles rate average between start and end times for all bins.
          for (int singles_bin = 0; singles_bin < total_singles_units; ++singles_bin)
            {
              new_singles[new_slice][singles_bin] = round(get_singles(singles_bin, start_time, end_time));
            }
        }

      // Next new time slice starts at the end of this slice.
      start_time = end_time;
    }

  // Set the singles and times using the new sets.
  _singles = new_singles;
  _times = new_end_times;
  _num_time_slices = _times.size();

  return (_num_time_slices);
}

std::vector<double>
SinglesRatesForTimeSlices::get_times() const
{
  return _times;
}

int
SinglesRatesForTimeSlices::get_num_time_slices() const
{
  return (_num_time_slices);
}

double
SinglesRatesForTimeSlices::get_singles_time_interval() const
{
  return (_singles_time_interval);
}

float
SinglesRatesForTimeSlices::get_singles(const int singles_bin_index, const double start_time, const double end_time) const
{

  // First Calculate an inclusive range. start_time_slice is the
  // the first slice with an ending time greater than start_time.
  // end_time_slice is the first time slice that ends at, or after,
  // end_time.
  int start_slice = this->get_start_time_slice_index(start_time);
  int end_slice = this->get_end_time_slice_index(end_time);

  // Total contribution from all slices.
  double total_singles;
  // Start and end times for starting and ending slices.
  double slice_start_time;
  double slice_end_time;

  // Calculate the fraction of the start_slice to include.
  slice_start_time = get_slice_start_time(start_slice);
  slice_end_time = _times[start_slice];

  if (start_time > end_time)
    error(format("get_singles() called with start_time {} larger than end time {}", start_time, end_time));
  if (start_time < get_slice_start_time(start_slice) - 1e-2 /* allow for some rounding */)
    error(format("get_singles() called with start time {} which is smaller than the start time in the data ({})",
                 start_time,
                 slice_start_time));
  if (end_time > _times[end_slice] + 1e-2 /* allow for some rounding */)
    error(format(
        "get_singles() called with end time {} which is larger than the end time in the data ({})", end_time, slice_end_time));

  double old_duration = slice_end_time - slice_start_time;
  double included_duration = slice_end_time - start_time;

  double fraction = included_duration / old_duration;

  // Set the total singles so far to be the fraction of the bin.
  total_singles = fraction * _singles[start_slice][singles_bin_index];
  if (start_slice < end_slice)
    {
      // Calculate the fraction of the end_slice to include.
      slice_start_time = get_slice_start_time(end_slice);
      slice_end_time = _times[end_slice];

      old_duration = slice_end_time - slice_start_time;
      included_duration = end_time - slice_start_time;

      fraction = included_duration / old_duration;

      // Add the fraction of the bin to the running total.
      total_singles += fraction * _singles[end_slice][singles_bin_index];

      // Add all intervening slices.
      for (int slice = start_slice + 1; slice < end_slice; ++slice)
        {
          total_singles += _singles[slice][singles_bin_index];
        }
    }

  return (static_cast<float>(total_singles));
}

/*
 *
 * Protected methods.
 *
 */

void
SinglesRatesForTimeSlices::set_time_interval()
{

  // Run through the _times vector and calculate an average difference
  // between the starts of consecutive time slices.

  // Min and max differences (slice durations).
  double min_diff = 0;
  double max_diff = 0;
  double total = 0;

  for (std::vector<double>::const_iterator t = _times.begin(); t < _times.end() - 1; ++t)
    {
      double diff = *(t + 1) - *t;
      total += diff;

      if (min_diff == 0 || diff < min_diff)
        {
          min_diff = diff;
        }

      if (diff > max_diff)
        {
          max_diff = diff;
        }
    }

  _singles_time_interval = total / (_times.size() - 1);

  if ((max_diff - min_diff) / (_singles_time_interval) > MAX_INTERVAL_DIFFERENCE)
    {
      // Slice durations are not consistent enough to be considered the same.
      _singles_time_interval = 0;
    }
}

// get slice start time.
double
SinglesRatesForTimeSlices::get_slice_start_time(int slice_index) const
{

  if (slice_index >= _num_time_slices)
    {
      slice_index = _num_time_slices - 1;
    }

  if (slice_index == 0)
    {
      return (_times[0] - _singles_time_interval);
    }
  else
    {
      return (_times[slice_index - 1]);
    }
}

TimeFrameDefinitions
SinglesRatesForTimeSlices::get_time_frame_definitions() const
{

  std::vector<std::pair<double, double>> start_ends(get_num_time_slices());
  for (int i = 0; i < get_num_time_slices(); ++i)
    {
      start_ends[i].first = get_slice_start_time(i);
      start_ends[i].second = _times[i];
    }
  return TimeFrameDefinitions(start_ends);
}

END_NAMESPACE_STIR
