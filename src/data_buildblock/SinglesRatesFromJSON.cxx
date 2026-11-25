/*
    Copyright (C) 2026, London South Bank University
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup singles_buildblock

  \brief Implementation of class stir::SinglesRatesFromJSON

  \author Alan Osys
*/

#include "stir/data/SinglesRatesFromJSON.h"
#include "stir/DetectionPosition.h"
#include "stir/error.h"

#include "stir/IndexRange2D.h"

#include <algorithm>
#include <fstream>
#ifdef HAVE_JSON
#  include <nlohmann/json.hpp>
#endif

START_NAMESPACE_STIR
bool
SinglesRatesFromJSON::read_from_file(const std::string& filename)
{
#ifndef HAVE_JSON
  error("SinglesRatesFromJSON: STIR needs to be compiled with the nlohmann JSON library to read '" + filename + "'");
  return false;
#else
  std::ifstream f(filename);
  if (!f)
    return false;

  try
    {
      const nlohmann::json j = nlohmann::json::parse(f);
      const auto interval_rates = j.at("bucket_rates").get<std::vector<std::vector<float>>>();
      const auto start_times = j.at("interval_start").get<std::vector<double>>();
      const auto end_times = j.at("interval_end").get<std::vector<double>>();
      _num_axial_buckets = j.at("num_axial_buckets").get<int>();
      _num_transaxial_buckets = j.at("num_transaxial_buckets").get<int>();
      _num_rings = j.at("num_rings").get<int>();
      _num_detectors_per_ring = j.at("num_detectors_per_ring").get<int>();

      // the layout must be consistent with the number of rates
      const int num_intervals = static_cast<int>(interval_rates.size());
      const int num_buckets = _num_axial_buckets * _num_transaxial_buckets;
      if (num_intervals == 0 || static_cast<int>(start_times.size()) != num_intervals
          || static_cast<int>(end_times.size()) != num_intervals || num_buckets <= 0 || _num_rings <= 0
          || _num_detectors_per_ring <= 0)
        return false;

      std::vector<std::pair<double, double>> intervals;
      _singles = Array<2, float>(IndexRange2D(1, num_intervals, 0, num_buckets - 1));
      for (int k = 0; k < num_intervals; ++k)
        {
          if (static_cast<int>(interval_rates[k].size()) != num_buckets)
            return false;
          intervals.emplace_back(start_times[k], end_times[k]);
          // SinglesRates works with the counts in an interval, i.e. rate * duration
          for (int b = 0; b < num_buckets; ++b)
            _singles[k + 1][b] = interval_rates[k][b] * static_cast<float>(end_times[k] - start_times[k]);
        }
      _time_frame_defs = TimeFrameDefinitions(intervals);
    }
  catch (const nlohmann::json::exception&)
    {
      // not valid JSON, an entry is missing, or entries that are not numbers
      return false;
    }
  return true;
#endif
}

float
SinglesRatesFromJSON::get_singles_rate(const DetectionPosition<>& det_pos, const double start_time, const double end_time) const
{
  const int ring = static_cast<int>(det_pos.axial_coord());
  const int tangential = static_cast<int>(det_pos.tangential_coord());

  // no longer hardcoded
  const int rings_per_axial_bucket = _num_rings / _num_axial_buckets;
  const int detectors_per_trans_bucket = _num_detectors_per_ring / _num_transaxial_buckets;

  const int axial_bucket = ring / rings_per_axial_bucket;
  const int transaxial_bucket = tangential / detectors_per_trans_bucket;
  const int bucket_idx = transaxial_bucket + _num_transaxial_buckets * axial_bucket;

  if (bucket_idx < 0 || bucket_idx >= _num_axial_buckets * _num_transaxial_buckets)
    return 0.f;

  // average of the rates of the overlapping intervals, weighted by the overlap
  double sum = 0;
  double total_time = 0;
  for (unsigned int f = 1; f <= get_num_frames(); ++f)
    {
      const double interval_start = _time_frame_defs.get_start_time(f);
      const double interval_end = _time_frame_defs.get_end_time(f);
      const double overlap = std::min(end_time, interval_end) - std::max(start_time, interval_start);
      if (overlap > 0)
        {
          sum += get_singles(bucket_idx, f) / (interval_end - interval_start) * overlap;
          total_time += overlap;
        }
    }
  return total_time > 0 ? static_cast<float>(sum / total_time) : 0.f;
}
void
SinglesRatesFromJSON::initialise_keymap()
{
  parser.add_start_key("SinglesRatesFromJSON");
  parser.add_key("filename", &_filename);
  parser.add_stop_key("End SinglesRatesFromJSON");
}

bool
SinglesRatesFromJSON::post_processing()
{
  if (!read_from_file(_filename))
    error("SinglesRatesFromJSON: could not read valid singles rates from file '" + _filename + "'");
  return false;
}

void
SinglesRatesFromJSON::set_defaults()
{
  _filename = "";
}

END_NAMESPACE_STIR