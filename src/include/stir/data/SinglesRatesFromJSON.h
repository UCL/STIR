/*
    Copyright (C) 2026, London South Bank University
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup singles_buildblock

  \brief Declaration of class stir::SinglesRatesFromJSON

  \author Alan Osys
*/

#ifndef __stir_data_SinglesRatesFromJSON_H__
#define __stir_data_SinglesRatesFromJSON_H__

#include "stir/data/SinglesRatesForTimeFrames.h"
#include "stir/RegisteredParsingObject.h"
#include <string>
#include <vector>

START_NAMESPACE_STIR

/*!
  \ingroup singles_buildblock
  \brief A class that gets the singles rates of the singles units ("buckets") from a JSON file.

  The file needs the entries "interval_start" and "interval_end" (the times in seconds of each singles interval),
  "bucket_rates" (for each interval, an array with the singles rate of each singles unit), "num_axial_buckets",
  "num_transaxial_buckets", "num_rings" and "num_detectors_per_ring". Bucket \c i is at
  transaxial + num_transaxial_buckets * axial. This is what lm_to_projdata writes for list-mode data with singles records,
  for instance
  \code
  { "num_buckets": 224, "num_axial_buckets": 8, "num_transaxial_buckets": 28, "num_rings": 64,
    "num_detectors_per_ring": 504, "interval_start": [0, 1.0], "interval_end": [1.0, 2.0],
    "bucket_rates": [ [115378.4, 115102.2, ... ], [115280.1, 115015.7, ... ] ] }
  \endcode
  get_singles_rate() averages the rates of the intervals that overlap the requested times, weighted by the overlap.
  The singles of a single interval are given by get_singles() with the start and end time of that interval.

  In the parameter file, use
    \code
  singles rates := SinglesRatesFromJSON
  SinglesRatesFromJSON :=
    filename := head2_f1_singles.json
  End SinglesRatesFromJSON :=
  \endcode
*/
class SinglesRatesFromJSON : public RegisteredParsingObject<SinglesRatesFromJSON, SinglesRates, SinglesRatesForTimeFrames>
{
public:
  //! Name which will be used when parsing a SinglesRatesFromJSON object
  static constexpr const char* const registered_name = "SinglesRatesFromJSON";

  SinglesRatesFromJSON() { set_defaults(); }

  //! Reads the rates and the layout of the singles units from a JSON file
  /*! \return false if the file cannot be read, an entry is missing, or the layout does not match the number of rates.
      Any rates read before are replaced.
   */
  bool read_from_file(const std::string& filename);

  //! Singles rate of the singles unit that contains the detector
  float get_singles_rate(const DetectionPosition<>& det_pos, const double start_time, const double end_time) const override;

private:
private:
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

  std::string _filename;
  int _num_axial_buckets = 0;
  int _num_transaxial_buckets = 0;
  int _num_rings = 0;
  int _num_detectors_per_ring = 0;
};

END_NAMESPACE_STIR

#endif