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

#include <fstream>
#ifdef HAVE_JSON
#  include <nlohmann/json.hpp>
#else
#  include <iterator>
#  include <sstream>
#endif
#ifndef HAVE_JSON
static bool
find_int(const std::string& content, const std::string& key, int& value)
{
  std::size_t pos = content.find("\"" + key + "\":");
  if (pos == std::string::npos)
    return false;
  try
    {
      value = std::stoi(content.substr(pos + key.size() + 3));
    }
  catch (const std::exception&)
    {
      return false;
    }
  return true;
}
#endif

START_NAMESPACE_STIR
bool
SinglesRatesFromJSON::read_from_file(const std::string& filename)
{
  std::ifstream f(filename);
  if (!f)
    return false;

  _rates.clear();
#ifdef HAVE_JSON
  try
    {
      const nlohmann::json j = nlohmann::json::parse(f);
      _rates = j.at("bucket_rates").get<std::vector<float>>();
      _num_axial_buckets = j.at("num_axial_buckets").get<int>();
      _num_transaxial_buckets = j.at("num_transaxial_buckets").get<int>();
      _num_rings = j.at("num_rings").get<int>();
      _num_detectors_per_ring = j.at("num_detectors_per_ring").get<int>();
    }
  catch (const nlohmann::json::exception&)
    {
      // not valid JSON, no "bucket_rates" entry, or entries that are not numbers
      _rates.clear();
      return false;
    }
#else
  // without the nlohmann JSON library: find the array of "bucket_rates" in the text
  const std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (!find_int(content, "num_axial_buckets", _num_axial_buckets)
      || !find_int(content, "num_transaxial_buckets", _num_transaxial_buckets) || !find_int(content, "num_rings", _num_rings)
      || !find_int(content, "num_detectors_per_ring", _num_detectors_per_ring))
    return false;
  const std::string key = "\"bucket_rates\": [";
  std::size_t pos = content.find(key);
  if (pos == std::string::npos)
    return false;
  pos += key.size();
  const std::size_t end = content.find("]", pos);
  if (end == std::string::npos)
    return false;

  std::istringstream ss(content.substr(pos, end - pos));
  std::string token;
  try
    {
      while (std::getline(ss, token, ','))
        {
          token.erase(0, token.find_first_not_of(" \t\n\r"));
          token.erase(token.find_last_not_of(" \t\n\r") + 1);
          if (!token.empty())
            _rates.push_back(std::stof(token));
        }
    }
  catch (const std::exception&)
    {
      _rates.clear();
      return false;
    }
#endif
  // the layout must be consistent with the number of rates
  if (_num_axial_buckets <= 0 || _num_transaxial_buckets <= 0 || _num_rings <= 0 || _num_detectors_per_ring <= 0
      || static_cast<int>(_rates.size()) != _num_axial_buckets * _num_transaxial_buckets)
    {
      _rates.clear();
      return false;
    }
  return true;
}

float
SinglesRatesFromJSON::get_singles_rate(const DetectionPosition<>& det_pos,
                                       const double /*start_time*/,
                                       const double /*end_time*/) const
{
  const int ring = static_cast<int>(det_pos.axial_coord());
  const int tangential = static_cast<int>(det_pos.tangential_coord());

  // no longer hardcoded
  const int rings_per_axial_bucket = _num_rings / _num_axial_buckets;
  const int detectors_per_trans_bucket = _num_detectors_per_ring / _num_transaxial_buckets;

  const int axial_bucket = ring / rings_per_axial_bucket;
  const int transaxial_bucket = tangential / detectors_per_trans_bucket;
  const int bucket_idx = transaxial_bucket + _num_transaxial_buckets * axial_bucket;

  if (bucket_idx < 0 || bucket_idx >= static_cast<int>(_rates.size()))
    return 0.f;
  return _rates[bucket_idx];
}

float
SinglesRatesFromJSON::get_singles(const int singles_bin_index, const double /*start_time*/, const double /*end_time*/) const
{
  if (singles_bin_index < 0 || singles_bin_index >= static_cast<int>(_rates.size()))
    return 0.f;
  return _rates[singles_bin_index];
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