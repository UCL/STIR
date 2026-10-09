#ifndef __stir_data_SinglesRatesFromJSON_H__
#define __stir_data_SinglesRatesFromJSON_H__

#include "stir/data/SinglesRates.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

START_NAMESPACE_STIR

class SinglesRatesFromJSON : public SinglesRates
{
public:
  SinglesRatesFromJSON() {}

  std::string get_registered_name() const override { return "SinglesRatesFromJSON"; }

  bool read_from_file(const std::string& filename)
  {
    std::ifstream f(filename);
    if (!f)
      return false;

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // parse bucket_rates array
    const std::string key = "\"bucket_rates\": [";
    std::size_t pos = content.find(key);
    if (pos == std::string::npos)
      return false;
    pos += key.size();

    std::size_t end = content.find("]", pos);
    if (end == std::string::npos)
      return false;

    std::string array_str = content.substr(pos, end - pos);
    std::istringstream ss(array_str);
    std::string token;
    while (std::getline(ss, token, ','))
      {
        // trim whitespace
        token.erase(0, token.find_first_not_of(" \t\n\r"));
        token.erase(token.find_last_not_of(" \t\n\r") + 1);
        if (!token.empty())
          _rates.push_back(std::stof(token));
      }
    return !_rates.empty();
  }

float get_singles_rate(const DetectionPosition<>& det_pos,
                       const double /*start_time*/,
                       const double /*end_time*/) const override
{
  const int ring       = static_cast<int>(det_pos.axial_coord());
  const int tangential = static_cast<int>(det_pos.tangential_coord());

  const int num_axial_buckets      = 8;
  const int num_transaxial_buckets = 28;
  const int rings_per_axial_bucket = 64 / num_axial_buckets;

  const int axial_bucket      = std::min(ring / rings_per_axial_bucket, num_axial_buckets - 1);
  const int transaxial_bucket = std::min((tangential * num_transaxial_buckets) / 504,
                                          num_transaxial_buckets - 1);
  const int bucket_idx        = transaxial_bucket + num_transaxial_buckets * axial_bucket;

  if (bucket_idx < 0 || bucket_idx >= static_cast<int>(_rates.size()))
      return 0.f;
  return _rates[bucket_idx];
}

  float get_singles(const int singles_bin_index, const double start_time, const double end_time) const override
  {
    if (singles_bin_index < 0 || singles_bin_index >= static_cast<int>(_rates.size()))
      return 0.f;
    return _rates[singles_bin_index];
  }

private:
  std::vector<float> _rates;
};

END_NAMESPACE_STIR

#endif
