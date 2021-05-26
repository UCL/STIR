/*!

  \file
  \ingroup ImageProcessor
  \brief Implementation of class stir::HUToMuImageProcessor

  \author Kris Thielemans
  \author Benjamin A. Thomas

*/
/*
    Copyright (C) 2019, 2020, UCL
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

#include "stir/HUToMuImageProcessor.h"
#include "stir/info.h"
#include "stir/round.h"
#include <nlohmann/json.hpp>

START_NAMESPACE_STIR

template <typename TargetT>
HUToMuImageProcessor<TargetT>::HUToMuImageProcessor() {
  this->set_defaults();
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::set_slope_filename(const std::string& arg) {
  this->filename = arg;
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::set_manufacturer_name(const std::string& arg) {
  this->manufacturer_name = arg;
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::set_kilovoltage_peak(const float arg) {
  this->kilovoltage_peak = arg;
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::set_target_photon_energy(const float arg) {
  this->target_photon_energy = arg;
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::set_defaults() {
  base_type::set_defaults();
  manufacturer_name = "GENERIC";
  kilovoltage_peak = 120.F;
  target_photon_energy = 511.F;
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::initialise_keymap() {
  base_type::initialise_keymap();
  this->parser.add_start_key(std::string(this->registered_name) + " Parameters");
  this->parser.add_stop_key(std::string("End ") + this->registered_name + " Parameters");
  this->parser.add_key("slope filename", &filename);
  this->parser.add_key("manufacturer_name", &manufacturer_name);
  this->parser.add_key("kilovoltage_peak", &kilovoltage_peak);
  this->parser.add_key("target_photon_energy", &target_photon_energy);
}

template <typename TargetT>
bool
HUToMuImageProcessor<TargetT>::post_processing() {
  return base_type::post_processing();
}

template <typename TargetT>
Succeeded
HUToMuImageProcessor<TargetT>::virtual_set_up(const TargetT& image) {
  this->get_record_from_json();
  return Succeeded::yes;
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::get_record_from_json() {
  if (this->filename.empty())
    error("HUToMu: no filename set for the slope info");
  if (this->manufacturer_name.empty())
    error("HUToMu: no manufacturer set for the slope info");

  // Read slope file
  std::ifstream slope_json_file_stream(this->filename);
  nlohmann::json slope_json;
  slope_json_file_stream >> slope_json;

  if (slope_json.find("scale") == slope_json.end()) {
    error("HUToMu: No or incorrect JSON slopes set (could not find \"scale\" in file \"" + filename + "\")");
  }
  // Put user-specified manufacturer into upper case.
  std::string manufacturer_upper_case = this->manufacturer_name;
  std::locale loc;
  for (std::string::size_type i = 0; i < manufacturer_name.length(); ++i)
    manufacturer_upper_case[i] = std::toupper(manufacturer_name[i], loc);

  // Get desired keV as integer value
  const int keV = stir::round(this->target_photon_energy);
  // Get desired kVp as integer value
  const int kVp = stir::round(this->kilovoltage_peak);
  stir::info(boost::format("HUToMu: finding record with manufacturer: '%s', keV=%d, kVp=%d in file '%s'") %
                 manufacturer_upper_case % keV % kVp % this->filename,
             2);

  // Extract appropriate chunk of JSON file for given manufacturer.
  nlohmann::json target = slope_json["scale"][manufacturer_upper_case]["transform"];

  int location = -1;
  int pos = 0;
  for (auto entry : target) {
    if ((stir::round(float(entry["kev"])) == keV) && (stir::round(float(entry["kvp"])) == kVp))
      location = pos;
    pos++;
  }

  if (location == -1) {
    stir::error("HUToMu: Desired slope not found!");
  }

  // Extract transform for specific keV and kVp.
  nlohmann::json transform = target[location];
  {
    std::stringstream str;
    str << transform.dump(4);
    info("HUToMu: JSON record found:" + str.str(), 2);
  }
  this->a1 = transform["a1"];
  this->b1 = transform["b1"];

  this->a2 = transform["a2"];
  this->b2 = transform["b2"];

  this->breakPoint = transform["break"];

  // std::cout << transform.dump(4);
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::apply_scaling_to_HU(TargetT& output_image, const TargetT& input_image) const {
  auto out_iter = output_image.begin_all();
  auto in_iter = input_image.begin_all();

  while (in_iter != input_image.end_all()) {
    if (*in_iter < breakPoint) {
      const float mu = a1 + b1 * (*in_iter);
      *out_iter = (mu < 0.0f) ? 0.0f : mu;
    } else {
      *out_iter = a2 + b2 * (*in_iter);
    }

    ++in_iter;
    ++out_iter;
  }
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::virtual_apply(TargetT& out_density, const TargetT& in_density) const {
  this->apply_scaling_to_HU(out_density, in_density);
}

template <typename TargetT>
void
HUToMuImageProcessor<TargetT>::virtual_apply(TargetT& density) const {
  shared_ptr<TargetT> copy_sptr(density.clone());
  this->apply_scaling_to_HU(density, *copy_sptr);
}

#ifdef _MSC_VER
// prevent warning message on reinstantiation,
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable : 4660)
#endif

// Register this class in the ImageProcessor registry
// static HUToMuImageProcessor<DiscretisedDensity<3, float>>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template class HUToMuImageProcessor<DiscretisedDensity<3, float>>;

END_NAMESPACE_STIR
