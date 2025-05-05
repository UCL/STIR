/*
    Copyright (C) 2015, 2016 University of Leeds
    Copyright (C) 2017, 2018 University of Hull
    Copyright (C) 2016, 2017, 2020, 2023, 2024 University College London
    Copyright (C) 2018 University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Implementation of class stir::CListModeDataROOT

  \author Nikos Efthimiou
  \author Harry Tsoumpas
  \author Kris Thielemans
  \author Robbie Twyman
*/

#include "stir/listmode/CListModeDataROOT.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/FilePath.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/format.h"

START_NAMESPACE_STIR

constexpr static char max_num_timing_bins_keyword[] = "Maximum number of (unmashed) TOF time bins";
constexpr static char size_timing_bin_keyword[] = "Size of unmashed TOF time bins (ps)";
constexpr static char timing_resolution_keyword[] = "TOF timing resolution (ps)";

CListModeDataROOT::CListModeDataROOT(const std::string& hroot_filename)
    : hroot_filename(hroot_filename)
{
  set_defaults();
  std::string error_str;
  int num_virtual_axial_crystals_per_block = -1;      // -1 means: use scanner default
  int num_virtual_transaxial_crystals_per_block = -1; // -1 means: use scanner default

  this->parser.add_start_key("ROOT header");
  this->parser.add_stop_key("End ROOT header");

  // Scanner related & Physical dimensions.
  this->parser.add_key("originating system", &this->originating_system);

  this->parser.add_key("Number of rings", &this->num_rings);
  this->parser.add_key("Number of detectors per ring", &this->num_detectors_per_ring);
  this->parser.add_key("Inner ring diameter (cm)", &this->inner_ring_diameter);
  this->parser.add_key("Average depth of interaction (cm)", &this->average_depth_of_interaction);
  this->parser.add_key("Distance between rings (cm)", &this->ring_spacing);
  this->parser.add_key("Default bin size (cm)", &this->bin_size);
  this->parser.add_key("View offset (degrees)", &this->view_offset);
  this->parser.add_key("Maximum number of non-arc-corrected bins", &this->max_num_non_arccorrected_bins);
  this->parser.add_key("Default number of arc-corrected bins", &this->default_num_arccorrected_bins);
  this->parser.add_key("Number of virtual axial crystals per block", &num_virtual_axial_crystals_per_block);
  this->parser.add_key("Number of virtual transaxial crystals per block", &num_virtual_transaxial_crystals_per_block);
  // end Scanner and physical dimensions.

  this->parser.add_key("energy resolution", &this->energy_resolution);
  this->parser.add_key("reference energy", &this->reference_energy);

  this->parser.add_key(max_num_timing_bins_keyword, &max_num_timing_bins);
#if STIR_VERSION < 070000
  this->parser.add_alias_key(max_num_timing_bins_keyword, "Number of TOF time bins");
#endif
  this->parser.add_key(size_timing_bin_keyword, &size_timing_bin);
#if STIR_VERSION < 070000
  this->parser.add_alias_key(size_timing_bin_keyword, "Size of timing bin (ps)");
#endif
  this->parser.add_key(timing_resolution_keyword, &this->timing_resolution);
#if STIR_VERSION < 070000
  this->parser.add_alias_key(timing_resolution_keyword, "timing resolution (ps)");
#endif
  this->parser.add_key("TOF mashing factor", &this->tof_mash_factor);
#if STIR_VERSION < 070000
  this->parser.add_alias_key("TOF mashing factor", "%TOF mashing factor");
#endif
  //

  // ROOT related
  this->parser.add_parsing_key("GATE scanner type", &this->root_file_sptr);
  if (!this->parser.parse(hroot_filename.c_str()))
    error("CListModeDataROOT: error parsing '%s'", hroot_filename.c_str());

  FilePath f(hroot_filename);
  if (root_file_sptr->set_up(f.get_path_only()) == Succeeded::no)
    error("CListModeDataROOT: Unable to set_up() from the input Header file (.hroot).");

  // ExamInfo initialisation
  shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);

  // Only PET scanners supported
  _exam_info_sptr->imaging_modality = ImagingModality::PT;
  _exam_info_sptr->originating_system = this->originating_system;
  _exam_info_sptr->set_low_energy_thres(this->root_file_sptr->get_low_energy_thres());
  _exam_info_sptr->set_high_energy_thres(this->root_file_sptr->get_up_energy_thres());

  this->exam_info_sptr = _exam_info_sptr;

  shared_ptr<Scanner> this_scanner_sptr;

  // If the user set Scanner::User_defined_scanner then the local geometry valiables must be set.
  bool give_it_a_try = false;
  if (this->originating_system != "User_defined_scanner") //
    {
      this_scanner_sptr.reset(Scanner::get_scanner_from_name(this->originating_system));
      if (this_scanner_sptr->get_type() == Scanner::Unknown_scanner)
        {
          warning(format("CListModeDataROOT: Unknown value for originating_system keyword: '{}'.\n WIll try to "
                         "figure out the scanner's geometry from the parameters",
                         originating_system));
          give_it_a_try = true;
        }
      else
        warning("CListModeDataROOT: I've set the scanner from STIR settings and ignored values in the hroot header.");
    }
  // If the user provide a Scanner name then, the local variables will be ignored and the Scanner
  // will be the selected.
  else if (this->originating_system == "User_defined_scanner" || give_it_a_try)
    {
      warning("CListModeDataROOT: Trying to figure out the scanner geometry from the information "
              "given in the ROOT header file.");

      if (check_scanner_definition(error_str) == Succeeded::no)
        {
          error(error_str.c_str());
        }
      if (default_num_arccorrected_bins == -1)
        {
          default_num_arccorrected_bins = max_num_non_arccorrected_bins;
        }

      this_scanner_sptr.reset(new Scanner(Scanner::User_defined_scanner,
                                          std::string("ROOT_defined_scanner"),
                                          /* num dets per ring */
                                          this->num_detectors_per_ring,
                                          /* num of rings */
                                          this->num_rings,
                                          /* number of non arccor bins */
                                          this->max_num_non_arccorrected_bins,
                                          /* number of maximum arccor bins */
                                          this->default_num_arccorrected_bins,
                                          /* inner ring radius */
                                          this->inner_ring_diameter / 0.2f,
                                          /* doi */ this->average_depth_of_interaction * 10.f,
                                          /* ring spacing */
                                          this->ring_spacing * 10.f,
                                          this->bin_size * 10.f,
                                          /* offset*/
                                          this->view_offset * _PI / 180,
                                          /*num_axial_blocks_per_bucket_v */
                                          this->root_file_sptr->get_num_axial_blocks_per_bucket_v(),
                                          /*num_transaxial_blocks_per_bucket_v*/
                                          this->root_file_sptr->get_num_transaxial_blocks_per_bucket_v(),
                                          /*num_axial_crystals_per_block_v*/
                                          this->root_file_sptr->get_num_axial_crystals_per_block_v(),
                                          /*num_transaxial_crystals_per_block_v*/
                                          this->root_file_sptr->get_num_transaxial_crystals_per_block_v(),
                                          /*num_axial_crystals_per_singles_unit_v*/
                                          this->root_file_sptr->get_num_axial_crystals_per_singles_unit(),
                                          /*num_transaxial_crystals_per_singles_unit_v*/
                                          this->root_file_sptr->get_num_trans_crystals_per_singles_unit(),
                                          /*num_detector_layers_v*/ 1,
                                          this->energy_resolution,
                                          this->reference_energy,
                                          /* maximum number of timing bins */
                                          max_num_timing_bins,
                                          /* size of basic TOF bin */
                                          size_timing_bin,
                                          /* Scanner's timing resolution */
                                          timing_resolution));
    }
  // have to do this here currently as these variables cannot be set via the constructor
  if (num_virtual_axial_crystals_per_block >= 0)
    this_scanner_sptr->set_num_virtual_axial_crystals_per_block(num_virtual_axial_crystals_per_block);
  if (num_virtual_transaxial_crystals_per_block >= 0)
    this_scanner_sptr->set_num_virtual_transaxial_crystals_per_block(num_virtual_transaxial_crystals_per_block);
  // put virtual block info in root_file_sptr
  this->root_file_sptr->set_num_virtual_axial_crystals_per_block(this_scanner_sptr->get_num_virtual_axial_crystals_per_block());
  this->root_file_sptr->set_num_virtual_transaxial_crystals_per_block(
      this_scanner_sptr->get_num_virtual_transaxial_crystals_per_block());

  // Compare with InputStreamFromROOTFile scanner generated geometry and throw error if wrong.
  if (check_scanner_match_geometry(error_str, this_scanner_sptr) == Succeeded::no)
    {
      error(error_str.c_str());
    }

  proj_data_info_sptr = std::const_pointer_cast<const ProjDataInfo>(
      ProjDataInfo::construct_proj_data_info(this_scanner_sptr,
                                             1,
                                             this_scanner_sptr->get_num_rings() - 1,
                                             this_scanner_sptr->get_num_detectors_per_ring() / 2,
                                             this_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                             /* arc_correction*/ false,
                                             tof_mash_factor)
          ->create_shared_clone());
  // this->set_proj_data_info_sptr(tmp);

  if (this->open_lm_file() == Succeeded::no)
    error("CListModeDataROOT: error opening ROOT file for filename '%s'", hroot_filename.c_str());
}

std::string
CListModeDataROOT::get_name() const
{
  return hroot_filename;
}

shared_ptr<CListRecord>
CListModeDataROOT::get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordROOT(this->get_proj_data_info_sptr()));
  return sptr;
}

Succeeded
CListModeDataROOT::open_lm_file()
{
  info(format("CListModeDataROOT: used ROOT file {}", this->root_file_sptr->get_ROOT_filename()));
  return Succeeded::yes;
}

Succeeded
CListModeDataROOT::get_next_record(CListRecord& record_of_general_type) const
{
  CListRecordROOT& record = dynamic_cast<CListRecordROOT&>(record_of_general_type);
  return root_file_sptr->get_next_record(record);
}

Succeeded
CListModeDataROOT::reset()
{
  return root_file_sptr->reset();
}

unsigned long
CListModeDataROOT::get_total_number_of_events() const
{
  return root_file_sptr->get_total_number_of_events();
}

CListModeData::SavedPosition
CListModeDataROOT::save_get_position()
{
  return static_cast<SavedPosition>(root_file_sptr->save_get_position());
}

Succeeded
CListModeDataROOT::set_get_position(const CListModeDataROOT::SavedPosition& pos)
{
  return root_file_sptr->set_get_position(pos);
}

void
CListModeDataROOT::set_defaults()
{
  num_rings = -1;
  num_detectors_per_ring = -1;
  max_num_non_arccorrected_bins = -1;
  default_num_arccorrected_bins = -1;
  inner_ring_diameter = -1.f;
  average_depth_of_interaction = -1.f;
  ring_spacing = -.1f;
  bin_size = -1.f;
  view_offset = 0.f;
  max_num_timing_bins = -1;
  size_timing_bin = -1.F;
  tof_mash_factor = 1;
  reference_energy = 511.F;
  energy_resolution = -1.F;
}

Succeeded
CListModeDataROOT::check_scanner_match_geometry(std::string& ret, const shared_ptr<Scanner>& scanner_sptr)
{
  std::ostringstream stream("CListModeDataROOT: The Scanner does not match the GATE geometry. Check: ");
  bool ok = true;

  if (scanner_sptr->get_num_rings() != root_file_sptr->get_num_rings())
    {
      stream << "the number of rings, ";
      ok = false;
    }

  if (scanner_sptr->get_num_detectors_per_ring() != root_file_sptr->get_num_dets_per_ring())
    {
      stream << "the number of detector per ring, ";
      ok = false;
    }

  if (scanner_sptr->get_num_axial_blocks_per_bucket() != root_file_sptr->get_num_axial_blocks_per_bucket_v())
    {
      stream << "the number of axial blocks per bucket, ";
      ok = false;
    }

  if (scanner_sptr->get_num_transaxial_blocks_per_bucket() != root_file_sptr->get_num_transaxial_blocks_per_bucket_v())
    {
      stream << "the number of transaxial blocks per bucket, ";
      ok = false;
    }

  if (scanner_sptr->get_num_axial_crystals_per_block() != root_file_sptr->get_num_axial_crystals_per_block_v())
    {
      stream << "the number of axial crystals per block, ";
      ok = false;
    }

  if (scanner_sptr->get_num_transaxial_crystals_per_block() != root_file_sptr->get_num_transaxial_crystals_per_block_v())
    {
      stream << "the number of transaxial crystals per block, ";
      ok = false;
    }

  if (scanner_sptr->get_num_axial_crystals_per_singles_unit() != root_file_sptr->get_num_axial_crystals_per_singles_unit())
    {
      stream << "the number of axial crystals per singles unit, ";
      ok = false;
    }

  if (scanner_sptr->get_num_transaxial_crystals_per_singles_unit() != root_file_sptr->get_num_trans_crystals_per_singles_unit())
    {
      stream << "the number of transaxial crystals per singles unit, ";
      ok = false;
    }

  if (!ok)
    {
      ret = stream.str();
      return Succeeded::no;
    }

  return Succeeded::yes;
}

Succeeded
CListModeDataROOT::check_scanner_definition(std::string& ret)
{
  if (num_rings == -1 || num_detectors_per_ring == -1 || max_num_non_arccorrected_bins == -1 || inner_ring_diameter == -1.f
      || average_depth_of_interaction == -1.f || ring_spacing == -.1f || bin_size == -1.f)
    {
      std::ostringstream stream(
          "CListModeDataROOT: The User_defined_scanner has not been fully described.\nPlease include in the hroot:\n");

      if (num_rings == -1)
        stream << "Number of rings := \n";

      if (num_detectors_per_ring == -1)
        stream << "Number of detectors per ring := \n";

      if (max_num_non_arccorrected_bins == -1)
        stream << "Maximum number of non-arc-corrected bins := \n";

      if (inner_ring_diameter == -1)
        stream << "Inner ring diameter (cm) := \n";

      if (average_depth_of_interaction == -1)
        stream << "Average depth of interaction (cm) := \n";

      if (ring_spacing == -1)
        stream << "Distance between rings (cm) := \n";

      if (bin_size == -1)
        stream << "Default bin size (cm) := \n";

      ret = stream.str();

      return Succeeded::no;
    }
  if (max_num_timing_bins <= 0 || size_timing_bin <= 1.F || timing_resolution <= 0.F)
    info(format("CListModeDataROOT: TOF information is missing. Set relevant keywords if you need TOF:\n\t{}\n\t{}\n\t{}",
                max_num_timing_bins_keyword,
                size_timing_bin_keyword,
                timing_resolution_keyword));

  return Succeeded::yes;
}

END_NAMESPACE_STIR
