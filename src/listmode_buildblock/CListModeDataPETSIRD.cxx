/* CListModeDataPETSIRD.cxx

Coincidence LM Data Class for PETSIRD: Implementation

     Copyright 2015 ETH Zurich, Institute of Particle Physics
     Copyright 2020 Positrigo AG, Zurich
     Copyright 2021 University College London
     Copyright 2025 National Physical Laboratory


     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

             http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
*/
/*!

\file
\ingroup listmode
\brief implementation of class stir::CListModeDataPETSIRD

\author Jannis Fischer
\author Kris Thielemans
\author Markus Jehl
\author Daniel Deidda
*/
#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/error.h"

#include "../../PETSIRD/cpp/helpers/include/petsird_helpers.h"
#include "helpers/include/petsird_helpers/create.h"
#include "helpers/include/petsird_helpers/geometry.h"

#include "../../PETSIRD/cpp/generated/binary/protocols.h"
#include "../../PETSIRD/cpp/generated/hdf5/protocols.h"

#include "stir/listmode/CListModeDataPETSIRD.h"
#include "stir/listmode/CListRecordPETSIRD.h"
#include <set>

START_NAMESPACE_STIR

CListModeDataPETSIRD::CListModeDataPETSIRD(const std::string& listmode_filename, bool use_hdf5)
    : use_hdf5(use_hdf5)
{
  this->listmode_filename = listmode_filename;

  petsird::Header header;
  if (use_hdf5)
    current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(listmode_filename));
  else
    current_lm_data_ptr.reset(new petsird::binary::PETSIRDReader(listmode_filename));

  current_lm_data_ptr->ReadHeader(header);
  petsird::ScannerInformation scanner_info = header.scanner;
  petsird::ScannerGeometry scanner_geo = scanner_info.scanner_geometry;
  const petsird::TypeOfModule type_of_module{ 0 };
  // need to get  inner_ring_radius, num_detector_layers, num_transaxial_crystals_per_block, num_axial_crystals_per_block
  //  transaxial_crystal_spacing,average_depth_of_interaction,axial_crystal_spacing, num_rings, ring_spacing, num_axial_blocks,
  //  num_oftransaxial_blocks
  //  these are from rep_module.object

  // Get the first TimeBlock
  // if (
  current_lm_data_ptr->ReadTimeBlocks(curr_time_block);
  // )
  // error("CListModeDataPETSIRD: Could not read the first TimeBlock. Abord.");

  if (std::holds_alternative<petsird::EventTimeBlock>(curr_time_block))
    curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
  else
    error("CListModeDataPETSIRD: holds_alternative not true. Abord.");

  std::vector<petsird::ReplicatedDetectorModule> replicated_module_list = scanner_geo.replicated_modules;
  int num_modules = scanner_geo.replicated_modules[type_of_module].transforms.size();

  int num_elements_per_module = scanner_geo.replicated_modules[type_of_module].object.detecting_elements.transforms.size();

  const auto& tof_bin_edges = header.scanner.tof_bin_edges[type_of_module][type_of_module];
  // const auto num_tof_bins = tof_bin_edges.NumberOfBins();
  const auto& event_energy_bin_edges = header.scanner.event_energy_bin_edges[type_of_module];
  const auto num_event_energy_bins = event_energy_bin_edges.NumberOfBins();
  //   coordinates of first detecting bin (module_id,element_id, energy_id)
  const petsird::ExpandedDetectionBin expanded_detection_bin{ 0, 0, 0 };
  // const auto box_shape = petsird_helpers::geometry::get_detecting_box(header.scanner, type_of_module, expanded_detection_bin);

  // get center of box (this should be in a loop to create a map

  std::cout << scanner_info.gantry_alignment->matrix << std::endl;
  float radius = 0;

  int dd = 0;

  double epsilon = 1e-6;
  std::set<double> unique_z_values;
  for (uint32_t module = 0; module < num_modules; module++)
    {
      petsird::RigidTransformation& mod_trans = scanner_geo.replicated_modules[type_of_module].transforms[module];

      double tz = mod_trans.matrix.at(2, 3); // third row, fourth column
      unique_z_values.insert(tz);

      std::cout << mod_trans.matrix << "\r" << std::endl;
    }

  std::vector<double> sorted_z(unique_z_values.begin(), unique_z_values.end());
  std::vector<double> spacings;
  for (size_t i = 1; i < sorted_z.size(); ++i)
    {
      spacings.push_back(std::abs(sorted_z[i] - sorted_z[i - 1]));
    }

  double first = spacings.front();
  for (const auto& s : spacings)
    {
      if (std::abs(s - first) > epsilon)
        error("Unequally spaced blocks. Probably. Abord.");
    }

  std::cout << "I counted " << unique_z_values.size() << " axial number of blocks with spacing " << spacings[0] << std::endl;
  int num_heads = num_modules / unique_z_values.size();
  std::cout << "I deduce that the scanner has " << num_heads << "transaxial number of blocks" << std::endl;

  for (uint32_t elem = 0; elem < num_elements_per_module; elem++)
    {
      for (uint32_t ener = 0; ener < num_event_energy_bins; ener++)
        {

          // const auto& rep_module = scanner.scanner_geometry.replicated_modules[type_of_module];
          // const auto& det_els = rep_module.object.detecting_elements;
          // const auto& mod_transform = rep_module.transforms[expanded_detection_bin.module_index];
          // const auto& transform = det_els.transforms[expanded_detection_bin.element_index];
          // return transform_BoxShape(mult_transforms({ mod_transform, transform }), det_els.object.shape);

          // petsird::ExpandedDetectionBin expanded_detection_bin{ module, elem, ener };
          // const auto box_shape
          //     = petsird_helpers::geometry::get_detecting_box(header.scanner, type_of_module, expanded_detection_bin);

          petsird::RigidTransformation& el_trans
              = scanner_geo.replicated_modules[type_of_module].object.detecting_elements.transforms[elem]; // .transforms[module];
          if (radius == 0)
            radius = el_trans.matrix.at(0, 3);
          else if (radius != el_trans.matrix.at(0, 3))
            error("Unsupported mixed radii. Abord.");

          dd++;
        }
    }

  int nikos = 0;

  // this_scanner_sptr.reset(new Scanner(Scanner::User_defined_scanner,
  //                                     std::string("PETSIRD_defined_scanner"),
  //                                     /* num dets per ring */
  //                                     num_modules,
  //                                     /* num of rings */
  //                                     num_elements_per_module,
  //                                     /* number of non arccor bins */
  //                                     num_modules / 2,
  //                                     /* number of maximum arccor bins */
  //                                     this->default_num_arccorrected_bins,
  //                                     /* inner ring radius */
  //                                     radius,
  //                                     /* doi */ 0.1F,
  //                                     /* ring spacing */
  //                                     this->ring_spacing * 10.f,
  //                                     this->bin_size * 10.f,
  //                                     /* offset*/
  //                                     this->view_offset * _PI / 180,
  //                                     /*num_axial_blocks_per_bucket_v */
  //                                     this->root_file_sptr->get_num_axial_blocks_per_bucket_v(),
  //                                     /*num_transaxial_blocks_per_bucket_v*/
  //                                     this->root_file_sptr->get_num_transaxial_blocks_per_bucket_v(),
  //                                     /*num_axial_crystals_per_block_v*/
  //                                     this->root_file_sptr->get_num_axial_crystals_per_block_v(),
  //                                     /*num_transaxial_crystals_per_block_v*/
  //                                     this->root_file_sptr->get_num_transaxial_crystals_per_block_v(),
  //                                     /*num_axial_crystals_per_singles_unit_v*/
  //                                     this->root_file_sptr->get_num_axial_crystals_per_singles_unit(),
  //                                     /*num_transaxial_crystals_per_singles_unit_v*/
  //                                     this->root_file_sptr->get_num_trans_crystals_per_singles_unit(),
  //                                     /*num_detector_layers_v*/ 1,
  //                                     this->energy_resolution,
  //                                     this->reference_energy,
  //                                     /* maximum number of timing bins */
  //                                     max_num_timing_bins,
  //                                     /* size of basic TOF bin */
  //                                     size_timing_bin,
  //                                     /* Scanner's timing resolution */
  //                                     timing_resolution));

  //   petsird::ReplicatedDetectorModule = scanner_geo.replicated_modules;

  // if (!crystal_map_filename.empty())
  //   {
  //     this->map = MAKE_SHARED<DetectorCoordinateMap>(crystal_map_filename, lor_randomization_sigma);
  //   }
  // else
  //   {
  //     if (lor_randomization_sigma != 0)
  //       error("PETSIRD currently does not support LOR-randomisation unless a map is specified");
  //   }
  // shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);
  // _exam_info_sptr->imaging_modality = ImagingModality::PT;
  // this->exam_info_sptr = _exam_info_sptr;

  //        // Here we are reading the scanner data from the template projdata
  // shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(template_proj_data_filename);
  // this->set_proj_data_info_sptr(template_proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone());

  if (this->open_lm_file() == Succeeded::no)
    {
      error("CListModeDataPETSIRD: Could not open listmode file " + listmode_filename + "\n");
    }
}

Succeeded
CListModeDataPETSIRD::open_lm_file() const
{
  current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(listmode_filename));
  return Succeeded::yes;
}

shared_ptr<CListRecord>
CListModeDataPETSIRD::get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordPETSIRD);
  std::dynamic_pointer_cast<CListRecordPETSIRD>(sptr)->event_PETSIRD().set_scanner_sptr(
      this->get_proj_data_info_sptr()->get_scanner_sptr());
  std::dynamic_pointer_cast<CListRecordPETSIRD>(sptr)->event_PETSIRD().set_map_sptr(map);

  return sptr;
}

Succeeded
CListModeDataPETSIRD::get_next_record(CListRecord& record_of_general_type) const
{
  CListRecordPETSIRD& record = dynamic_cast<CListRecordPETSIRD&>(record_of_general_type);
  petsird::CoincidenceEvent& curr_event
      = curr_event_block.prompt_events[type_of_module_pair[0]][type_of_module_pair[1]].at(curr_event_in_event_block);

  Succeeded ok = record.init_from_data_ptr(curr_event);

  if (ok == Succeeded::no)
    return Succeeded::no;

  curr_event_in_event_block++;
  if (curr_event_in_event_block == curr_event_block.prompt_events.size())
    {
      if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
        return Succeeded::no;
      curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
      curr_event_in_event_block = 0;
    }
  return Succeeded::yes;
}

END_NAMESPACE_STIR
