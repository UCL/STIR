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
#include <iostream>
#include <fstream>

#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/error.h"

#include "binary/protocols.h"
#include "helpers/include/petsird_helpers.h"
// #include "helpers/include/petsird_helpers/create.h"
#include "helpers/include/petsird_helpers/geometry.h"
// #include "boost/static_assert.hpp"

#include "stir/listmode/CListModeDataPETSIRD.h"
#include "stir/listmode/CListRecordPETSIRD.h"

START_NAMESPACE_STIR

CListModeDataPETSIRD::CListModeDataPETSIRD(const std::string& listmode_filename)
{
  this->listmode_filename = listmode_filename;

  petsird::Header header;
  petsird::binary::PETSIRDReader petsird_reader(listmode_filename);
  petsird_reader.ReadHeader(header);
  petsird::ScannerInformation scanner_info = header.scanner;
  petsird::ScannerGeometry scanner_geo = scanner_info.scanner_geometry;
  const petsird::TypeOfModule type_of_module{ 0 };
  // need to get  inner_ring_radius, num_detector_layers, num_transaxial_crystals_per_block, num_axial_crystals_per_block
  //  transaxial_crystal_spacing,average_depth_of_interaction,axial_crystal_spacing, num_rings, ring_spacing, num_axial_blocks,
  //  num_oftransaxial_blocks
  //  these are from rep_module.object

  std::vector<petsird::ReplicatedDetectorModule> replicated_module_list = scanner_geo.replicated_modules;
  int num_modules = scanner_geo.replicated_modules[type_of_module].transforms.size();
  int num_elements_per_module = scanner_geo.replicated_modules[type_of_module].object.detecting_elements.transforms.size();
  const auto& tof_bin_edges = header.scanner.tof_bin_edges[type_of_module][type_of_module];
  const auto num_tof_bins = tof_bin_edges.NumberOfBins();
  const auto& event_energy_bin_edges = header.scanner.event_energy_bin_edges[type_of_module];
  const auto num_event_energy_bins = event_energy_bin_edges.NumberOfBins();
  //   coordinates of first detecting bin (module_id,element_id, energy_id)
  //   const petsird::ExpandedDetectionBin expanded_detection_bin{ 0, 0, 0 };
  const auto box_shape = petsird_helpers::geometry::get_detecting_box(header.scanner, type_of_module, expanded_detection_bin);
  // get center of box (this should be in a loop to create a map

  for (uint32_t module = 0; module < num_modules; module++)
    for (uint32_t elem = 0; elem < num_elements_per_module; elem++)
      for (uint32_t ener = 0; ener < num_event_energy_bins; ener++)
        {
          petsird::ExpandedDetectionBin expanded_detection_bin{ module, elem, ener };
          const auto box_shape
              = petsird_helpers::geometry::get_detecting_box(header.scanner, type_of_module, expanded_detection_bin);
          CartesianCoordinate3D<float> mean_pos;
          for (auto& corner : box_shape.corners)
            { // if STIR (z,y,x) -> PETSIRD (-y, -x, z) pheraps  the order below needs to be changed
              mean_pos.x() = +corner.c[0] / box_shape.corners.size();
              mean_pos.y() = +corner.c[1] / box_shape.corners.size();
              mean_pos.z() = +corner.c[2] / box_shape.corners.size();
            }
          //       save  mean pos into map
        }

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
  shared_ptr<CListRecord> sptr(new CListRecordPETSIRD());
  return sptr;
}

Succeeded
CListModeDataPETSIRD::get_next_record(CListRecord& record_of_general_type) const
{
  CListRecordPETSIRD& record = dynamic_cast<CListRecordPETSIRD&>(record_of_general_type);
  // return current_lm_data_ptr->get_next_record(record);
}

END_NAMESPACE_STIR
