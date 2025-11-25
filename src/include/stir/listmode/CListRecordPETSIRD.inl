/* CListRecordPETSIRD.inl

 Coincidence Event Class for PETSIRD: Inline File

        Copyright 2015 ETH Zurich, Institute of Particle Physics
        Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
        Copyright 2020, 2022 Positrigo AG, Zurich
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
  \brief Inline implementation of class stir::CListEventPETSIRD and stir::CListRecordPETSIRD with supporting classes

  \author Nikos Efthimiou
  \author Daniel Deidda
*/

#include "stir/LORCoordinates.h"
#include "stir/listmode/CListRecord.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/error.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"

START_NAMESPACE_STIR

stir::DetectionPosition<> 
CListEventPETSIRD::get_stir_det_pos_from_PETSIRD_id(const petsird::ExpandedDetectionBin& exp_det_bin) const
{
// const-friendly lookup
  auto it = petsird_to_stir->find(exp_det_bin);
  if (it == petsird_to_stir->end()) {
    // handle missing key however STIR usually does:
    // - throw
    // - or call error(...)
    // - or return a default DetectionPosition
    error("get_stir_det_pos_from_PETSIRD_id: PETSIRD id not found in petsird_to_stir map", exp_det_bin.module_index, exp_det_bin.element_index, exp_det_bin.energy_index);
  }

  return it->second; // copy of DetectionPosition<>
}

LORAs2Points<float>
CListEventPETSIRD::get_LOR() const
{
  LORAs2Points<float> lor;
  DetectionPositionPair<> det_pos_pair;

  det_pos_pair.pos1() = get_stir_det_pos_from_PETSIRD_id(exp_det_0); 
  det_pos_pair.pos2() = get_stir_det_pos_from_PETSIRD_id(exp_det_1); 

  lor.p1() = map_to_use().get_coordinate_for_index(det_pos_pair.pos1());
  lor.p2() = map_to_use().get_coordinate_for_index(det_pos_pair.pos2());
  
  // std::cout << "lor_p1 " << lor.p1().x() << " " << lor.p1().y() << " " << lor.p1().z() << std::endl;
  // std::cout << "lor_p2 " <<lor.p2().x() << " " << lor.p2().y() << " " << lor.p2().z() << std::endl;

  return lor;
}

void
CListEventPETSIRD::get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{

  DetectionPositionPair<> det_pos_pair;

  if(scanner_sptr->get_scanner_geometry() == "Cylindrical")
    {
      det_pos_pair.pos1() = get_stir_det_pos_from_PETSIRD_id(exp_det_0); 
      det_pos_pair.pos2() = get_stir_det_pos_from_PETSIRD_id(exp_det_1); 
      // this->get_data().get_detection_position_pair(det_pos_pair);
      dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(proj_data_info).get_bin_for_det_pos_pair(bin, det_pos_pair);
    }
    else
    {
      if (!map_sptr)
        {
          std::cerr << "Error: No detector map set in CListEventPETSIRD::get_bin()" << std::endl;
          // this->get_data().get_detection_position_pair(det_pos_pair);
        }
        else{
          DetectionPositionPair<> det_pos_pair;
          det_pos_pair.pos1() = get_stir_det_pos_from_PETSIRD_id(exp_det_0); 
          det_pos_pair.pos2() = get_stir_det_pos_from_PETSIRD_id(exp_det_1); 
          // std::cout<< exp_det_0.module_index << ", " << exp_det_0.element_index << " ---- " << exp_det_1.module_index << ", " << exp_det_1.element_index << std::endl;
          const stir::CartesianCoordinate3D<float> c1 = map_sptr->get_coordinate_for_index(det_pos_pair.pos1());
          const stir::CartesianCoordinate3D<float> c2 = map_sptr->get_coordinate_for_index(det_pos_pair.pos2());

          // std::cout << "CListEventPETSIRD::get_bin(): det_pos1: " << det_pos_pair.pos1().tangential_coord() << ", "
          //           << det_pos_pair.pos1().axial_coord() << ", " << det_pos_pair.pos1().radial_coord() << std::endl;
          // std::cout << "CListEventPETSIRD::get_bin(): det_pos2: " << det_pos_pair.pos2().tangential_coord() << ", "
          //           << det_pos_pair.pos2().axial_coord() << ", " << det_pos_pair.pos2().radial_coord() << std::endl;
          // std::cout << "CListEventPETSIRD::get_bin(): c1: " << c1.x() << ", " << c1.y() << ", " << c1.z() << std::endl;
          // std::cout << "CListEventPETSIRD::get_bin(): c2: " << c2.x() << ", " << c2.y() << ", " << c2.z() << std::endl;
          const LORAs2Points<float> lor(c1, c2);
          bin = proj_data_info.get_bin(lor);
        }
      
    }


}


END_NAMESPACE_STIR
