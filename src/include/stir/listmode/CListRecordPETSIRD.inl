/*      CListRecordPETSIRD.inl
        Coincidence Event Class for PETSIRD: Inline File

        Copyright 2025, 2026 UMCG
        Copyright 2025 National Physical Laboratory

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details.

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
#include "stir/error.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"

START_NAMESPACE_STIR

void
CListEventPETSIRD::get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{
  if (petsird_info_sptr->is_cylindrical_configuration_used() || petsird_info_sptr->is_block_configuration_used())
    {
      DetectionPositionPair<> det_pos_pair;
      get_detection_position_pair(det_pos_pair);
      //! Warning: assuming that STIR ProjDataInfo and PETSIRD TOF binning match. If you have tof_mashing this can be wrong.
      det_pos_pair.timing_pos() = static_cast<int>(m_tof_bin) + proj_data_info.get_min_tof_pos_num();
      // Try Blocks-on-Cylindrical first
      if (const auto* proj_blocks = dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr*>(&proj_data_info))
        {
          if (proj_blocks->get_bin_for_det_pos_pair(bin, det_pos_pair) == Succeeded::yes)
            bin.set_bin_value(1);
          else
            bin.set_bin_value(0);
        }
      // Else try Cylindrical
      else if (const auto* proj_cyl = dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(&proj_data_info))
        {
          if (proj_cyl->get_bin_for_det_pos_pair(bin, det_pos_pair) == Succeeded::yes)
            bin.set_bin_value(1);
          else
            bin.set_bin_value(0);
        }
    }
  else
    {
      error("CListEventPETSIRD::Implement Generic Scanner.");
    }
}

LORAs2Points<float>
CListEventPETSIRD::get_LOR() const
{
  // LORAs2Points<float> lor;
  // DetectionPositionPair<> det_pos_pair;

  // if (! is_null_ptr(petsird_to_stir))
  //   {
  //     if (!petsird_to_stir)
  //       {
  //         error("CListEventPETSIRD::get_bin: petsird_to_stir map not set. Probably your ProjDataInfo point to a Generic
  //         Scanner. "
  //               "\n The Scanner in the Listmode data and the one in the ProjDataInfo must match.");
  //       }
  //       get_detection_position_pair(det_pos_pair);
  //       lor.p1() = det_pos_pair.pos1();
  //       lor.p2() = det_pos_pair.pos2();

  //   }
  //   else
  //   {
  //     error ("CListEventPETSIRD::Implement Generic Scanner.");
  //   }
  // // else if (proj_data_info.get_scanner_sptr()->get_scanner_geometry() == "Generic")
  // //   {
  // //     // if (!map_sptr)
  // //     //         {
  // //     //           std::cerr << "Error: No detector map set in CListEventPETSIRD::get_bin()" << std::endl;
  // //     //           // this->get_data().get_detection_position_pair(det_pos_pair);
  // //     //         }
  // //     //         else{
  // //     //           DetectionPositionPair<> det_pos_pair;
  // //     //           det_pos_pair.pos1() = get_stir_det_pos_from_PETSIRD_id(exp_det_0);
  // //     //           det_pos_pair.pos2() = get_stir_det_pos_from_PETSIRD_id(exp_det_1);
  // //     //           // std::cout<< exp_det_0.module_index << ", " << exp_det_0.element_index << " ---- " <<
  // exp_det_1.module_index
  // //     //           << ", " << exp_det_1.element_index << std::endl; const stir::CartesianCoordinate3D<float> c1 =
  // //     //           map_sptr->get_coordinate_for_index(det_pos_pair.pos1()); const stir::CartesianCoordinate3D<float> c2 =
  // //     //           map_sptr->get_coordinate_for_index(det_pos_pair.pos2());

  // //     //           // std::cout << "CListEventPETSIRD::get_bin(): det_pos1: " << det_pos_pair.pos1().tangential_coord() <<
  // ", "
  // //     //           //           << det_pos_pair.pos1().axial_coord() << ", " << det_pos_pair.pos1().radial_coord() <<
  // std::endl;
  // //     //           // std::cout << "CListEventPETSIRD::get_bin(): det_pos2: " << det_pos_pair.pos2().tangential_coord() <<
  // ", "
  // //     //           //           << det_pos_pair.pos2().axial_coord() << ", " << det_pos_pair.pos2().radial_coord() <<
  // std::endl;
  // //     //           // std::cout << "CListEventPETSIRD::get_bin(): c1: " << c1.x() << ", " << c1.y() << ", " << c1.z() <<
  // //     //           std::endl;
  // //     //           // std::cout << "CListEventPETSIRD::get_bin(): c2: " << c2.x() << ", " << c2.y() << ", " << c2.z() <<
  // //     //           std::endl; const LORAs2Points<float> lor(c1, c2); bin = proj_data_info.get_bin(lor);
  // //     //         }
  // //   }
  // // else if (proj_data_info.get_scanner_sptr()->get_scanner_geometry() == "BlocksOnCylindrical")
  // //   {
  // //     if (!petsird_to_stir)
  // //       {
  // //         error("CListEventPETSIRD::get_bin: petsird_to_stir map not set. Probably your ProjDataInfo point to a Generic
  // Scanner. "
  // //               "\n The Scanner in the Listmode data and the one in the ProjDataInfo must match.");
  // //       }
  // //     det_pos_pair.pos1() = (*petsird_to_stir)[exp_det_1];
  // //     det_pos_pair.pos2() = (*petsird_to_stir)[exp_det_0];
  // //     // dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr&>(proj_data_info).get_bin_for_det_pos_pair(bin,
  // det_pos_pair);
  // //   }
  // // else
  // //   {
  // //     error("CListEventPETSIRD::get_bin: How did I get with an unsupported scanner geometry ? -",
  // //           proj_data_info.get_scanner_sptr()->get_scanner_geometry());
  // //   }

  // return lor;
}

END_NAMESPACE_STIR
