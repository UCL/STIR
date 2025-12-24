/* CListModeDataPETSIRD.h

Coincidence LM Data Class for PETSIRD

     Copyright 2025, UMCG

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
\brief Place that converts PETSIRD geometry to STIR geometry

\author Nikos Efthimiou
*/

#ifndef __stir_listmode_PETSIRDInfo_H__
#define __stir_listmode_PETSIRDInfo_H__

#include "stir/DetectionPosition.h"
#include "stir/DetectionPositionPair.h"
#include "petsird/protocols.h"
#include "stir/Scanner.h"
#include "stir/DetectorCoordinateMap.h"
#include "stir/ProjDataInfo.h"
#include <set>
#include <boost/format.hpp>

#include "stir/info.h"
#include "stir/error.h"

START_NAMESPACE_STIR

/*!
  \brief Comparator for ordering petsird::ExpandedDetectionBin in std::map.

  \details
  Orders by module_index, then element_index, then energy_index.
*/
struct ExpandedDetectionBinLess
{

  bool operator()(const petsird::ExpandedDetectionBin& a, const petsird::ExpandedDetectionBin& b) const
  {
    // Adjust field names if needed (I assume: module, element, energy_bin)
    if (a.module_index < b.module_index)
      return true;
    if (a.module_index > b.module_index)
      return false;

    if (a.element_index < b.element_index)
      return true;
    if (a.element_index > b.element_index)
      return false;
    return a.energy_index < b.energy_index;
  }
};

/*!
  \brief Mapping type from PETSIRD ExpandedDetectionBin to STIR DetectionPosition.
*/
using PETSIRDToSTIRDetectorIndexMap
    = std::map<petsird::ExpandedDetectionBin, stir::DetectionPosition<>, ExpandedDetectionBinLess>;

using STIRToPETSIRDDetectorIndexMap = std::map<stir::DetectionPosition<>, petsird::ExpandedDetectionBin>;

/*!
  \brief Class to hold PETSIRD-related information for STIR and do any necessary conversions.
*/
class PETSIRDInfo
{
public:
  explicit PETSIRDInfo(const petsird::Header& header, std::string scanner_geometry = "");

  // void initialize();

  inline std::shared_ptr<Scanner> get_scanner_sptr() const { return stir_scanner_sptr; }

  inline shared_ptr<const petsird::ScannerInformation> get_petsird_scanner_info_sptr() const { return petsird_scanner_info_sptr; }

  inline shared_ptr<PETSIRDToSTIRDetectorIndexMap> get_petsird_to_stir_map() const { return petsird_to_stir; }

  inline shared_ptr<STIRToPETSIRDDetectorIndexMap> get_stir_to_petsird_map() const { return stir_to_petsird; }

  inline shared_ptr<DetectorCoordinateMap::det_pos_to_coord_type> get_petsird_map_sptr() const { return petsird_map_sptr; }

  inline bool is_generic_geometry_used() const { return is_generic_geometry; }

  inline bool is_block_configuration_used() const { return is_block_configuration; }

  inline bool is_cylindrical_configuration_used() const { return is_cylindrical; };

  inline float get_detection_efficiency_for_bin(const stir::DetectionPositionPair<>& dp) const
  {
    // std::cout << "WIth det efficiencies: " <<
    // petsird_scanner_info_sptr->detection_efficiencies.detection_bin_efficiencies->size()
    //           << std::endl;
    float eff = 1.0f;

    const auto& detection_bin_efficiencies = petsird_scanner_info_sptr->detection_efficiencies.detection_bin_efficiencies;

    if (!detection_bin_efficiencies)
      return eff; // no efficiencies available

    auto it0 = stir_to_petsird->find(dp.pos1());
    if (it0 == stir_to_petsird->end())
      {
        info(boost::format("DetectionPosition pos1(): "
                           "tangential %1%, "
                           "axial %2%, "
                           "radial %3%")
             % dp.pos1().tangential_coord() % dp.pos1().axial_coord() % dp.pos1().radial_coord());
        error("BinNormalisationFromPETSIRD: DetectionPosition not found in STIR→PETSIRD map");
      }
    auto it1 = stir_to_petsird->find(dp.pos2());

    if (it1 == stir_to_petsird->end())
      {
        info(boost::format("DetectionPosition pos2(): "
                           "tangential %1%, "
                           "axial %2%, "
                           "radial %3%")
             % dp.pos2().tangential_coord() % dp.pos2().axial_coord() % dp.pos2().radial_coord());
        error("BinNormalisationFromPETSIRD: DetectionPosition not found in STIR→PETSIRD map");
      }

    return 1.f;
    // if (detection_bin_efficiencies)
    //   {
    //     // eff *= ((*detection_bin_efficiencies)[0](detection_bin_1)
    //     //         * (*detection_bin_efficiencies)[1](detection_bin_2));
    //     if (eff == 0.F)
    //       return 0.F;
    //   }

    // it0->second().module_index;
    // it0->second().element_index;
    // it1->second().module_index;
    // it1->second().element_index;

    // const auto& num_en0 = petsird_scanner_info_sptr->event_energy_bin_edges[0].NumberOfBins();
    // const auto& num_en1 = petsird_scanner_info_sptr->event_energy_bin_edges[0].NumberOfBins();

    // // TODO create helper for next calculation
    // // eff *= module_pair_efficiencies.values(expanded_det_bin0.element_index * num_en0 + expanded_det_bin0.energy_index,
    // //                                        expanded_det_bin1.element_index * num_en1 + expanded_det_bin1.energy_index);
    // // Placeholder implementation
    int nikos = 0;
  }

  float get_lower_energy_threshold() const
  {
    if (petsird_scanner_info_sptr->event_energy_bin_edges.size() == 0)
      return 0.0f;
    float min_energy = std::numeric_limits<float>::max();
    for (const auto& bin_edges : petsird_scanner_info_sptr->event_energy_bin_edges)
      {
        if (bin_edges.edges.front() < min_energy)
          min_energy = bin_edges.edges.front();
      }
    return min_energy;
  }

  float get_upper_energy_threshold() const
  {
    if (petsird_scanner_info_sptr->event_energy_bin_edges.size() == 0)
      return 0.0f;
    float max_energy = std::numeric_limits<float>::lowest();
    for (const auto& bin_edges : petsird_scanner_info_sptr->event_energy_bin_edges)
      {
        if (bin_edges.edges.back() > max_energy)
          max_energy = bin_edges.edges.back();
      }
    return max_energy;
  }

private:
  /*!
     \brief Infer scanner blocks and rotation axis from PETSIRD replicated modules.
     \param unique_dim1_values Output set of unique translations along X.
     \param unique_dim2_values Output set of unique translations along Y.
     \param unique_dim3_values Output set of unique translations along Z.
     \param replicated_module_list PETSIRD replicated detector modules.
     \return Index of rotation axis (0=x, 1=y, 2=z) or -1 if not found.
   */
  int figure_out_scanner_blocks_and_rotation_axis(std::set<float>& unique_dim1_values,
                                                  std::set<float>& unique_dim2_values,
                                                  std::set<float>& unique_dim3_values);
  /*!
  \brief Compute unique module rotation angles around the given axis.
  \param unique_angle_modules Output set of unique angles (radians).
  \param rot_axis Rotation axis index (0=x, 1=y, 2=z).
  */
  void figure_out_block_angles(std::set<float>& unique_angle_modules, const int rot_axis);
  /*!
  \brief Determine block element translations and radius.
  \param unique_dim1_values Output set of element translations along X.
  \param unique_dim2_values Output set of element translations along Y.
  \param unique_dim3_values Output set of element translations along Z.
  \param radius Output inferred radius.
  \param radius_index Output axis index containing the radius component.
  \param rotation_axis Known rotation axis (0=x, 1=y, 2=z).
  \param replicated_module_list PETSIRD replicated detector modules.
*/
  void figure_out_block_element_transformations(std::set<float>& unique_dim1_values,
                                                std::set<float>& unique_dim2_values,
                                                std::set<float>& unique_dim3_values,
                                                float& radius,
                                                int& radius_index,
                                                const int rotation_axis);

  //! Scanner information as provided by PETSIRD.
  shared_ptr<const petsird::ScannerInformation> petsird_scanner_info_sptr;
  //! Active scanner instance.
  shared_ptr<Scanner> stir_scanner_sptr;

  shared_ptr<const petsird::Header> petsird_header_sptr;

  //! Number of replicated modules.
  uint32_t numberOfModules;
  //! Number of element indices per module.
  uint32_t numberOfElementsIndices;
  //! Transaxial blocks per bucket (scanner metadata).
  uint32_t blocks_per_bucket_transaxial;
  //! Axial blocks per bucket (scanner metadata).
  uint32_t blocks_per_bucket_axial;
  //! Number of axial crystals per block.
  uint32_t num_axial_crystals_per_block;
  //! Number of transaxial crystals per block.
  uint32_t num_trans_crystals_per_block;

  bool is_cylindrical = true;

  bool is_generic_geometry = false;

  bool is_block_configuration = false;

  std::string forced_geometry = "";
  //! Mapping from PETSIRD expanded bins to STIR detection positions.
  shared_ptr<PETSIRDToSTIRDetectorIndexMap> petsird_to_stir;

  shared_ptr<STIRToPETSIRDDetectorIndexMap> stir_to_petsird;

  //! Mapping from STIR detection positions to PETSIRD coordinates.
  shared_ptr<DetectorCoordinateMap::det_pos_to_coord_type> petsird_map_sptr;
};

END_NAMESPACE_STIR

#endif