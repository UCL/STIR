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

#include "petsird/protocols.h"
#include "stir/Scanner.h"
#include "stir/DetectorCoordinateMap.h"
#include <set>

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
using PETSIRDToSTIRDetectorIndexMap = std::map<petsird::ExpandedDetectionBin, stir::DetectionPosition<>, ExpandedDetectionBinLess>;

/*!
  \brief Class to hold PETSIRD-related information for STIR and do any necessary conversions.
*/
class PETSIRDInfo
{
public:
  explicit PETSIRDInfo(shared_ptr<const petsird::ScannerInformation>);

  // void initialize();

  inline std::shared_ptr<Scanner> get_scanner_sptr() const
  {
    return stir_scanner_sptr;
  }

  inline shared_ptr<PETSIRDToSTIRDetectorIndexMap> get_petsird_to_stir_map() const
  {
    return petsird_to_stir;
  }

  inline shared_ptr<DetectorCoordinateMap::det_pos_to_coord_type> get_petsird_map_sptr() const
  {
    return petsird_map_sptr;
  } 

  bool is_cylindrical_configuration() { return is_cylindrical; };

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
  //! Mapping from PETSIRD expanded bins to STIR detection positions.
  shared_ptr<PETSIRDToSTIRDetectorIndexMap> petsird_to_stir;
  //! Mapping from STIR detection positions to PETSIRD coordinates.
  shared_ptr<DetectorCoordinateMap::det_pos_to_coord_type> petsird_map_sptr;
};

END_NAMESPACE_STIR