/* CListModeDataPETSIRD.h

Coincidence LM Data Class for PETSIRD

     Copyright 2025, UMCG
     Copyright 2025, MGH / HST A. Martinos Center for Biomedical Imaging

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
\brief Declaration of class stir::CListModeDataPETSIRD

\author Daniel Deidda
\author Nikos Efthimiou
*/

#ifndef __stir_listmode_CListModeDataPETSIRD_H__
#define __stir_listmode_CListModeDataPETSIRD_H__

#include <set>
#include "stir/listmode/CListModeDataBasedOnCoordinateMap.h"
#include "stir/ProjData.h"
#include "stir/listmode/CListRecord.h"
#include "stir/shared_ptr.h"

#include "petsird/protocols.h"

START_NAMESPACE_STIR

/*!
  \brief Comparator for ordering petsird::ExpandedDetectionBin in std::map.

  \details
  Orders by module_index, then element_index, then energy_index.
*/
struct ExpandedDetectionBinLess
{

    bool operator()(const petsird::ExpandedDetectionBin& a,
                    const petsird::ExpandedDetectionBin& b) const
    {
        // Adjust field names if needed (I assume: module, element, energy_bin)
        if (a.module_index < b.module_index) return true;
        if (a.module_index > b.module_index) return false;

        if (a.element_index < b.element_index) return true;
        if (a.element_index > b.element_index) return false;
        return a.energy_index < b.energy_index;
    }
};

/*!
  \brief Mapping type from PETSIRD ExpandedDetectionBin to STIR DetectionPosition.
*/
using PETSIRDToSTIRMap = std::map<
    petsird::ExpandedDetectionBin,
    stir::DetectionPosition<>,
    ExpandedDetectionBinLess
>;

/*!
  \class CListModeDataPETSIRD
  \brief Reader for PETSIRD listmode data supporting variable geometry.
  \ingroup listmode

  \par Overview
  - Supports HDF5 and binary PETSIRD formats.
  - Infers scanner geometry:
    - Cylindrical → creates cylindrical scanner.
    - Block-based → creates block-based scanner.
    - Otherwise → creates generic scanner using crystal positions.
  - Builds a DetectorCoordinateMap when needed and stores to disk. 
  \par

  Infering the scanner geometry makes a lot of assumptions about what PET is. 
  In particular, it assumes:
  \li A PET scanner is made of rings of detectors
  \li The largest axis is the axial one.
  \li So far we support only a single layer. This is partly hard-coded for simplicity. (look in the code for relevant TODOs and comments.)
  \li Some of the hardcoded assumptions are in CListRecordPETSIRD as well.

  \note Exact PETSIRD format specification is defined in the PETSIRD project documentation. (NOTE: It looks like GATE.)
  \note Initially, I wanted to: 
        - Is close to a cylindrical geometry ?
          - then yes use a cylindrical scanner that is simpler. 
          - Else, is it made of blocks arranged on a cylinder.

        However, now I do the following: 
        - Is close to cylindriacl geometry? 
          - yes use cylindrical scanner
          - Check if blocks-on-cylinder configuration, are a good match. 
            - yes use blocks-on-cylinder scanner
            - else use generic scanner and export the map to the disk.  

  If listmode reconstruciton is done, the map is regenerated on-the-fly.

*/
class CListModeDataPETSIRD : public CListModeDataBasedOnCoordinateMap
{
public:
  /*!
    \brief Construct reader.
    \param listmode_filename Path to PETSIRD listmode file.
    \param use_hdf5 If true, use HDF5 reader; otherwise use binary reader.
  */
  CListModeDataPETSIRD(const std::string& listmode_filename, bool use_hdf5);

  virtual shared_ptr<CListRecord> get_empty_record_sptr() const override;

  Succeeded get_next_record(CListRecord& record_of_general_type) const override;

  SavedPosition save_get_position() override { return static_cast<SavedPosition>(curr_event_in_event_block); }

  Succeeded set_get_position(const SavedPosition& pos) override { return Succeeded::yes; }

  virtual bool has_delayeds() const override { return m_has_delayeds; }

  Succeeded reset() override { return Succeeded::yes; }

protected:
  virtual Succeeded open_lm_file() const override;

  mutable shared_ptr<petsird::PETSIRDReaderBase> current_lm_data_ptr;

private:
  //! Whether to use the HDF5 reader.
  const bool use_hdf5;

  mutable unsigned long int curr_event_in_event_block = 0;

  mutable petsird::TimeBlock curr_time_block;

  //! Number of replicated modules.
  int numberOfModules;
  //! Number of element indices per module.
  int numberOfElementsIndices;
  //! Transaxial blocks per bucket (scanner metadata).
  int blocks_per_bucket_transaxial; 
  //! Axial blocks per bucket (scanner metadata).
  int blocks_per_bucket_axial;
  //! Number of axial crystals per block.
  int num_axial_crystals_per_block;
  //! Number of transaxial crystals per block.
  int num_trans_crystals_per_block; 

  mutable petsird::EventTimeBlock curr_event_block;
  //! Mapping from PETSIRD expanded bins to STIR detection positions.
  shared_ptr<PETSIRDToSTIRMap> petsird_to_stir;
  //! Active module pair (prompt/delayed, or two modules for coincidences).
  //! \todo: This hard-codes a single/matterial layer detector assumption.
  petsird::TypeOfModulePair type_of_module_pair{ 0, 0 };
  //! Active scanner instance.
  shared_ptr<Scanner> this_scanner_sptr;
  //! Scanner information as provided by PETSIRD.
  shared_ptr<petsird::ScannerInformation> scanner_info;
  //! Current event prompt flag.
  mutable bool curr_is_prompt = true;
  //! Whether delayed events are present.
  mutable bool m_has_delayeds;
  /*!
    \brief Detect if the PETSIRD geometry is cylindrical and initialise scanner/map accordingly.
    \param replicated_module_list PETSIRD replicated detector modules.
    \return True if cylindrical configuration was detected.
  */
  bool isCylindricalConfiguration(const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list);
  
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
                                                  std::set<float>& unique_dim3_values,
                                                  const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list);
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
                                                const int rotation_axis,
                                                const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list);
    /*!
    \brief Compute unique module rotation angles around the given axis.
    \param unique_angle_modules Output set of unique angles (radians).
    \param rot_axis Rotation axis index (0=x, 1=y, 2=z).
    \param replicated_module_list PETSIRD replicated detector modules.
  */
  void figure_out_block_angles(std::set<float>& unique_angle_modules,
                               const int rot_axis,
                               const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list);
};

END_NAMESPACE_STIR
#endif // CLISTMODEDATAPETSIRD_H
