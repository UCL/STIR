/* CListModeDataPETSIRD.h

Coincidence LM Data Class for PETSIRD

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
  \brief Class for reading PETSIRD listmode data with variable geometry
  \ingroup listmode
  \par
  By providing crystal map and template projection data files, the coordinates are read from files and used defining the LOR
  coordinates.
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

using PETSIRDToSTIRMap = std::map<
    petsird::ExpandedDetectionBin,
    stir::DetectionPosition<>,
    ExpandedDetectionBinLess
>;

class CListModeDataPETSIRD : public CListModeDataBasedOnCoordinateMap
{
public:
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
  const bool use_hdf5;

  mutable unsigned long int curr_event_in_event_block = 0;

  mutable petsird::TimeBlock curr_time_block;

  int numberOfModules;

  int numberOfElementsIndices;

  int blocks_per_bucket_transaxial; 

  int blocks_per_bucket_axial;

  int num_axial_crystals_per_block;

  int num_trans_crystals_per_block; 

  mutable petsird::EventTimeBlock curr_event_block;

  shared_ptr<PETSIRDToSTIRMap> petsird_to_stir;

  petsird::TypeOfModulePair type_of_module_pair{ 0, 0 };

  shared_ptr<Scanner> this_scanner_sptr;

  shared_ptr<petsird::ScannerInformation> scanner_info;

  mutable bool curr_is_prompt = true;

  mutable bool m_has_delayeds;

  bool isCylindricalConfiguration(const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list);

  void find_uniqe_values_1D(std::set<float>& values, const std::vector<float>& input);

  void find_uniqe_values_2D(std::set<float>& values, const std::vector<std::vector<float>>& input);

  int figure_out_scanner_blocks_and_rotation_axis(std::set<float>& unique_dim1_values,
                                                  std::set<float>& unique_dim2_values,
                                                  std::set<float>& unique_dim3_values,
                                                  const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list);
  void figure_out_block_element_transformations(std::set<float>& unique_dim1_values,
                                                std::set<float>& unique_dim2_values,
                                                std::set<float>& unique_dim3_values,
                                                float& radius,
                                                int& radius_index,
                                                const int rotation_axis,
                                                const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list);

  void figure_out_block_angles(std::set<float>& unique_angle_modules,
                               const int rot_axis,
                               const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list);
};

END_NAMESPACE_STIR
#endif // CLISTMODEDATAPETSIRD_H
