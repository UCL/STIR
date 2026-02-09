/* CListModeDataPETSIRD.cxx

Coincidence LM Data Class for PETSIRD: Implementation

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
\brief implementation of class stir::CListModeDataPETSIRD

\author Daniel Deidda
\author Nikos Efthimiou
*/

#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/error.h"


// #include "petsird_helpers/create.h"
// #include "petsird_helpers/geometry.h"

#include "stir/listmode/CListModeDataPETSIRD.h"
#include "stir/listmode/CListRecordPETSIRD.h"

START_NAMESPACE_STIR


CListModeDataPETSIRD::CListModeDataPETSIRD(const std::string& listmode_filename, bool use_hdf5)
    : use_hdf5(use_hdf5)
{
  CListModeDataBasedOnCoordinateMap::listmode_filename = listmode_filename;

  petsird::Header header;
  if (use_hdf5)
    current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(listmode_filename));
  else
    current_lm_data_ptr.reset(new petsird::binary::PETSIRDReader(listmode_filename));

  m_has_delayeds = false;

  current_lm_data_ptr->ReadHeader(header);

  // Get the first TimeBlock
  current_lm_data_ptr->ReadTimeBlocks(curr_time_block);
  if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
    error("CListModeDataPETSIRD: Could not read the first TimeBlock. Abord.");

  ++m_time_block_index;

  if (std::holds_alternative<petsird::EventTimeBlock>(curr_time_block))
    curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
  else
    error("CListModeDataPETSIRD: holds_alternative not true. Abort.");

  petsird_info_sptr = std::make_shared<PETSIRDInfo>(header);
  auto stir_scanner_sptr = petsird_info_sptr->get_scanner_sptr();

  int tof_mash_factor = 1;
  this->set_proj_data_info_sptr(std::const_pointer_cast<const ProjDataInfo>(
      ProjDataInfo::construct_proj_data_info(petsird_info_sptr->get_scanner_sptr(),
                                             1,
                                             petsird_info_sptr->get_scanner_sptr()->get_num_rings() - 1,
                                             petsird_info_sptr->get_scanner_sptr()->get_num_detectors_per_ring() / 2,
                                             petsird_info_sptr->get_scanner_sptr()->get_max_num_non_arccorrected_bins(),
                                             /* arc_correction*/ false,
                                             tof_mash_factor)
          ->create_shared_clone()));

  shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);
  // Only PET scanners supported
  _exam_info_sptr->imaging_modality = ImagingModality::PT;
  _exam_info_sptr->originating_system = std::string("PETSIRD_defined_scanner");
  _exam_info_sptr->set_low_energy_thres(petsird_info_sptr->get_lower_energy_threshold());
  _exam_info_sptr->set_high_energy_thres(petsird_info_sptr->get_upper_energy_threshold());

  this->exam_info_sptr = _exam_info_sptr;

  // N.E.: In my experience the first time block is always empty.
  // So I use this unncessessary call to skip to the next.
  if (this->open_lm_file() == Succeeded::no)
    {
      error("CListModeDataPETSIRD: Could not open listmode file " + listmode_filename + "\n");
    }
}

Succeeded
CListModeDataPETSIRD::open_lm_file() const
{
  // current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(listmode_filename));
  if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
    return Succeeded::no;
  curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
  return Succeeded::yes;
}

shared_ptr<CListRecord>
CListModeDataPETSIRD::get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordPETSIRD(petsird_info_sptr));
  return sptr;
}

Succeeded
CListModeDataPETSIRD::get_next_record(CListRecord& record_of_general_type) const
{
  auto& record = dynamic_cast<CListRecordPETSIRD&>(record_of_general_type);
  const auto& prompt_list = curr_event_block.prompt_events.at(0).at(0); // TODO: support mulitple pairs of modules.
  const auto& delayed_list = m_has_delayeds ? curr_event_block.delayed_events.at(0).at(0) : prompt_list;

  const auto& event_list = curr_is_prompt ? prompt_list : delayed_list;

  if (event_list.size() == 0)
    return Succeeded::no;

  auto event = event_list.at(curr_event_in_event_block);

  if (record.init_from_data(event, curr_is_prompt) == Succeeded::no
      || record_of_general_type.time().set_time_in_millisecs(curr_event_block.time_interval.start) == Succeeded::no)
    {
      return Succeeded::no;
    }

  ++curr_event_in_event_block;

  if (curr_event_in_event_block < event_list.size())
    {
      return Succeeded::yes;
    }

  // - Once we hit the size of the vector
  curr_event_in_event_block = 0;

  if (!m_has_delayeds || curr_is_prompt)
    {
      if (m_has_delayeds)
        {
          curr_is_prompt = false;
        }
      else
        {
          if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
          {
            current_lm_data_ptr->Close();
            return Succeeded::no;
          }
          ++m_time_block_index;
          curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
        }
    }
  else
    {
      curr_is_prompt = true;
      if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
      {
        current_lm_data_ptr->Close();
        return Succeeded::no;
      }
      ++m_time_block_index;
      curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
    }

  return Succeeded::yes;
}

END_NAMESPACE_STIR
