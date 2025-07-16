/* CListModeDataSAFIR.h

 Coincidence LM Data Class for SAFIR: Header File
 Jannis Fischer

        Copyright 2015 ETH Zurich, Institute of Particle Physics
        Copyright 2020 Positrigo AG, Zurich

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
  \brief Declaration of class stir::CListModeDataSAFIR

  \author Jannis Fischer
*/

#ifndef __stir_listmode_CListModeDataBasedOnCoordinateMap_H__
#define __stir_listmode_CListModeDataBasedOnCoordinateMap_H__

#include <string>
#include <vector>

#include "stir/listmode/CListModeData.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/listmode/CListRecord.h"
#include "stir/IO/InputStreamWithRecords.h"
#include "stir/shared_ptr.h"

// #include "stir/listmode/CListRecordSAFIR.h"
#include "stir/DetectorCoordinateMap.h"

START_NAMESPACE_STIR

class CListModeDataBasedOnCoordinateMap : public CListModeData
{
public:
  std::string get_name() const override;
  // shared_ptr<CListRecord> get_empty_record_sptr() const override;
  // Succeeded get_next_record(CListRecord& record_of_general_type) const override;
  Succeeded reset() override;

  virtual shared_ptr<InputStreamWithRecords<CListRecord, bool>> get_current_lm_file() = 0;

  SavedPosition save_get_position() override { return static_cast<SavedPosition>(get_current_lm_file()->save_get_position()); }
  Succeeded set_get_position(const SavedPosition& pos) override { return get_current_lm_file()->set_get_position(pos); }

protected:
  std::string listmode_filename;

  mutable std::vector<unsigned int> saved_get_positions;
  virtual Succeeded open_lm_file() const = 0;

  shared_ptr<DetectorCoordinateMap> map;
};

// CListModeDataBasedOnCoordinateMap::CListModeDataBasedOnCoordinateMap(const std::string& listmode_filename,
//                                                                      const std::string& crystal_map_filename,
//                                                                      const std::string& template_proj_data_filename,
//                                                                      const double lor_randomization_sigma)
//     : listmode_filename(listmode_filename)
// {
//   if (!crystal_map_filename.empty())
//     {
//       map = MAKE_SHARED<DetectorCoordinateMap>(crystal_map_filename, lor_randomization_sigma);
//     }
//   else
//     {
//       if (lor_randomization_sigma != 0)
//         error("SAFIR currently does not support LOR-randomisation unless a map is specified");
//     }
//   shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);
//   _exam_info_sptr->imaging_modality = ImagingModality::PT;
//   this->exam_info_sptr = _exam_info_sptr;

//          // Here we are reading the scanner data from the template projdata
//   shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(template_proj_data_filename);
//   this->set_proj_data_info_sptr(template_proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone());

//   if (open_lm_file() == Succeeded::no)
//     {
//       error("CListModeDataSAFIR: Could not open listmode file " + listmode_filename + "\n");
//     }
// }

// // CListModeDataBasedOnCoordinateMap::CListModeDataBasedOnCoordinateMap(const std::string& listmode_filename,
// //                                                      const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
// //     : listmode_filename(listmode_filename)
// // {
// //   shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);
// //   _exam_info_sptr->imaging_modality = ImagingModality::PT;
// //   this->exam_info_sptr = _exam_info_sptr;
// //   this->set_proj_data_info_sptr(proj_data_info_sptr->create_shared_clone());

// //   if (open_lm_file() == Succeeded::no)
// //     {
// //       error("CListModeDataSAFIR: opening file \"" + listmode_filename + "\"");
// //     }
// // }

END_NAMESPACE_STIR

#endif
