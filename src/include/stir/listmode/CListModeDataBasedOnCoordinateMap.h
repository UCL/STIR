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

  // virtual shared_ptr<InputStreamWithRecords<CListRecord, bool>> get_current_lm_file() = 0;

  SavedPosition save_get_position() override = 0;

  Succeeded set_get_position(const SavedPosition& pos) override = 0;

protected:
  std::string listmode_filename;

  mutable std::vector<unsigned int> saved_get_positions;
  virtual Succeeded open_lm_file() const = 0;

  shared_ptr<DetectorCoordinateMap> map;
};

END_NAMESPACE_STIR

#endif
