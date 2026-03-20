/*
        Copyright 2026, UMCG
        Copyright 2020 Positrigo AG, Zurich

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details

 */

/*!

  \file
  \ingroup listmode
  \brief Declaration of class stir::CListModeDataBasedOnCoordinateMap

  \author Nikos Efthimiou
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
