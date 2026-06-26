/*

     Copyright 2015 ETH Zurich, Institute of Particle Physics
     Copyright 2020 Positrigo AG, Zurich

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for detail.
 */

/*!

\file
\ingroup listmode
\brief Declaration of class stir::CListModeDataSAFIR

\author Jannis Fischer
*/

#ifndef __stir_listmode_CListModeDataSAFIR_H__
#define __stir_listmode_CListModeDataSAFIR_H__

#include <string>

#include "stir/listmode/CListModeDataBasedOnCoordinateMap.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/listmode/CListRecord.h"
#include "stir/IO/InputStreamWithRecords.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

/*!
  \brief Class for reading SAFIR listmode data with variable geometry
  \ingroup listmode
  \par
  By providing crystal map and template projection data files, the coordinates are read from files and used defining the LOR
  coordinates.
*/
template <class CListRecordT>
class CListModeDataSAFIR : public CListModeDataBasedOnCoordinateMap
{
public:
  /*! Constructor
   \par
   Takes as arguments the filenames of the coicidence listmode file, the crystal map (text) file, and the template projection data
   file
   */
  CListModeDataSAFIR(const std::string& listmode_filename,
                     const std::string& crystal_map_filename,
                     const std::string& template_proj_data_filename,
                     const double lor_randomization_sigma = 0.0);

  CListModeDataSAFIR(const std::string& listmode_filename, const shared_ptr<const ProjDataInfo>& proj_data_info_sptr);

  shared_ptr<CListRecord> get_empty_record_sptr() const override;
  Succeeded get_next_record(CListRecord& record_of_general_type) const override;

  bool has_delayeds() const override { return false; }

  Succeeded reset() override;

  SavedPosition save_get_position() override { return static_cast<SavedPosition>(current_lm_data_ptr->save_get_position()); }

  Succeeded set_get_position(const SavedPosition& pos) override { return current_lm_data_ptr->set_get_position(pos); }

protected:
  virtual Succeeded open_lm_file() const override;
  mutable shared_ptr<InputStreamWithRecords<CListRecordT, bool>> current_lm_data_ptr;
};

END_NAMESPACE_STIR
#endif // CLISTMODEDATASAFIR_H
