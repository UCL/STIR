/* CListModeDataSAFIR.h

 Coincidence LM Data Class for SAFIR: Header File
 Jannis Fischer
 jannis.fischer@cern.ch
 
	Copyright 2015 ETH Zurich, Institute of Particle Physics

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

#ifndef __stir_listmode_CListModeDataSAFIR_H__
#define __stir_listmode_CListModeDataSAFIR_H__

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "stir/listmode/CListModeData.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/listmode/CListRecord.h"
#include "stir/IO/InputStreamWithRecords.h"
#include "stir/shared_ptr.h"

#include "stir/listmode/CListRecordSAFIR.h"
#include "stir/listmode/DetectorCoordinateMapFromFile.h"

START_NAMESPACE_STIR

/*!
  \brief Class for reading SAFIR listmode data with variable geometry 
  \ingroup listmode
  \par
  By providing crystal map and template projection data files, the coordinates are read from files and used defining the LOR coordinates.
*/
template <class CListRecordT> class CListModeDataSAFIR : public CListModeData
{
public:
	/*! Constructor
	\par
	Takes as arguments the filenames of the coicidence listmode file, the crystal map (text) file, and the template projection data file
	*/
	CListModeDataSAFIR( const std::string& listmode_filename, const std::string& crystal_map_filename, const std::string& template_proj_data_filename);
	
	virtual std::string get_name() const;
	virtual shared_ptr <CListRecord> get_empty_record_sptr() const;
	virtual Succeeded get_next_record(CListRecord& record_of_general_type) const;
	virtual Succeeded reset();
	
	/*!
	This function should save the position in input file. This is not implemented but disabled.
	Returns 0 in the moement.
	\todo Maybe provide real implementation?
	*/
	virtual SavedPosition save_get_position()
	{ return static_cast<SavedPosition>(current_lm_data_ptr->save_get_position()); }
	virtual Succeeded set_get_position(const SavedPosition& pos)
	{ return current_lm_data_ptr->set_get_position(pos); }

	/*! 
	Returns just false in the moment.
	\todo Implement this properly to check for delayed events in LM files.
	*/
	virtual bool has_delayeds() const { return false; }

private:
	std::string listmode_filename;
	mutable shared_ptr<InputStreamWithRecords<CListRecordT, bool> > current_lm_data_ptr;
	mutable std::vector< unsigned int> saved_get_positions;
	Succeeded open_lm_file() const;
	shared_ptr<DetectorCoordinateMapFromFile> map;
};
	
END_NAMESPACE_STIR

#endif
