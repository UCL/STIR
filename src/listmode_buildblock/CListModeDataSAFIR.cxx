/* CListModeDataSAFIR.cxx

Coincidence LM Data Class for SAFIR: Implementation
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

#include <iostream>
#include <fstream>
#include <typeinfo>

#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"

#include "boost/static_assert.hpp"
#include "boost/pointer_cast.hpp"

#include "stir/listmode/CListModeDataSAFIR.h"


//#include "stir/ExamInfo.h"


#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ios;
using std::fstream;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR;


template <class CListRecordT>
CListModeDataSAFIR<CListRecordT>::
CListModeDataSAFIR(const std::string& listmode_filename, const std::string& crystal_map_filename, const std::string& template_proj_data_filename)
  : listmode_filename(listmode_filename), map(boost::make_shared<DetectorCoordinateMapFromFile>(crystal_map_filename))
{
	this->exam_info_sptr.reset(new ExamInfo);

	// Here we are reading the scanner data from the template projdata
	shared_ptr<ProjData> template_proj_data_sptr =
			ProjData::read_from_file(template_proj_data_filename);
	scanner_sptr.reset( new Scanner(*template_proj_data_sptr->get_proj_data_info_ptr()->get_scanner_ptr()));

	if( open_lm_file() == Succeeded::no )
	{
		error("CListModeDataSAFIR: Could not open listmode file " +listmode_filename + "\n");
	}
}


template <class CListRecordT>
std::string
CListModeDataSAFIR<CListRecordT>::
get_name() const
{
	return listmode_filename;
}

template <class CListRecordT>
shared_ptr <CListRecord>
CListModeDataSAFIR<CListRecordT>::
get_empty_record_sptr() const
{
	boost::shared_ptr<CListRecordSAFIR> sptr(new CListRecordSAFIR);
	sptr->event_SAFIR().set_map(map);
	return boost::static_pointer_cast<CListRecord>(sptr);
}

template <class CListRecordT>
Succeeded
CListModeDataSAFIR<CListRecordT>::
get_next_record(CListRecord& record_of_general_type) const
{
	CListRecordT& record = static_cast<CListRecordT&>(record_of_general_type);
	Succeeded status = current_lm_data_ptr->get_next_record(record);
	if( status == Succeeded::yes ) record.event_SAFIR().set_map(map);
	return status;
	
}

template <class CListRecordT>
Succeeded
CListModeDataSAFIR<CListRecordT>::
reset()
{
	return current_lm_data_ptr->reset();
}

template <class CListRecordT>
Succeeded
CListModeDataSAFIR<CListRecordT>::
open_lm_file() const
{
	cerr << "CListModeDataSAFIR: opening file " << listmode_filename << endl;
	shared_ptr<istream> stream_ptr(new fstream(listmode_filename.c_str(), ios::in | ios::binary ));
	if(!(*stream_ptr))
	{
		warning("CListModeDataSAFIR: cannot open file " + listmode_filename + "\n");
		return Succeeded::no;
	}
	stream_ptr->seekg((std::streamoff)32);
	current_lm_data_ptr.reset(
			new InputStreamWithRecords<CListRecordT, bool>
			( stream_ptr, sizeof(CListTimeDataSAFIR),
					sizeof(CListTimeDataSAFIR),
					ByteOrder::little_endian !=ByteOrder::get_native_order()));
	return Succeeded::yes;
}

template <class CListRecordT>
shared_ptr<ProjDataInfo>
CListModeDataSAFIR<CListRecordT>::
get_proj_data_info_sptr() const
{
    assert(!is_null_ptr(proj_data_info_sptr));
    return proj_data_info_sptr;
}
	
template class CListModeDataSAFIR<CListRecordSAFIR>;

END_NAMESPACE_STIR

