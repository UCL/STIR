/* CListModeDataSAFIR.cxx

Coincidence LM Data Class for SAFIR: Implementation

	Copyright 2015 ETH Zurich, Institute of Particle Physics
	Copyright 2020 Positrigo AG, Zurich
    Copyright 2021 University College London

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
  \brief implementation of class stir::CListModeDataSAFIR

  \author Jannis Fischer
  \author Kris Thielemans
*/
#include <iostream>
#include <fstream>
#include <typeinfo>

#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"

//#include "boost/static_assert.hpp"

#include "stir/listmode/CListModeDataSAFIR.h"

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
CListModeDataSAFIR<CListRecordT>::CListModeDataSAFIR(const std::string& listmode_filename,
                                                     const std::string& crystal_map_filename,
                                                     const std::string& template_proj_data_filename,
                                                     const double lor_randomization_sigma)
    : listmode_filename(listmode_filename)
{
  if (!crystal_map_filename.empty())
    {
      map = MAKE_SHARED<DetectorCoordinateMap>(crystal_map_filename, lor_randomization_sigma);
    }
  else
    {
      if (lor_randomization_sigma != 0)
        error("SAFIR currently does not support LOR-randomisation unless a map is specified");
    }
  this->exam_info_sptr.reset(new ExamInfo);

  // Here we are reading the scanner data from the template projdata
  shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(template_proj_data_filename);
  this->set_proj_data_info_sptr(template_proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone());

  if (open_lm_file() == Succeeded::no)
    {
      error("CListModeDataSAFIR: Could not open listmode file " + listmode_filename + "\n");
    }
}

template <class CListRecordT>
CListModeDataSAFIR<CListRecordT>::CListModeDataSAFIR(const std::string& listmode_filename,
                                                     const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
    : listmode_filename(listmode_filename)
{
  this->exam_info_sptr.reset(new ExamInfo);
  this->set_proj_data_info_sptr(proj_data_info_sptr->create_shared_clone());

  if (open_lm_file() == Succeeded::no)
    {
      error("CListModeDataSAFIR: Could not open listmode file " + listmode_filename + "\n");
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
  shared_ptr<CListRecordSAFIR> sptr(new CListRecordSAFIR);
  sptr->event_SAFIR().set_scanner_sptr(this->get_proj_data_info_sptr()->get_scanner_sptr());
  sptr->event_SAFIR().set_map_sptr(map);
  return static_pointer_cast<CListRecord>(sptr);
}

template <class CListRecordT>
Succeeded
CListModeDataSAFIR<CListRecordT>::
get_next_record(CListRecord& record_of_general_type) const
{
	CListRecordT& record = static_cast<CListRecordT&>(record_of_general_type);
	Succeeded status = current_lm_data_ptr->get_next_record(record);
        // if( status == Succeeded::yes ) record.event_SAFIR().set_map_sptr(map);
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
	
template class CListModeDataSAFIR<CListRecordSAFIR>;

END_NAMESPACE_STIR

