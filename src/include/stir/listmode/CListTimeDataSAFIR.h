/* CListTimeDataSAFIR.h

 Coincidence Event Class for SAFIR: Header File
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
  \brief Declaration of class stir::CListEventSAFIR and stir::CListTimeDataSAFIR supporting class

  \author Jannis Fischer
*/

#ifndef __stir_listmode_CListTimeDataSAFIR_H__
#define __stir_listmode_CListTimeDataSAFIR_H__

#include "stir/listmode/CListEventSAFIR.h"
#include "stir/listmode/CListEventDataSAFIR.h"

#include "stir/listmode/CListRecord.h"
#include "stir/DetectionPositionPair.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"

#include "boost/static_assert.hpp"
#include "boost/cstdint.hpp"


START_NAMESPACE_STIR

//! Class for record with time data
class CListTimeDataSAFIR
{
public:
	inline unsigned long get_time_in_millisecs() const
	{ return static_cast<unsigned long>(time);  }
	inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
	{
		time = ((boost::uint64_t(1)<<49)-1) & static_cast<boost::uint64_t>(time_in_millisecs);
		return Succeeded::yes;
	}
private:
	friend class CListRecordSAFIR;
#if STIRIsNativeByteOrderBigEndian
	boost::uint64_t type : 1;
	boost::uint64_t reserved : 15;
	boost::uint64_t time : 48;
#else
	boost::uint64_t time : 48;
	boost::uint64_t reserved : 15;
	boost::uint64_t type : 1;
#endif
}; 

END_NAMESPACE_STIR

#endif
