/* CListRecordSAFIR.h

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
  \brief Declaration of class stir::CListEventSAFIR and stir::CListRecordSAFIR with supporting classes

  \author Jannis Fischer
*/

#ifndef __stir_listmode_CListRecordSAFIR_H__
#define __stir_listmode_CListRecordSAFIR_H__

#include "stir/listmode/CListRecord.h"
#include "stir/DetectionPositionPair.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"

#include "boost/make_shared.hpp"
#include "boost/static_assert.hpp"
#include "boost/cstdint.hpp"

#include "stir/listmode/DetectorCoordinateMapFromFile.h"


START_NAMESPACE_STIR

/*!
Provides interface of the record class to STIR by implementing get_LOR(). It uses a map from detector indices to coordinates to specify LORAs2Points from given detection pair indices.

The record has the following format (for little-endian byte order)
\code
	unsigned ringA : 8;
	unsigned ringB : 8;
	unsigned detA : 16;
	unsigned detB : 16;
	unsigned layerA : 4;
	unsigned layerB : 4;
	unsigned reserved : 6;
	unsigned isRandom : 1;
	unsigned type : 1;
\endcode
*/
template <class Derived>
class CListEventSAFIR : public CListEvent
{
public:
	/*! Constructor which initializes map upon construction.
	*/
	inline CListEventSAFIR( shared_ptr<DetectorCoordinateMapFromFile> map ) : map(map) {}
	//! Returns LOR corresponding to the given event.
	inline virtual LORAs2Points<float> get_LOR() const;

	//! Returns 0 if event is prompt and 1 if random/delayed
	inline bool is_prompt()
		const { return !(static_cast<const Derived*>(this)->is_prompt()); }
	//! Function to set map for detector indices to coordinates.
	inline void set_map( shared_ptr<DetectorCoordinateMapFromFile> new_map ) { map = new_map; }

private:
	friend class CListRecordSAFIR;
	/*! Default constructor will not work as it does not initialize a map to relate
	detector indices and space coordinates. Always use other constructor with a map pointer. Or use set_map( shared_ptr<DetectorCoordinateMapFromFile> new_map ) after default construction.
	*/
	inline CListEventSAFIR( ) {}
	shared_ptr<DetectorCoordinateMapFromFile> map;
};



//! Class for record with coincidence data
class CListEventDataSAFIR
{
public:
	//! Writes detection position pair to reference given as argument.
	inline void get_detection_position_pair(DetectionPositionPair<>& det_pos_pair);

	//! Returns 0 if event is prompt and 1 if random/delayed
	inline bool is_prompt()
		const { return !isRandom; }

	//! Can be used to set "promptness" of event.
	inline Succeeded set_prompt( const bool prompt = true ) { 
		isRandom = !prompt;
		return Succeeded::yes; 
	}


private:
	friend class CListRecordSAFIR;

#if STIRIsNativeByteOrderBigEndian
	unsigned type : 1;
	unsigned isRandom : 1;
	unsigned reserved : 6;
	unsigned layerB : 4;
	unsigned layerA : 4;
	unsigned detB : 16;
	unsigned detA : 16;
	unsigned ringB : 8;
	unsigned ringA : 8;
#else
	unsigned ringA : 8;
	unsigned ringB : 8;
	unsigned detA : 16;
	unsigned detB : 16;
	unsigned layerA : 4;
	unsigned layerB : 4;
	unsigned reserved : 6;
	unsigned isRandom : 1;
	unsigned type : 1;
#endif
};


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

//! Class for general record, containing a union of data, time and raw record and providing access to certain elements.
class CListRecordSAFIR : public CListRecord, public CListTime, public CListEventSAFIR<CListRecordSAFIR>
{
public:
	typedef CListEventDataSAFIR DataType;
	
	//! Returns event_data (without checking if the type is really event and not time).
	DataType get_data() const
	{ return this->event_data; }

	CListRecordSAFIR() : CListEventSAFIR<CListRecordSAFIR>() {}

	virtual ~CListRecordSAFIR() {}

	virtual bool is_time() const
	{ return time_data.type == 1; }

	virtual bool is_event() const
	{ return time_data.type == 0; }

	virtual CListEvent&  event()
	{ return *this; }

	virtual const CListEvent&  event() const
	{ return *this; }

	virtual CListEventSAFIR<CListRecordSAFIR>&  event_SAFIR()
	{ return *this; }

	virtual const CListEventSAFIR<CListRecordSAFIR>&  event_SAFIR() const
	{ return *this; }

	virtual CListTime&   time()
	{ return *this; }

	virtual const CListTime&   time() const
	{ return *this; }

	virtual bool operator==(const CListRecord& e2) const
	{
		return dynamic_cast<CListRecordSAFIR const *>(&e2) != 0 &&
				raw == static_cast<CListRecordSAFIR const &>(e2).raw;
	}

	inline unsigned long get_time_in_millisecs() const
	{ return time_data.get_time_in_millisecs(); }

	inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
	{ return time_data.set_time_in_millisecs(time_in_millisecs); }

	inline bool is_prompt() const { return !(event_data.isRandom); }

	Succeeded
	init_from_data_ptr(const char * const data_ptr,
	                     const std::size_t size_of_record,
	                     const bool do_byte_swap)
	{
		assert(size_of_record >= 8);
		std::copy(data_ptr, data_ptr+8, reinterpret_cast<char *>(&raw));// TODO necessary for operator==
		if (do_byte_swap) ByteOrder::swap_order(raw);
		return Succeeded::yes;
	}


	std::size_t
	size_of_record_at_ptr(const char * const /*data_ptr*/, const std::size_t /*size*/,
	                        const bool /*do_byte_swap*/) const
	{ return 8; }

private:
// use C++ union to save data, you can only use one at a time,
// but compiler will not check which one was used!
// Be careful not to read event data from time record and vice versa!!
// However, this is used as a feature if comparing events over the 'raw' type.
	union {
		CListEventDataSAFIR event_data;
		CListTimeDataSAFIR time_data;
	    boost::int64_t raw;
	};
	BOOST_STATIC_ASSERT(sizeof(boost::uint64_t)==8);
	BOOST_STATIC_ASSERT(sizeof(CListEventDataSAFIR)==8);
	BOOST_STATIC_ASSERT(sizeof(CListTimeDataSAFIR)==8);
};


END_NAMESPACE_STIR

#include "CListRecordSAFIR.inl"

#endif
