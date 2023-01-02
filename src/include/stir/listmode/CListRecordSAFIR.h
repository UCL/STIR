/* CListRecordSAFIR.h

 Coincidence Event Class for SAFIR: Header File

	Copyright 2015 ETH Zurich, Institute of Particle Physics
	Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
	Copyright 2020, 2022 Positrigo AG, Zurich

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
  \author Parisa Khateri
  \author Markus Jehl
*/

#ifndef __stir_listmode_CListRecordSAFIR_H__
#define __stir_listmode_CListRecordSAFIR_H__

#include<random>

#include "stir/listmode/CListRecord.h"
#include "stir/DetectionPositionPair.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"

#include "boost/static_assert.hpp"
#include "boost/cstdint.hpp"

#include "stir/DetectorCoordinateMap.h"
#include "boost/make_shared.hpp"

START_NAMESPACE_STIR

/*!
Provides interface of the record class to STIR by implementing get_LOR(). It uses an optional map from detector indices to coordinates to specify LORAs2Points from given detection pair indices.

The record has the following format (for little-endian byte order)
\code
	unsigned ringA : 8;
	unsigned ringB : 8;
	unsigned detA : 16;
	unsigned detB : 16;
	unsigned layerA : 4;
	unsigned layerB : 4;
	unsigned reserved : 6;
	unsigned isDelayed : 1;
	unsigned type : 1;
\endcode
  \ingroup listmode
*/
template <class Derived>
class CListEventSAFIR : public CListEvent
{
public:
	/*! Default constructor will not work as it does not initialize a map to relate
	detector indices and space coordinates. Always use either set_scanner_sptr or set_map_sptr after default construction.
	*/
	inline CListEventSAFIR( ) {}

	//! Returns LOR corresponding to the given event.
	inline virtual LORAs2Points<float> get_LOR() const;
	
  //! Override the default implementation
  inline virtual void get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const;
  
  //! This method checks if the template is valid for LmToProjData
  /*! Used before the actual processing of the data (see issue #61), before calling get_bin()
   *  Most scanners have listmode data that correspond to non arc-corrected data and
   *  this check avoids a crash when an unsupported template is used as input.
   */
	inline virtual bool is_valid_template(const ProjDataInfo&) const {return true;}

	//! Returns 0 if event is prompt and 1 if delayed
	inline bool is_prompt()
		const { return !(static_cast<const Derived*>(this)->is_prompt()); }
	//! Function to set map for detector indices to coordinates.
	/*! Use a null pointer to disable the mapping functionality */
	inline void set_map_sptr( shared_ptr<const DetectorCoordinateMap> new_map_sptr ) { map_sptr = new_map_sptr; }
    /*! Set the scanner */
    /*! Currently only used if the map is not set. */
	inline void set_scanner_sptr( shared_ptr<const Scanner> new_scanner_sptr ) { scanner_sptr = new_scanner_sptr; }

private:
	shared_ptr<const DetectorCoordinateMap> map_sptr;
	shared_ptr<const Scanner> scanner_sptr;

	const DetectorCoordinateMap& map_to_use() const
	{ return  map_sptr ? *map_sptr : *this->scanner_sptr->get_detector_map_sptr(); }
};



//! Class for record with coincidence data using SAFIR bitfield definition
/*! \ingroup listmode */
class CListEventDataSAFIR
{
public:
	//! Writes detection position pair to reference given as argument.
	inline void get_detection_position_pair(DetectionPositionPair<>& det_pos_pair);

	//! Returns 0 if event is prompt and 1 if delayed
	inline bool is_prompt()
		const { return !isDelayed; }

	//! Returns 1 if if event is time and 0 if it is prompt
	inline bool is_time() const { return type; }

	//! Can be used to set "promptness" of event.
	inline Succeeded set_prompt( const bool prompt = true ) { 
		isDelayed = !prompt;
		return Succeeded::yes; 
	}


private:

#if STIRIsNativeByteOrderBigEndian
	unsigned type : 1;
	unsigned isDelayed : 1;
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
	unsigned isDelayed : 1;
	unsigned type : 1;
#endif
};


//! Class for record with coincidence data using NeuroLF bitfield definition
/*! \ingroup listmode */
class CListEventDataNeuroLF
{
public:
	//! Writes detection position pair to reference given as argument.
	inline void get_detection_position_pair(DetectionPositionPair<>& det_pos_pair);

	//! Returns 0 if event is prompt and 1 if delayed
	inline bool is_prompt()
		const { return !isDelayed; }

	//! Returns 1 if if event is time and 0 if it is prompt
	inline bool is_time() const { return type; }

	//! Can be used to set "promptness" of event.
	inline Succeeded set_prompt( const bool prompt = true ) { 
		isDelayed = !prompt;
		return Succeeded::yes; 
	}


private:

#if STIRIsNativeByteOrderBigEndian
	unsigned type : 1;
	unsigned isDelayed : 1;
	unsigned reserved : 8;
	unsigned layerB : 3;
	unsigned layerA : 3;
	unsigned detB : 16;
	unsigned detA : 16;
	unsigned ringB : 8;
	unsigned ringA : 8;
#else
	unsigned ringA : 8;
	unsigned ringB : 8;
	unsigned detA : 16;
	unsigned detB : 16;
	unsigned layerA : 3;
	unsigned layerB : 3;
	unsigned reserved : 8;
	unsigned isDelayed : 1;
	unsigned type : 1;
#endif
};


//! Class for record with time data using SAFIR bitfield definition
/*! \ingroup listmode */
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
	inline bool is_time() const { return type; }
private:
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

//! Class for general SAFIR record, containing a union of data, time and raw record and providing access to certain elements.
/*! \ingroup listmode */
template <class DataType>
class CListRecordSAFIR : public CListRecord, public ListTime, public CListEventSAFIR<CListRecordSAFIR<DataType>>
{
public:

	//! Returns event_data (without checking if the type is really event and not time).
	DataType get_data() const
	{ return this->event_data; }

	CListRecordSAFIR() : CListEventSAFIR<CListRecordSAFIR<DataType>>() {}

	virtual ~CListRecordSAFIR() {}

	virtual bool is_time() const
	{ return time_data.is_time(); }

	virtual bool is_event() const
	{ return !time_data.is_time(); }

	virtual CListEvent&  event()
	{ return *this; }

	virtual const CListEvent&  event() const
	{ return *this; }

	virtual CListEventSAFIR<CListRecordSAFIR<DataType>>&  event_SAFIR()
	{ return *this; }

	virtual const CListEventSAFIR<CListRecordSAFIR<DataType>>&  event_SAFIR() const
	{ return *this; }

    virtual ListTime&   time()
	{ return *this; }

    virtual const ListTime&   time() const
	{ return *this; }

	virtual bool operator==(const CListRecord& e2) const
	{
		return dynamic_cast<CListRecordSAFIR<DataType> const *>(&e2) != 0 &&
				raw == static_cast<CListRecordSAFIR<DataType> const &>(e2).raw;
	}

	inline unsigned long get_time_in_millisecs() const
	{ return time_data.get_time_in_millisecs(); }

	inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
	{ return time_data.set_time_in_millisecs(time_in_millisecs); }

	inline bool is_prompt() const { return event_data.is_prompt(); }

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
		DataType event_data;
		CListTimeDataSAFIR time_data;
	    boost::int64_t raw;
	};
	BOOST_STATIC_ASSERT(sizeof(boost::uint64_t)==8);
	BOOST_STATIC_ASSERT(sizeof(DataType)==8);
	BOOST_STATIC_ASSERT(sizeof(CListTimeDataSAFIR)==8);
};


END_NAMESPACE_STIR

#include "CListRecordSAFIR.inl"

#endif
