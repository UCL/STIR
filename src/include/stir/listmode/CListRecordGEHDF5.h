/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd (CListRecordECAT.h)
    Copyright (C) 2013, 2016, 2020 University College London
    Copyright (C) 2017-2018 University of Leeds
*/
/*!
  \file
  \ingroup listmode
  \ingroup GE
  \brief Classes for listmode records of GE RDF9 data

  \author Kris Thielemans (major mods for GE Dimension data)
  \author Ottavia Bertolli (major mods for GE Signa (i.e. RDF 9) data)
  \author Palak Wadhwa (fix to STIR conventions and checks)
  \author Ander Biguri (generalise from Signa to RDF9)
*/

#ifndef __stir_listmode_CListRecordGEHDF5_H__
#define __stir_listmode_CListRecordGEHDF5_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"
#include <boost/static_assert.hpp>
#include <boost/cstdint.hpp>
#include <iostream>

START_NAMESPACE_STIR

namespace GE {
namespace RDF_HDF5 {

  namespace detail {
    /***********************************
     * Supported Event Length Modes
     ***********************************/
    enum EventLength
    {
      /* RESERVED         = 0x0, */
      LENGTH_6_EVT        = 0x1,
      LENGTH_8_EVT        = 0x2,
      LENGTH_16_EVT       = 0x3
    };

    /***********************************
     * Supported Event Types
     ***********************************/
    enum EventType
    {
      EXTENDED_EVT    = 0x0,
      COINC_EVT       = 0x1
    };

    /***********************************
     * Supported Extended Event Types
     ***********************************/
    enum ExtendedEvtType
    {
      TIME_MARKER_EVT = 0x0,
      COINC_COUNT_EVT = 0x1,
      EXTERN_TRIG_EVT = 0x2,
      TABLE_POS_EVT   = 0x3,
      /* RESERVED     = 0x4 to 0xE */
      /* 0xE is temporary taken here to mark end of it. */
      END_LIST_EVT = 0xE,
      SINGLE_EVT      = 0xF
    };

    //! Class for finding out what the event/size-type is in a GE RDF9 listmode file
    /*! \ingroup listmode
      \ingroup GE
    */
    class CListAnyRecordDataGEHDF5
    {
    public:
#if STIRIsNativeByteOrderBigEndian
      // Do byteswapping first before using this bit field.
      TODO;
#else
      boost::uint16_t eventLength:2;       /* Event Length : Enum for the number of bytes in the event */
      boost::uint16_t eventType:1;         /* Event Type : Coin or Extended types */
      boost::uint16_t eventTypeExt:4;      /*  If not a coincidence, Extended Event Type : Time Marker, Trigger, Single..etc */
      boost::uint16_t dummy:9;
#endif
    }; /*any record */

    //! Class for storing and using a coincidence event from a GE RDF9 listmode file
    /*! \ingroup listmode
      \ingroup GE
      This class cannot have virtual functions, as it needs to just store the data 6 bytes for CListRecordGEHDF5 to work.
    */
    class CListEventDataGEHDF5
    {
    public:
      inline bool is_prompt() const { return true; } // TODO
      inline Succeeded set_prompt(const bool prompt = true) 
      { 
        //if (prompt) random=1; else random=0; return Succeeded::yes; 
        return Succeeded::no;
      }
      inline bool is_event() const
      { 
        return (eventType==COINC_EVT)/* && eventTypeExt==COINC_COUNT_EVT)*/; 
      } // TODO need to find out how to see if it's a coincidence event

#if STIRIsNativeByteOrderBigEndian
      // Do byteswapping first before using this bit field.
      TODO
#else
        boost::uint16_t eventLength:2;       /* Event Length : Enum for the number of bytes in the event */
      boost::uint16_t eventType:1;         /* Event Type : Coin or Extended types */
      boost::uint16_t hiXtalShortInteg:1;  /* High Crystal Short Integration on / off */
      boost::uint16_t loXtalShortInteg:1;  /* Low Crystal Short Integration on / off */
      boost::uint16_t hiXtalScatterRec:1;  /* High Crystal Scatter Recovered on / off */
      boost::uint16_t loXtalScatterRec:1;  /* Low Crystal Scatter Recovered on / off */
      boost::int16_t  deltaTime:9;         /* TOF 'signed' delta time (units defined by electronics */
      boost::uint16_t hiXtalAxialID:6;     /* High Crystal Axial Id */
      boost::uint16_t hiXtalTransAxID:10;  /* High Crystal Trans-Axial Id */
      boost::uint16_t loXtalAxialID:6;     /* Low Crystal Axial Id */
      boost::uint16_t loXtalTransAxID:10;  /* Low Crystal Trans-Axial Id */
#endif
    }; /*-coincidence event*/


    //! A class for storing and using a timing 'event' from a GE RDF9 listmode file
    /*! \ingroup listmode
      \ingroup GE
      This class cannot have virtual functions, as it needs to just store the data 6 bytes for CListRecordGEHDF5 to work.
    */
    class ListTimeDataGEHDF5
    {
    public:
      inline unsigned long get_time_in_millisecs() const
      { return (time_hi()<<16) | time_lo(); }
      inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
      { 
        data.timeMarkerLS = ((1UL<<16)-1) & (time_in_millisecs); 
        data.timeMarkerMS = (time_in_millisecs) >> 16; 
        // TODO return more useful value
        return Succeeded::yes;
      }
      inline bool is_time() const
      { // TODO need to find out how to see if it's a timing event
	return (data.eventType==EXTENDED_EVT) && (data.eventTypeExt==TIME_MARKER_EVT); 
      }// TODO
      
    private:
      typedef union{
        struct {
#if STIRIsNativeByteOrderBigEndian
          TODO
#else
          boost::uint16_t eventLength:2;     /* Event Length : Enum for the number of bytes in the event */
          boost::uint16_t eventType:1;       /* Event Type : Coin or Extended types */
          boost::uint16_t eventTypeExt:4;    /* Extended Event Type : Time Marker, Trigger, Single..etc */
          boost::uint16_t unused1:5;         /* Unused */
          boost::uint16_t externEvt3:1;	    /* External Event Input 3 Level */
          boost::uint16_t externEvt2:1;	    /* External Event Input 2 Level */
          boost::uint16_t externEvt1:1;	    /* External Event Input 1 Level */
          boost::uint16_t externEvt0:1;	    /* External Event Input 0 Level */
          boost::uint16_t timeMarkerLS:16;   /* Least Significant 16 bits of 32-bit Time Marker */
          boost::uint16_t timeMarkerMS:16;   /* Most Significant 16 bits of 32-bitTime Marker */
#endif
        };      
      } data_t;
      data_t data;

      unsigned long time_lo() const
      { return data.timeMarkerLS; }
      unsigned long time_hi() const
      { return data.timeMarkerMS; }
    };

  }

//! A class for a general element (or "record") of a GE RDF9 listmode file
/*! \ingroup listmode
  \ingroup GE
  All types of records are stored in a (private) union with the "basic" classes such as CListEventDataGEHDF5.
  This class essentially just forwards the work to the "basic" classes.
*/
class CListRecordGEHDF5 : public CListRecord, public ListTime, // public CListGatingInput,
    public  CListEventCylindricalScannerWithDiscreteDetectors
{
  typedef detail::CListEventDataGEHDF5 DataType;
  typedef detail::ListTimeDataGEHDF5 TimeType;
  //typedef CListGatingDataGEHDF5 GatingType;

 public:
  //! constructor
  /*! Takes the scanner and first_time stamp. The former will be used for checking and swapping,
    the latter for adjusting the time of each event, as GE listmode files do not start with time-stamp 0.

    get_time_in_millisecs() should therefore be zero at the first time stamp.
  */
 CListRecordGEHDF5(const shared_ptr<Scanner>& scanner_sptr, const unsigned long first_time_stamp) :
  CListEventCylindricalScannerWithDiscreteDetectors(scanner_sptr),
    first_time_stamp(first_time_stamp)
    {}

  bool is_time() const
  { 
    return this->time_data.is_time();
  }
#if 0
  bool is_gating_input() const
  {
    return this->gating_data.is_gating_input();
  }
#endif

  bool is_event() const
  { return this->event_data.is_event(); }
  virtual CListEvent&  event() 
    { return *this; }
  virtual const CListEvent&  event() const
    { return *this; }
  virtual ListTime&   time()
    { return *this; }
  virtual const ListTime&   time() const
    { return *this; }
#if 0
  virtual CListGatingInput&  gating_input()
    { return *this; }
  virtual const CListGatingInput&  gating_input() const
  { return *this; }
#endif
  bool operator==(const CListRecord& e2) const
  {
    return false;
#if 0
// TODO
dynamic_cast<CListRecordGEHDF5 const *>(&e2) != 0 &&
      raw[0] == static_cast<CListRecordGEHDF5 const &>(e2).raw[0] &&
      (this->is_event() || (raw[1] == static_cast<CListRecordGEHDF5 const &>(e2).raw[1]));
#endif
  }	    

  // time 
  inline unsigned long get_time_in_millisecs() const 
    { return time_data.get_time_in_millisecs() - first_time_stamp; }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
    { return time_data.set_time_in_millisecs(time_in_millisecs); }
#if 0
  inline unsigned int get_gating() const
    { return gating_data.get_gating(); }
  inline Succeeded set_gating(unsigned int g) 
    { return gating_data.set_gating(g); }
#endif
  // event
  inline bool is_prompt() const { return event_data.is_prompt(); }
  inline Succeeded set_prompt(const bool prompt = true) 
  { return event_data.set_prompt(prompt); }

  virtual void get_detection_position(DetectionPositionPair<>& det_pos) const
  {
    det_pos.pos1().tangential_coord() = scanner_sptr->get_num_detectors_per_ring() - 1 - event_data.loXtalTransAxID;
    det_pos.pos1().axial_coord() = event_data.loXtalAxialID;
    det_pos.pos2().tangential_coord() = scanner_sptr->get_num_detectors_per_ring() - 1 - event_data.hiXtalTransAxID;
    det_pos.pos2().axial_coord() = event_data.hiXtalAxialID;
  }

  //! This routine sets in a coincidence event from detector "indices"
  virtual void set_detection_position(const DetectionPositionPair<>&)
  {
    error("TODO");
  }

  virtual std::size_t size_of_record_at_ptr(const char * const data_ptr, const std::size_t, 
                                            const bool do_byte_swap) const
  { 
    // TODO don't know what to do with byteswap.
    assert(do_byte_swap == false);

    // Figure out the actual size from the eventLength bits.
    union
    {
      detail::CListAnyRecordDataGEHDF5 rec;
      boost::uint16_t raw;
    };
    std::copy(data_ptr, data_ptr+2, &raw);
    switch(rec.eventLength)
      {
      case detail::LENGTH_6_EVT: return std::size_t(6);
      case detail::LENGTH_8_EVT: return std::size_t(8);
      case detail::LENGTH_16_EVT: return std::size_t(16);
      default:
        error("ClistRecordGEHDF5: error decoding event (eventLength bits are incorrect)");
        return std::size_t(0); // avoid compiler warnings
      }
  }

  virtual Succeeded init_from_data_ptr(const char * const data_ptr, 
                                       const std::size_t size,
                                       const bool do_byte_swap)
  {
    assert(size >= 6);
    assert(size <= 16);
    std::copy(data_ptr, data_ptr+size, reinterpret_cast<char *>(&this->raw[0]));

    if (do_byte_swap)
      {
        error("ClistRecordGEHDF5: byte-swapping not supported yet. sorry");
        //ByteOrder::swap_order(this->raw[0]);
      }
    return Succeeded::yes;
  }

private:
  unsigned long first_time_stamp;
  union {
    DataType  event_data;
    TimeType   time_data; 
    //GatingType gating_data;
    boost::int32_t  raw[16/4];
  };
  BOOST_STATIC_ASSERT(sizeof(boost::int32_t)==4);
  BOOST_STATIC_ASSERT(sizeof(DataType)==6);
  BOOST_STATIC_ASSERT(sizeof(TimeType)==6); 
  //BOOST_STATIC_ASSERT(sizeof(GatingType)==8); 

};


} // namespace
} // namespace

END_NAMESPACE_STIR

#endif
