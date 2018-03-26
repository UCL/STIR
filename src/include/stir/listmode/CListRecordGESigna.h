/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd (CListRecordECAT.h)
    Copyright (C) 2013 University College London (major mods for GE Dimension data)
*/
/*!
  \file
  \ingroup listmode
  \brief Classes for listmode records of GE Dimension console data

  This file is based on GE proprietary information and can therefore not be distributed outside UCL
  without approval from GE.
    
  \author Kris Thielemans
*/

#ifndef __stir_listmode_CListRecordGESigna_H__
#define __stir_listmode_CListRecordGESigna_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"
#include <boost/static_assert.hpp>
#include <boost/cstdint.hpp>
#include <iostream>

START_NAMESPACE_STIR

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


//! Class for storing and using a coincidence event from a GE Dimension listmode file
/*! \ingroup listmode
  This class cannot have virtual functions, as it needs to just store the data 4 bytes for CListRecordGESigna to work.
*/
class CListEventDataGESigna
{
 public:  
  inline bool is_prompt() const { return true; } // TODO
  inline Succeeded set_prompt(const bool prompt = true) 
  { 
    //if (prompt) random=1; else random=0; return Succeeded::yes; 
    return Succeeded::no;
  }
  inline void get_detection_position(DetectionPositionPair<>& det_pos) const
  {
    det_pos.pos1().tangential_coord() = loXtalTransAxID;
    det_pos.pos1().axial_coord() = loXtalAxialID;
    det_pos.pos2().tangential_coord() = hiXtalTransAxID;
    det_pos.pos2().axial_coord() = hiXtalAxialID;
  }
  inline bool is_event() const
    { 
      return (eventType==COINC_EVT)/* && eventTypeExt==COINC_COUNT_EVT)*/; 
     } // TODO need to find out how to see if it's a coincidence event

 private:

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


//! A class for storing and using a timing 'event' from a GE Signa PET/MR listmode file
/*! \ingroup listmode
  This class cannot have virtual functions, as it needs to just store the data 8 bytes for CListRecordGESigna to work.
 */
class CListTimeDataGESigna
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

#if 0
//! A class for storing and using a trigger 'event' from a GE Dimension listmode file
/*! \ingroup listmode
  This class cannot have virtual functions, as it needs to just store the data 8 bytes for CListRecordGESigna to work.
 */
class CListGatingDataGESigna
{
 public:
  #if 0
  inline unsigned long get_time_in_millisecs() const
    { return (time_hi()<<24) | time_lo(); }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
    { 
      words[0].value = ((1UL<<24)-1) & (time_in_millisecs); 
      words[1].value = (time_in_millisecs) >> 24; 
      // TODO return more useful value
      return Succeeded::yes;
    }
  #endif
  inline bool is_gating_input() const
    { return (words[0].signature==21) && (words[1].signature==29); }
  inline unsigned int get_gating() const
    { return words[0].reserved; } // return "reserved" bits. might be something in there
  inline Succeeded set_gating(unsigned int g) 
    { words[0].reserved = g&7; return Succeeded::yes; }
      
private:
  typedef union{
    struct {
#if STIRIsNativeByteOrderBigEndian
      boost::uint32_t signature : 5;
      boost::uint32_t reserved : 3;
      boost::uint32_t value : 24; // timing info here in the first word, but we're ignoring it
#else
      boost::uint32_t value : 24;
      boost::uint32_t reserved : 3;
      boost::uint32_t signature : 5;
#endif
    };      
    boost::uint32_t raw;
  } oneword_t;
  oneword_t words[2];
};

#endif

//! A class for a general element (or "record") of a GE Dimension listmode file
/*! \ingroup listmode
  All types of records are stored in a (private) union with the "basic" classes such as CListEventDataGESigna.
  This class essentially just forwards the work to the "basic" classes.

  A complication for GE Dimension data is that not all events are the same size:
  coincidence events are 4 bytes, and others are 8 bytes. 

  \todo Currently we always assume the data is from a DSTE. We should really read this from the RDF header.
*/
class CListRecordGESigna : public CListRecord, public CListTime, // public CListGatingInput,
    public  CListEventCylindricalScannerWithDiscreteDetectors
{
  typedef CListEventDataGESigna DataType;
  typedef CListTimeDataGESigna TimeType;
  //typedef CListGatingDataGESigna GatingType;

 public:  
  CListRecordGESigna() :
  CListEventCylindricalScannerWithDiscreteDetectors(shared_ptr<Scanner>(new Scanner(Scanner::PETMR_Signa)))
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
  virtual CListTime&   time()
    { return *this; }
  virtual const CListTime&   time() const
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
dynamic_cast<CListRecordGESigna const *>(&e2) != 0 &&
      raw[0] == static_cast<CListRecordGESigna const &>(e2).raw[0] &&
      (this->is_event() || (raw[1] == static_cast<CListRecordGESigna const &>(e2).raw[1]));
#endif
  }	    

  // time 
  inline unsigned long get_time_in_millisecs() const 
    { return time_data.get_time_in_millisecs(); }
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
  { event_data.get_detection_position(det_pos); }

  //! This routine sets in a coincidence event from detector "indices"
  virtual void set_detection_position(const DetectionPositionPair<>&)
  {
    error("TODO");
  }

  virtual std::size_t size_of_record_at_ptr(const char * const data_ptr, const std::size_t /*size*/, 
                                            const bool do_byte_swap) const
  { 
    // TODO: get size of record from the file, whereas here I have hard-coded as being 6bytes (I know it's the case for the Orsay data) OtB 15/09

    return std::size_t(6); // std::size_t(data_ptr[0]&0x80);
  }

  virtual Succeeded init_from_data_ptr(const char * const data_ptr, 
                                       const std::size_t
#ifndef NDEBUG
                                       size // only used within assert, so don't define otherwise to avoid compiler warning
#endif
                                       , const bool do_byte_swap)
  {
//    std::cout << " Size  =" << size << " \n" ;
    assert(size >= 6);
//std::cout << " Got to here \n" ;
    std::copy(data_ptr, data_ptr+6, reinterpret_cast<char *>(&this->raw[0]));
    // TODO might have to swap raw[0] and raw[1] if byteswap

    if (do_byte_swap)
      {
        ByteOrder::swap_order(this->raw[0]);
      }
    if (this->is_event() || this->is_time())
      {
//	std::cout << "This is an event \n" ;
        assert(size >= 6);
	
        std::copy(data_ptr+6, data_ptr+6, reinterpret_cast<char *>(&this->raw[1]));
//	std::cout << "after assert an event \n" ;
      }
    if (do_byte_swap)
      {
	error("don't know how to byteswap");
        ByteOrder::swap_order(this->raw[1]);
      }
    return Succeeded::yes;
  }

private:
  union {
    DataType  event_data;
    TimeType   time_data; 
    //GatingType gating_data;
    boost::int32_t  raw[2];
  };
  BOOST_STATIC_ASSERT(sizeof(boost::int32_t)==4);
  BOOST_STATIC_ASSERT(sizeof(DataType)==6); 
  BOOST_STATIC_ASSERT(sizeof(TimeType)==6); 
  //BOOST_STATIC_ASSERT(sizeof(GatingType)==8); 

};


END_NAMESPACE_STIR

#endif
