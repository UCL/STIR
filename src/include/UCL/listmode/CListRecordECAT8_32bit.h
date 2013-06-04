//
// $Id: CListRecordECAT966.h,v 1.9 2011-12-31 16:42:45 kris Exp $
//
/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013 University College London

  This file contains proprietary information supplied by Siemens so cannot
  be redistributed without their consent.
*/
/*!
  \file
  \ingroup listmode
  \brief Classes for listmode events for the ECAT 8 format
    
  \author Kris Thielemans
      
  $Date: 2011-12-31 16:42:45 $
  $Revision: 1.9 $
*/

#ifndef __stir_listmode_CListRecordECAT966_H__
#define __stir_listmode_CListRecordECAT966_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"
#include "stir/round.h"
#include "boost/static_assert.hpp"
#include "boost/cstdint.hpp"
#include "stir/DetectionPositionPair.h"

START_NAMESPACE_STIR
namespace UCL {

//! Class for decoding storing and using a raw coincidence event from a listmode file from the ECAT 966 scanner
/*! \ingroup listmode

     This class just provides the bit-field definitions. You should normally use CListEventECAT966.

     For the 966 the event word is 32 bit. To save 1 bit in size, a 2d sinogram
     encoding is used (as opposed to a detector number on the ring
     for both events).
     Both bin and view use 9 bits, so their maximum range is
     512 values, which is fine for the 966 (which needs only 288).

*/
class CListEventDataECAT8_32bit
{
 public:
  
    /* 'random' bit:
        0 if event is Random (it fell in delayed time window) */

#if STIRIsNativeByteOrderBigEndian
  unsigned    type    : 1; /* 0-coincidence event, 1-time tick */
  unsigned    delayed  : 1;
  unsigned    offset : 30;
#else
  // Do byteswapping first before using this bit field.
  unsigned    offset : 30;
  unsigned    delayed  : 1;
  unsigned    type    : 1; /* 0-coincidence event, 1-time tick */

#endif
}; /*-coincidence event*/

//! Class for storing and using a coincidence event from a listmode file from the ECAT 8_32bit scanner
class CListEventECAT8_32bit : public CListEventCylindricalScannerWithDiscreteDetectors
{
 private:
 public:
  typedef CListEventDataECAT8_32bit DataType;
  DataType get_data() const { return this->data; }

 public:  
  CListEventECAT8_32bit();

 //! This routine returns the corresponding detector pair   
  virtual void get_detection_position(DetectionPositionPair<>&) const;

  //! This routine sets in a coincidence event from detector "indices"
  virtual void set_detection_position(const DetectionPositionPair<>&);

  Succeeded init_from_data_ptr(const void * const ptr)
    {
      const char * const data_ptr = reinterpret_cast<const char * const >(ptr);
      std::copy(data_ptr, data_ptr+sizeof(this->raw), reinterpret_cast<char *>(&this->raw));
      return Succeeded::yes;
    }
  inline bool is_prompt() const { return this->data.delayed == 1; }
  inline Succeeded set_prompt(const bool prompt = true) 
  { if (prompt) this->data.delayed=1; else this->data.delayed=0; return Succeeded::yes; }

 private:
  BOOST_STATIC_ASSERT(sizeof(CListEventDataECAT8_32bit)==4); 
  union 
  {
    CListEventDataECAT8_32bit   data;
    boost::int32_t         raw;
  };
  vector<int> segment_sequence;
  vector<int> sizes;

};

//! A class for decoding a raw  timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
     This class just provides the bit-field definitions. You should normally use CListTimeECAT8_32bit.

 */
class CListTimeDataECAT8_32bit
{
 public:

#if STIRIsNativeByteOrderBigEndian
  unsigned    type : 1;    /* 0-coincidence event, 1-time tick */
  unsigned    deadtimeetc : 2;  /* some info about the gating signals, zero if timing event */
  unsigned    time : 29 ;  /* since scan start */
#else
  // Do byteswapping first before using this bit field.
  unsigned    time : 29 ;  /* since scan start */
  unsigned    deadtimeetc : 2;  /* some info about the gating signals, zero if timing event */
  unsigned    type : 1;    /* 0-coincidence event, 1-time tick */
#endif
};


//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeECAT8_32bit : public CListTime, public CListGatingInput
{
 public:
  Succeeded init_from_data_ptr(const void * const ptr)
    {
      const char * const data_ptr = reinterpret_cast<const char * const >(ptr);
      std::copy(data_ptr, data_ptr+sizeof(this->raw), reinterpret_cast<char *>(&this->raw));
      return Succeeded::yes;
    }
  bool is_time() const
  { return this->data.type == 1U && this->data.deadtimeetc == 0U; }
  inline unsigned long get_time_in_millisecs() const
  { return static_cast<unsigned long>(this->data.time);  }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
  { 
    this->data.time = ((1U<<30)-1) & static_cast<unsigned>(time_in_millisecs); 
    // TODO return more useful value
    return Succeeded::yes;
  }
  // TODO
  inline unsigned int get_gating() const
  { return 0; }
  inline Succeeded set_gating(unsigned int g)
  { return Succeeded::no;}

 private:
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT8_32bit)==4); 
  union 
  {
    CListTimeDataECAT8_32bit   data;
    boost::int32_t         raw;
  };
};

//! A class for a general element of a listmode file
/*! \ingroup listmode
   For the 8_32bit it's either a coincidence event, or a timing flag.*/
class CListRecordECAT8_32bit : public CListRecordWithGatingInput
{

  //public:

  bool is_time() const
  { return this->time_data.is_time(); }
  bool is_gating_input() const
  { return this->is_time(); }
  bool is_event() const
  { return !this->is_time(); }
  virtual CListEventECAT8_32bit&  event() 
    { return this->event_data; }
  virtual const CListEventECAT8_32bit&  event() const
    { return this->event_data; }
  virtual CListTimeECAT8_32bit&   time()
    { return this->time_data; }
  virtual const CListTimeECAT8_32bit&   time() const
    { return this->time_data; }
  virtual CListTimeECAT8_32bit&  gating_input()
    { return this->time_data; }
  virtual const CListTimeECAT8_32bit&  gating_input() const
  { return this->time_data; }

  bool operator==(const CListRecord& e2) const
  {
    return dynamic_cast<CListRecordECAT8_32bit const *>(&e2) != 0 &&
      raw == dynamic_cast<CListRecordECAT8_32bit const &>(e2).raw;
  }	 

 public:     
  virtual Succeeded init_from_data_ptr(const char * const data_ptr, 
                                       const std::size_t
#ifndef NDEBUG
                                       size // only use within assert
#endif
                                       , const bool do_byte_swap)
  {
    assert(size >= 4);
    std::copy(data_ptr, data_ptr+4, reinterpret_cast<char *>(&raw));// TODO necessary for operator==
    if (do_byte_swap)
      ByteOrder::swap_order(raw);
    this->time_data.init_from_data_ptr(&raw);
    // should in principle check return value, but it's always Succeeded::yes anyway
    if (!this->is_time())
      return this->event_data.init_from_data_ptr(&raw);
    else
      return Succeeded::yes;
  }

  virtual std::size_t size_of_record_at_ptr(const char * const /*data_ptr*/, const std::size_t /*size*/, 
                                            const bool /*do_byte_swap*/) const
  { return 4; }

 private:
  CListEventECAT8_32bit  event_data;
  CListTimeECAT8_32bit   time_data; 
  boost::int32_t         raw;

};

} // namespace UCL 
END_NAMESPACE_STIR

#endif
