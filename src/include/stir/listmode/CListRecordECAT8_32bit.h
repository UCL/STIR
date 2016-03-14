/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013-2014 University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Classes for listmode events for the ECAT 8 format
    
  \author Kris Thielemans
*/

#ifndef __stir_listmode_CListRecordECAT8_32bit_H__
#define __stir_listmode_CListRecordECAT8_32bit_H__

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
namespace ecat {

//! Class for decoding storing and using a raw coincidence event from a listmode file from the ECAT 966 scanner
/*! \ingroup listmode

     This class is based on Siemens information on the PETLINK protocol, available at
     http://usa.healthcare.siemens.com/siemens_hwem-hwem_ssxa_websites-context-root/wcm/idc/groups/public/@us/@imaging/@molecular/documents/download/mdax/mjky/~edisp/petlink_guideline_j1-00672485.pdf

     This class just provides the bit-field definitions. You should normally use CListEventECAT8_32bit.

     In the 32-bit event format, the listmode data just stores on offset into a (3D) sinogram. Its
     characteristics are given in the Interfile header.
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

//! Class for storing and using a coincidence event from a listmode file from Siemens scanners using the ECAT 8_32bit format
/*! \todo This implementation only works if the list-mode data is stored without axial compression.
  \todo If the target sinogram has the same characteristics as the sinogram encoding used in the list file 
  (via the offset), the code could be sped-up dramatically by using the information. 
  At present, we go a huge round-about (offset->sinogram->detectors->sinogram->offset)
*/
class CListEventECAT8_32bit : public CListEventCylindricalScannerWithDiscreteDetectors
{
 private:
 public:
  typedef CListEventDataECAT8_32bit DataType;
  DataType get_data() const { return this->data; }

 public:  
  CListEventECAT8_32bit(const shared_ptr<ProjDataInfo>& proj_data_info_sptr);

 //! This routine returns the corresponding detector pair   
 //! \date  11/03/16
 //! \author Nikos Efthimiou
 //! \details I changed the return type is Succeeded.
  virtual Succeeded get_detection_position(DetectionPositionPair<>&) const;

  //! This routine sets in a coincidence event from detector "indices"
  //! \date  11/03/16
  //! \author Nikos Efthimiou
  //! \details I changed the return type is Succeeded.
  virtual Succeeded set_detection_position(const DetectionPositionPair<>&);

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
  std::vector<int> segment_sequence;
  std::vector<int> sizes;

};

//! A class for decoding a raw events that is neither time or coincidence in a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeDataECAT8_32bit
{
 public:

#if STIRIsNativeByteOrderBigEndian
  unsigned    type : 1;    /* 0-coincidence event, 1-time tick */
  unsigned    deadtimeetc : 2;  /* extra bits differentiating between timing or other stuff, zero if timing event */
  unsigned    time : 29 ;  /* since scan start */
#else
  // Do byteswapping first before using this bit field.
  unsigned    time : 29 ;  /* since scan start */
  unsigned    deadtimeetc : 2;  /* extra bits differentiating between timing or other stuff, zero if timing event */
  unsigned    type : 1;    /* 0-coincidence event, 1-time tick */
#endif
};


class CListDataAnyECAT8_32bit
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
  bool is_other() const
  { return this->data.type == 1U && this->data.deadtimeetc != 0U; }
  bool is_event() const
  { return this->data.type == 0U; }


 private:
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT8_32bit)==4); 
  union 
  {
    CListTimeDataECAT8_32bit   data;
    boost::int32_t         raw;
  };
};


//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeECAT8_32bit : public CListTime
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

 private:
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT8_32bit)==4); 
  union 
  {
    CListTimeDataECAT8_32bit   data;
    boost::int32_t         raw;
  };
};

//! A class for a general element of a listmode file for a Siemens scanner using the ECAT8 32bit format.
/*! \ingroup listmode
   We currently only support coincidence events and  a timing flag.
   Here we only support the 32bit version specified by the PETLINK protocol.

   This class is based on Siemens information on the PETLINK protocol, available at
   http://usa.healthcare.siemens.com/siemens_hwem-hwem_ssxa_websites-context-root/wcm/idc/groups/public/@us/@imaging/@molecular/documents/download/mdax/mjky/~edisp/petlink_guideline_j1-00672485.pdf

*/
 class CListRecordECAT8_32bit : public CListRecord // currently no gating yet
{

  //public:

  bool is_time() const
  { return this->any_data.is_time(); }
  /*
  bool is_gating_input() const
  { return this->is_time(); }
  */
  bool is_event() const
  { return this->any_data.is_event(); }
  virtual CListEventECAT8_32bit&  event() 
    { return this->event_data; }
  virtual const CListEventECAT8_32bit&  event() const
    { return this->event_data; }
  virtual CListTimeECAT8_32bit&   time()
    { return this->time_data; }
  virtual const CListTimeECAT8_32bit&   time() const
    { return this->time_data; }

  bool is_random() const
    { return false; }
  bool is_scattered() const
    { return false; }

  bool operator==(const CListRecord& e2) const
  {
    return dynamic_cast<CListRecordECAT8_32bit const *>(&e2) != 0 &&
      raw == dynamic_cast<CListRecordECAT8_32bit const &>(e2).raw;
  }	 

 public:     
 CListRecordECAT8_32bit(const shared_ptr<ProjDataInfo>& proj_data_info_sptr) :
  event_data(proj_data_info_sptr)
    {}

  virtual Succeeded init_from_data_ptr(const char * const data_ptr, 
                                       const std::size_t
#ifndef NDEBUG
                                       size // only used within assert, so commented-out otherwise to avoid compiler warnings
#endif
                                       , const bool do_byte_swap)
  {
    assert(size >= 4);
    std::copy(data_ptr, data_ptr+4, reinterpret_cast<char *>(&raw));
    if (do_byte_swap)
      ByteOrder::swap_order(raw);
    this->any_data.init_from_data_ptr(&raw);
    // should in principle check return value, but it's always Succeeded::yes anyway
    if (this->any_data.is_time())
      return this->time_data.init_from_data_ptr(&raw);
     else if (this->any_data.is_event())
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
  CListDataAnyECAT8_32bit   any_data; 
  boost::int32_t         raw; // this raw field isn't strictly necessary, get rid of it?

};

} // namespace ecat
END_NAMESPACE_STIR

#endif

