/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013-2014 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
namespace ecat
{

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
  /* 'delayed' bit:
      0 if event is delayed (it fell in delayed time window) */

#if STIRIsNativeByteOrderBigEndian
  unsigned type : 1; /* 0-coincidence event, 1-time tick */
  unsigned delayed : 1;
  unsigned offset : 30;
#else
  // Do byteswapping first before using this bit field.
  unsigned offset : 30;
  unsigned delayed : 1;
  unsigned type : 1; /* 0-coincidence event, 1-time tick */

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
  CListEventECAT8_32bit(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr);

  //! This routine returns the corresponding detector pair
  void get_detection_position(DetectionPositionPair<>&) const override;

  //! This routine sets in a coincidence event from detector "indices"
  void set_detection_position(const DetectionPositionPair<>&) override;

  Succeeded init_from_data_ptr(const void* const ptr)
  {
    const char* const data_ptr = reinterpret_cast<const char* const>(ptr);
    std::copy(data_ptr, data_ptr + sizeof(this->raw), reinterpret_cast<char*>(&this->raw));
    return Succeeded::yes;
  }
  inline bool is_prompt() const override { return this->data.delayed == 1; }
  inline Succeeded set_prompt(const bool prompt = true) override
  {
    if (prompt)
      this->data.delayed = 1;
    else
      this->data.delayed = 0;
    return Succeeded::yes;
  }

private:
  BOOST_STATIC_ASSERT(sizeof(CListEventDataECAT8_32bit) == 4);
  union
  {
    CListEventDataECAT8_32bit data;
    boost::int32_t raw;
  };
  std::vector<int> segment_sequence;
  std::vector<int> timing_poss_sequence;
  std::vector<int> sizes;
};

//! Bit-field struct for the PETLINK time marker word (type=1, deadtimeetc=0)
/*! \ingroup listmode
    PETLINK 32-bit layout (little-endian):
      bits [28:0]  = time in milliseconds since scan start
      bits [30:29] = deadtimeetc (0 = pure time tick)
      bit  [31]    = type (1 = non-event)
*/
class CListTimeDataECAT8_32bit
{
public:
#if STIRIsNativeByteOrderBigEndian
  unsigned type : 1;        /* 0-coincidence event, 1-time tick */
  unsigned deadtimeetc : 2; /* extra bits differentiating between timing or other stuff, zero if timing event */
  unsigned time : 29;       /* since scan start */
#else
  // Do byteswapping first before using this bit field.
  unsigned time : 29;       /* since scan start */
  unsigned deadtimeetc : 2; /* extra bits differentiating between timing or other stuff, zero if timing event */
  unsigned type : 1;        /* 0-coincidence event, 1-time tick */
#endif
};

//! Bit-field struct for the PETLINK singles word (type=1, deadtimeetc=2)
/*! \ingroup listmode
    PETLINK 32-bit layout (little-endian):
      bits [18:0]  = singles count (NiftyPET stores this pre-shifted left by 3)
      bits [28:19] = bucket index (0-based, up to NBUCKTS buckets)
      bits [30:29] = subtype/deadtimeetc = 0b10 (singles marker)
      bit  [31]    = type = 1 (non-event)
*/
class CListSinglesDataECAT8_32bit
{
public:
#if STIRIsNativeByteOrderBigEndian
  unsigned type : 1;     /* 1 = non-event */
  unsigned subtype : 2;  /* 0b10 = singles record */
  unsigned bucket : 10;  /* singles bucket index */
  unsigned singles : 19; /* singles count */
#else
  // Do byteswapping first before using this bit field.
  unsigned singles : 19; /* singles count */
  unsigned bucket : 10;  /* singles bucket index */
  unsigned subtype : 2;  /* 0b10 = singles record */
  unsigned type : 1;     /* 1 = non-event */
#endif
};

//! A class for decoding a raw "other" word (neither time nor coincidence) from the ECAT 8_32bit listmode stream
/*! \ingroup listmode
    Used purely to classify the word type before dispatching to the appropriate handler.
    Reuses CListTimeDataECAT8_32bit's bit fields since type and deadtimeetc occupy the
    same bit positions across all non-event word subtypes.
*/
class CListDataAnyECAT8_32bit
{
public:
  Succeeded init_from_data_ptr(const void* const ptr)
  {
    const char* const data_ptr = reinterpret_cast<const char* const>(ptr);
    std::copy(data_ptr, data_ptr + sizeof(this->raw), reinterpret_cast<char*>(&this->raw));
    return Succeeded::yes;
  }
  bool is_time() const { return this->data.type == 1U && this->data.deadtimeetc == 0U; }
  bool is_other() const { return this->data.type == 1U && this->data.deadtimeetc != 0U; }
  bool is_event() const { return this->data.type == 0U; }

private:
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT8_32bit) == 4);
  union
  {
    CListTimeDataECAT8_32bit data;
    boost::int32_t raw;
  };
};

//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeECAT8_32bit : public ListTime
{
public:
  Succeeded init_from_data_ptr(const void* const ptr)
  {
    const char* const data_ptr = reinterpret_cast<const char* const>(ptr);
    std::copy(data_ptr, data_ptr + sizeof(this->raw), reinterpret_cast<char*>(&this->raw));
    return Succeeded::yes;
  }
  bool is_time() const { return this->data.type == 1U && this->data.deadtimeetc == 0U; }
  inline unsigned long get_time_in_millisecs() const override { return static_cast<unsigned long>(this->data.time); }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs) override
  {
    this->data.time = ((1U << 30) - 1) & static_cast<unsigned>(time_in_millisecs);
    // TODO return more useful value
    return Succeeded::yes;
  }

private:
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT8_32bit) == 4);
  union
  {
    CListTimeDataECAT8_32bit data;
    boost::int32_t raw;
  };
};

//! A class for storing and using a singles bucket record from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
    Singles records are emitted by the scanner approximately once per second per bucket.
    Each record encodes the singles count for one detector bucket over the preceding interval.
    These are used to estimate dead time and randoms.

    The PETLINK singles word has type=1, deadtimeetc=0b10 (i.e. is_other() == true in
    CListDataAnyECAT8_32bit), so it falls through the existing time/event checks.
*/
class CListSinglesECAT8_32bit : public ListSingles
{
public:
  Succeeded init_from_data_ptr(const void* const ptr)
  {
    const char* const data_ptr = reinterpret_cast<const char* const>(ptr);
    std::copy(data_ptr, data_ptr + sizeof(this->raw), reinterpret_cast<char*>(&this->raw));
    return Succeeded::yes;
  }

  //! Returns true if this word is a singles record (type=1, subtype=0b10)
  bool is_singles() const { return this->data.type == 1U && this->data.subtype == 1U; }

  unsigned int get_bucket_index() const override { return this->data.bucket; }
  float get_singles_count() const override { return static_cast<float>(this->data.singles); }

private:
  BOOST_STATIC_ASSERT(sizeof(CListSinglesDataECAT8_32bit) == 4);
  union
  {
    CListSinglesDataECAT8_32bit data;
    boost::int32_t raw;
  };
};

//! A class for a general element of a listmode file for a Siemens scanner using the ECAT8 32bit format.
/*! \ingroup listmode
   Supports coincidence events, timing records, and singles bucket records.
   Here we only support the 32bit version specified by the PETLINK protocol.

   This class is based on Siemens information on the PETLINK protocol, available at
   http://usa.healthcare.siemens.com/siemens_hwem-hwem_ssxa_websites-context-root/wcm/idc/groups/public/@us/@imaging/@molecular/documents/download/mdax/mjky/~edisp/petlink_guideline_j1-00672485.pdf

*/
class CListRecordECAT8_32bit : public CListRecord // currently no gating yet
{

public:
  bool is_time() const override { return this->any_data.is_time(); }
  /*
  bool is_gating_input() const
  { return this->is_time(); }
  */
  bool is_event() const override { return this->any_data.is_event(); }
  CListEventECAT8_32bit& event() override { return this->event_data; }
  const CListEventECAT8_32bit& event() const override { return this->event_data; }
  CListTimeECAT8_32bit& time() override { return this->time_data; }
  const CListTimeECAT8_32bit& time() const override { return this->time_data; }
  bool is_singles() const override { return this->singles_data.is_singles(); }
  CListSinglesECAT8_32bit& singles() override { return this->singles_data; }
  const CListSinglesECAT8_32bit& singles() const override { return this->singles_data; }

  bool operator==(const CListRecord& e2) const
  {
    return dynamic_cast<CListRecordECAT8_32bit const*>(&e2) != 0 && raw == dynamic_cast<CListRecordECAT8_32bit const&>(e2).raw;
  }

public:
  CListRecordECAT8_32bit(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
      : event_data(proj_data_info_sptr)
  {}

  virtual Succeeded init_from_data_ptr(const char* const data_ptr,
                                       const std::size_t
#ifndef NDEBUG
                                           size // only used within assert, so commented-out otherwise to avoid compiler warnings
#endif
                                       ,
                                       const bool do_byte_swap)
  {
    assert(size >= 4);
    std::copy(data_ptr, data_ptr + 4, reinterpret_cast<char*>(&raw));
    if (do_byte_swap)
      ByteOrder::swap_order(raw);
    this->any_data.init_from_data_ptr(&raw);
    // should in principle check return value, but it's always Succeeded::yes anyway
    if (this->any_data.is_time())
      return this->time_data.init_from_data_ptr(&raw);
    else if (this->any_data.is_event())
      return this->event_data.init_from_data_ptr(&raw);
    else if (this->any_data.is_other())
      return this->singles_data.init_from_data_ptr(&raw);
    else
      return Succeeded::yes;
  }

  virtual std::size_t
  size_of_record_at_ptr(const char* const /*data_ptr*/, const std::size_t /*size*/, const bool /*do_byte_swap*/) const
  {
    return 4;
  }

private:
  CListEventECAT8_32bit event_data;
  CListTimeECAT8_32bit time_data;
  CListDataAnyECAT8_32bit any_data;
  CListSinglesECAT8_32bit singles_data;
  boost::int32_t raw; // this raw field isn't strictly necessary, get rid of it?
};

} // namespace ecat
END_NAMESPACE_STIR

#endif