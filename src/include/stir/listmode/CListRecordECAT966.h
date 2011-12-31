//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
  \brief Classes for listmode events for the ECAT 966 (aka Exact 3d)
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/

#ifndef __stir_listmode_CListRecordECAT966_H__
#define __stir_listmode_CListRecordECAT966_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithViewTangRingRingEncoding.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/IO/stir_ecat_common.h" // for namespace macros
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"
#include "stir/round.h"
#include "boost/static_assert.hpp"
#include "boost/cstdint.hpp"
#include "stir/DetectionPositionPair.h"

START_NAMESPACE_STIR

START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! Class for decoding storing and using a raw coincidence event from a listmode file from the ECAT 966 scanner
/*! \ingroup listmode

     This class just provides the bit-field definitions. You should normally use CListEventECAT966.

     For the 966 the event word is 32 bit. To save 1 bit in size, a 2d sinogram
     encoding is used (as opposed to a detector number on the ring
     for both events).
     Both bin and view use 9 bits, so their maximum range is
     512 values, which is fine for the 966 (which needs only 288).

*/
class CListEventDataECAT966
{
 public:
  
  /*! This routine returns the corresponding tangential_pos_num,view_num,ring_a and ring_b
   */
  void get_sinogram_and_ring_coordinates(int& view, int& tangential_pos_num, unsigned int& ring_a, unsigned int& ring_b) const;
  
  /*! This routine constructs a coincidence event */
  void set_sinogram_and_ring_coordinates(
			const int view_num, const int tangential_pos_num, 
			const int ring_a, const int ring_b);

  /* ring encoding. use as follows:
       #define CRYSTALRINGSPERDETECTOR 8
       ringA = ( block_A_ring * CRYSTALRINGSPERDETECTOR ) + block_A_detector ;
       in practice, for the 966 block_A_ring ranges from 0 to 5.

       This organisation corresponds to physical detector blocks (which
       have 8 crystal rings). Names are not very good probably...
       */				
    /* 'random' bit:
        1 if event is Random (it fell in delayed time window) */
    /* bin field  is shifted in a funny way, use the following code to find
       bin_number:
         if ( bin > NumProjBinsBy2 ) bin -= NumProjBins ;
	 */

#if STIRIsNativeByteOrderBigEndian
  unsigned    type    : 1; /* 0-coincidence event, 1-time tick */
  unsigned    block_B_ring : 3;
  unsigned    block_A_ring : 3;
  unsigned    block_B_detector : 3;
  unsigned    block_A_detector : 3;
  unsigned    random  : 1;
  unsigned    bin : 9;
  unsigned    view : 9;
#else
  // Do byteswapping first before using this bit field.
  unsigned    view : 9;
  unsigned    bin : 9;
  unsigned    random  : 1;
  unsigned    block_A_detector : 3;
  unsigned    block_B_detector : 3;
  unsigned    block_A_ring : 3;
  unsigned    block_B_ring : 3;
  unsigned    type    : 1; /* 0-coincidence event, 1-time tick */

#endif
}; /*-coincidence event*/

//! Class for storing and using a coincidence event from a listmode file from the ECAT 966 scanner
class CListEventECAT966 : public CListEventCylindricalScannerWithViewTangRingRingEncoding<CListEventECAT966>
{
 private:
 public:
  typedef CListEventDataECAT966 DataType;
  DataType get_data() const { return this->data; }

 public:  
  CListEventECAT966() :
  CListEventCylindricalScannerWithViewTangRingRingEncoding<CListEventECAT966>(shared_ptr<Scanner>(new Scanner(Scanner::E966)))
    {}

  Succeeded init_from_data_ptr(const void * const ptr)
    {
      const char * const data_ptr = reinterpret_cast<const char * const >(ptr);
      std::copy(data_ptr, data_ptr+sizeof(this->raw), reinterpret_cast<char *>(&this->raw));
      return Succeeded::yes;
    }
  inline bool is_prompt() const { return this->data.random == 0; }
  inline Succeeded set_prompt(const bool prompt = true) 
  { if (prompt) this->data.random=0; else this->data.random=1; return Succeeded::yes; }

 private:
  BOOST_STATIC_ASSERT(sizeof(CListEventDataECAT966)==4); 
  union 
  {
    CListEventDataECAT966   data;
    boost::int32_t         raw;
  };
};

//! A class for decoding a raw  timing 'event' from a listmode file from the ECAT 966 scanner
/*! \ingroup listmode
     This class just provides the bit-field definitions. You should normally use CListTimeECAT966.

 */
class CListTimeDataECAT966
{
 public:

#if STIRIsNativeByteOrderBigEndian
  unsigned    type : 1;    /* 0-coincidence event, 1-time tick */
  unsigned    gating : 4;  /* some info about the gating signals */
  unsigned    time : 27 ;  /* since scan start */
#else
  // Do byteswapping first before using this bit field.
  unsigned    time : 27 ;  /* since scan start */
  unsigned    gating : 4;  /* some info about the gating signals */
  unsigned    type : 1;    /* 0-coincidence event, 1-time tick */
#endif
};


//! A class for storing and using a timing 'event' from a listmode file from the ECAT 966 scanner
/*! \ingroup listmode
 */
class CListTimeECAT966 : public CListTime, public CListGatingInput
{
 public:
  Succeeded init_from_data_ptr(const void * const ptr)
    {
      const char * const data_ptr = reinterpret_cast<const char * const >(ptr);
      std::copy(data_ptr, data_ptr+sizeof(this->raw), reinterpret_cast<char *>(&this->raw));
      return Succeeded::yes;
    }
  bool is_time() const
  { return this->data.type == 1U; }
  inline unsigned long get_time_in_millisecs() const
  { return static_cast<unsigned long>(this->data.time);  }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
  { 
    this->data.time = ((1U<<28)-1) & static_cast<unsigned>(time_in_millisecs); 
    // TODO return more useful value
    return Succeeded::yes;
  }
  inline unsigned int get_gating() const
  { return this->data.gating; }
  inline Succeeded set_gating(unsigned int g)
  { this->data.gating = g & 0xf; return this->data.gating==g ? Succeeded::yes : Succeeded::no;}

 private:
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT966)==4); 
  union 
  {
    CListTimeDataECAT966   data;
    boost::int32_t         raw;
  };
};

//! A class for a general element of a listmode file
/*! \ingroup listmode
   For the 966 it's either a coincidence event, or a timing flag.*/
class CListRecordECAT966 : public CListRecordWithGatingInput
{

  //public:

  bool is_time() const
  { return this->time_data.is_time(); }
  bool is_gating_input() const
  { return this->is_time(); }
  bool is_event() const
  { return !this->is_time(); }
  virtual CListEventECAT966&  event() 
    { return this->event_data; }
  virtual const CListEventECAT966&  event() const
    { return this->event_data; }
  virtual CListTimeECAT966&   time()
    { return this->time_data; }
  virtual const CListTimeECAT966&   time() const
    { return this->time_data; }
  virtual CListTimeECAT966&  gating_input()
    { return this->time_data; }
  virtual const CListTimeECAT966&  gating_input() const
  { return this->time_data; }

  bool operator==(const CListRecord& e2) const
  {
    return dynamic_cast<CListRecordECAT966 const *>(&e2) != 0 &&
      raw == dynamic_cast<CListRecordECAT966 const &>(e2).raw;
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
  CListEventECAT966  event_data;
  CListTimeECAT966   time_data; 
  boost::int32_t         raw;

};


END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
