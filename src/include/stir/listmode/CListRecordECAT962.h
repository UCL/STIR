//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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
  \brief Classes for listmode events for the ECAT 962 (aka Exact HR+)
    
  \author Nikos Efthimiou
  \author Kris Thielemans
      
*/

#ifndef __stir_listmode_CListRecordECAT962_H__
#define __stir_listmode_CListRecordECAT962_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithViewTangRingRingEncoding.h"

#include "stir/listmode/CListEventDataECAT962.h"
#include "stir/listmode/CListTimeDataECAT962.h"
#include "stir/listmode/CListRecordWithGatingInput.h"
#include "stir/listmode/CListGatingInput.h"

#include "stir/IO/stir_ecat_common.h" // for namespace macros
#include "stir/Succeeded.h"
#include "boost/static_assert.hpp"
#include "boost/cstdint.hpp"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! A class for a general element of a listmode file
/*! \ingroup listmode
   For the 962 it's either a coincidence event, or a timing flag.*/
class CListRecordECAT962 : public CListRecordWithGatingInput, public CListTime, public CListGatingInput,
    public  CListEventCylindricalScannerWithViewTangRingRingEncoding<CListRecordECAT962>
{
 public:
  typedef CListEventDataECAT962 DataType;
  DataType get_data() const { return this->event_data; }

 public:  
  CListRecordECAT962();

  bool is_time() const;
  bool is_gating_input() const;
  bool is_event() const;
  virtual CListEvent&  event() ;
  virtual const CListEvent&  event() const;
  virtual CListTime&   time();
  virtual const CListTime&   time() const;
  virtual CListGatingInput&  gating_input();
  virtual const CListGatingInput&  gating_input() const;

  bool operator==(const CListRecord& e2) const
  {
    return dynamic_cast<CListRecordECAT962 const *>(&e2) != 0 &&
      raw == static_cast<CListRecordECAT962 const &>(e2).raw;
  }	    

  // time 
  inline unsigned long get_time_in_millisecs() const 
    { return time_data.get_time_in_millisecs(); }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
    { return time_data.set_time_in_millisecs(time_in_millisecs); }
  inline unsigned int get_gating() const
    { return time_data.get_gating(); }
  inline Succeeded set_gating(unsigned int g) 
    { return time_data.set_gating(g); }

  // event
  inline bool is_prompt() const { return event_data.is_prompt(); }
  inline Succeeded set_prompt(const bool prompt = true) 
  { return event_data.set_prompt(prompt); }


  virtual Succeeded init_from_data_ptr(const char * const data_ptr, 
                                       const std::size_t
#ifndef NDEBUG
                                       size // only used within assert, so don't define otherwise to avoid compiler warning
#endif
                                       , const bool do_byte_swap);

  virtual std::size_t size_of_record_at_ptr(const char * const /*data_ptr*/, const std::size_t /*size*/, 
                                            const bool /*do_byte_swap*/) const
  { return 4; }

private:
  union {
    CListEventDataECAT962  event_data;
    CListTimeDataECAT962   time_data; 
    boost::int32_t         raw;
  };
  BOOST_STATIC_ASSERT(sizeof(boost::int32_t)==4);
  BOOST_STATIC_ASSERT(sizeof(CListEventDataECAT962)==4); 
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT962)==4); 

};



END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
