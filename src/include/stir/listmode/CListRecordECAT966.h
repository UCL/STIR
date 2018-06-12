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
  \brief Classes for listmode events for the ECAT 966 (aka Exact 3d)
    
  \author Kris Thielemans
      
*/

#ifndef __stir_listmode_CListRecordECAT966_H__
#define __stir_listmode_CListRecordECAT966_H__

#include "stir/listmode/CListRecordWithGatingInput.h"
#include "stir/listmode/CListEventECAT966.h"
#include "stir/listmode/CListTimeECAT966.h"
#include "stir/IO/stir_ecat_common.h" // for namespace macros
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "boost/cstdint.hpp"

START_NAMESPACE_STIR

START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7


//! A class for a general element of a listmode file
/*! \ingroup listmode
   For the 966 it's either a coincidence event, or a timing flag.*/
class CListRecordECAT966 : public CListRecordWithGatingInput
{

  //public:

  bool is_time() const;
  bool is_gating_input() const;
  bool is_event() const;
  virtual CListEventECAT966&  event() ;
  virtual const CListEventECAT966&  event() const;
  virtual CListTimeECAT966&   time();
  virtual const CListTimeECAT966&   time() const;
  virtual CListTimeECAT966&  gating_input();
  virtual const CListTimeECAT966&  gating_input() const;

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
