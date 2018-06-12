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

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/listmode/CListRecordECAT8_32bit.h"
#include "stir/ByteOrder.h"

START_NAMESPACE_STIR
namespace ecat {

    CListRecordECAT8_32bit::CListRecordECAT8_32bit(const shared_ptr<ProjDataInfo>& proj_data_info_sptr) :
     event_data(proj_data_info_sptr)
       {}

    bool CListRecordECAT8_32bit::is_time() const
    { return this->any_data.is_time(); }
    /*
    bool is_gating_input() const
    { return this->is_time(); }
    */
    bool CListRecordECAT8_32bit::is_event() const
    { return this->any_data.is_event(); }

    CListEventECAT8_32bit&  CListRecordECAT8_32bit::event()
      { return this->event_data; }

    const CListEventECAT8_32bit&  CListRecordECAT8_32bit::event() const
      { return this->event_data; }

    CListTimeECAT8_32bit&  CListRecordECAT8_32bit::time()
      { return this->time_data; }

   const CListTimeECAT8_32bit&  CListRecordECAT8_32bit::time() const
      { return this->time_data; }

   Succeeded
           CListRecordECAT8_32bit::init_from_data_ptr(const char * const data_ptr,
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

} // namespace ecat
END_NAMESPACE_STIR
