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

#include "stir/listmode/CListRecordECAT962.h"
#include "stir/ByteOrder.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

CListRecordECAT962::CListRecordECAT962() :
CListEventCylindricalScannerWithViewTangRingRingEncoding<CListRecordECAT962>(shared_ptr<Scanner>(new Scanner(Scanner::E962)))
  {}

bool
CListRecordECAT962::is_time() const
{ return time_data.type == 1U; }

bool
CListRecordECAT962::is_gating_input() const
{ return this->is_time(); }

bool
CListRecordECAT962::is_event() const
{ return time_data.type == 0U; }

CListEvent&
CListRecordECAT962::event()
  { return *this; }

const CListEvent&
CListRecordECAT962::event() const
  { return *this; }

CListTime&
CListRecordECAT962::time()
  { return *this; }

const CListTime&
CListRecordECAT962::time() const
  { return *this; }

CListGatingInput&
CListRecordECAT962::gating_input()
  { return *this; }

const CListGatingInput&
CListRecordECAT962::gating_input() const
{ return *this; }

Succeeded
CListRecordECAT962::init_from_data_ptr(const char * const data_ptr,
                                     const std::size_t
#ifndef NDEBUG
                                     size // only used within assert, so don't define otherwise to avoid compiler warning
#endif
                                     , const bool do_byte_swap)
{
  assert(size >= 4);
  std::copy(data_ptr, data_ptr+4, reinterpret_cast<char *>(&raw));// TODO necessary for operator==
  if (do_byte_swap)
    ByteOrder::swap_order(raw);
  return Succeeded::yes;
}

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR
