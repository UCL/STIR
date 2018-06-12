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

#ifndef __stir_listmode_CListTimeDataECAT962_H__
#define __stir_listmode_CListTimeDataECAT962_H__

#include "stir/IO/stir_ecat_common.h" // for namespace macros

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! A class for storing and using a timing 'event' from a listmode file
/*! \ingroup listmode
 */
class CListTimeDataECAT962
{
 public:
  inline unsigned long get_time_in_millisecs() const
  { return static_cast<unsigned long>(time);  }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
  { 
    time = ((1U<<28)-1) & static_cast<unsigned>(time_in_millisecs); 
    // TODO return more useful value
    return Succeeded::yes;
  }
  inline unsigned int get_gating() const
  { return gating; }
  inline Succeeded set_gating(unsigned int g)
  { gating = g & 0xf; return gating==g ? Succeeded::yes : Succeeded::no;}// TODONK check
private:
  friend class CListRecordECAT962; // to give access to type field
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

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
