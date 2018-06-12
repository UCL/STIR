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

#ifndef __stir_listmode_CListTimeDataECAT8_32bit_H__
#define __stir_listmode_CListTimeDataECAT8_32bit_H__


START_NAMESPACE_STIR
namespace ecat {

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

} // namespace ecat
END_NAMESPACE_STIR

#endif
