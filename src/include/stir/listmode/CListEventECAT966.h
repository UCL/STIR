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
    
  \author Nikos Efthimiou
  \author Kris Thielemans
      
*/

#ifndef __stir_listmode_CListEventECAT966_H__
#define __stir_listmode_CListEventECAT966_H__

#include "stir/listmode/CListEventDataECAT966.h"
#include "stir/listmode/CListEventCylindricalScannerWithViewTangRingRingEncoding.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

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

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
