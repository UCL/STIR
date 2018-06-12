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
  \brief Definition of CListEventDataECAT8_32bit
    
  \author Kris Thielemans
*/

#ifndef __stir_listmode_CListEventDataECAT8_32bit_H__
#define __stir_listmode_CListEventDataECAT8_32bit_H__

#include "stir/common.h"

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

} // namespace ecat
END_NAMESPACE_STIR

#endif
