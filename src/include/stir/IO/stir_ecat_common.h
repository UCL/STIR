//
// $Id$
//
/*!
  \file
  \ingroup ECAT
  \ingroup IO

  \brief Declaration of routines which convert ECAT things into our 
  building blocks and vice versa.

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_stir_ecat_common_H__
#define __stir_IO_stir_ecat_common_H__

#include "stir/common.h"

START_NAMESPACE_STIR

class NumericType;
class ByteOrder;
class Scanner;


//  MatDataType
typedef enum {
    ECAT_unknown_data_type = 0,
    ECAT_Byte_data_type = 1,
    ECAT_I2_little_endian_data_type = 2,
    ECAT_I4_little_endian_data_type = 3,
    ECAT_R4_VAX_data_type = 4,
    ECAT_R4_IEEE_big_endian_data_type = 5,
    ECAT_I2_big_endian_data_type = 6, 
    ECAT_I4_big_endian_data_type = 7 
} ECATDataType;


//! Find out which NumericType and ByteOrder corresponds to a ECAT data type
void find_type_from_ECAT_data_type(NumericType& type, ByteOrder& byte_order, const short data_type);

//! Find out which ECAT data type corresponds to a certain NumericType and ByteOrder
/*! Returns 0 when it does not recognise it */
short find_ECAT_data_type(const NumericType& type, const ByteOrder& byte_order);


//! Find the value used in the ECAT Main header for a given scanner 
/*! Returns 0 if the scanner is not recognised */
short find_ECAT_system_type(const Scanner& scanner);

//! Find the scanner corresponding to the system type in the main header
/*! Returns a Scanner(Scanner::Unknown_Scanner) object when the scanner is not recognised. */
Scanner* find_scanner_from_ECAT_system_type(const short system_type);

END_NAMESPACE_STIR
#endif
