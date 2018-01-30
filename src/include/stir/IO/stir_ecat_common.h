//
//
/*!
  \file
  \ingroup ECAT

  \brief Declaration of routines which convert ECAT things into our 
  building blocks and vice versa.

  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

#ifndef __stir_IO_stir_ecat_common_H__
#define __stir_IO_stir_ecat_common_H__

#include "stir/common.h"

//*************** namespace macros
#if !defined(STIR_NO_NAMESPACE)
# define START_NAMESPACE_ECAT namespace ecat {
# define END_NAMESPACE_ECAT }
# define USING_NAMESPACE_ECAT using namespace ecat;

# define START_NAMESPACE_ECAT6 namespace ecat6 {
# define END_NAMESPACE_ECAT6 }
# define USING_NAMESPACE_ECAT6 using namespace ecat6;

# define START_NAMESPACE_ECAT7 namespace ecat7 {
# define END_NAMESPACE_ECAT7 }
# define USING_NAMESPACE_ECAT7 using namespace ecat7;

#else

# define START_NAMESPACE_ECAT 
# define END_NAMESPACE_ECAT 
# define USING_NAMESPACE_ECAT 

# define START_NAMESPACE_ECAT6 
# define END_NAMESPACE_ECAT6 
# define USING_NAMESPACE_ECAT6 

# define START_NAMESPACE_ECAT7
# define END_NAMESPACE_ECAT7
# define USING_NAMESPACE_ECAT7 
#endif

START_NAMESPACE_STIR

class NumericType;
class ByteOrder;
class Scanner;

START_NAMESPACE_ECAT

//! Possible values for the data_type field in ECAT headers
/*!
  \ingroup ECAT
  The data_type field is used to indicate how the data are written.
  Note that the VAX_float type is <i>not</i> the same as 
  IEEE float with little_endian byte order.
*/
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
/*!
  \ingroup ECAT
*/
void find_type_from_ECAT_data_type(NumericType& type, ByteOrder& byte_order, const short data_type);

//! Find out which ECAT data type corresponds to a certain NumericType and ByteOrder
/*!
  \ingroup ECAT
  Returns 0 when it does not recognise it */
short find_ECAT_data_type(const NumericType& type, const ByteOrder& byte_order);


//! Find the value used in the ECAT Main header for a given scanner 
/*!
  \ingroup ECAT
  Returns 0 if the scanner is not recognised */
short find_ECAT_system_type(const Scanner& scanner);

//! Find the scanner corresponding to the system type in the main header
/*!
  \ingroup ECAT
  Returns a Scanner(Scanner::Unknown_Scanner) object when the scanner is not recognised. */
Scanner* find_scanner_from_ECAT_system_type(const short system_type);

END_NAMESPACE_ECAT

END_NAMESPACE_STIR
#endif
