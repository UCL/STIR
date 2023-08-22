//
//
/*!
  \file
  \ingroup ECAT

  \brief Implementation of routines which convert ECAT6, ECAT7 and ECAT8 things into our  building blocks and vice versa. 

  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2020, 2023 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/ByteOrder.h"
#include "stir/NumericType.h"
#include "stir/Scanner.h" 
#include "stir/ProjDataInfo.h"
#include "stir/IO/stir_ecat_common.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT


void find_type_from_ECAT_data_type(NumericType& type, ByteOrder& byte_order, const short data_type)
{
  switch(data_type)
  {
  case ECAT_Byte_data_type:
    type = NumericType("signed integer", 1);
    byte_order=ByteOrder::little_endian;
    return;
  case ECAT_I2_little_endian_data_type:
    type = NumericType("signed integer", 2);
    byte_order=ByteOrder::little_endian;
    return;
  case ECAT_I2_big_endian_data_type:
    type = NumericType("signed integer", 2);
    byte_order = ByteOrder::big_endian;
    return;
  case ECAT_R4_VAX_data_type:
    type = NumericType("float", 4);
    byte_order=ByteOrder::little_endian;
    return;
  case ECAT_R4_IEEE_big_endian_data_type:
    type = NumericType("float", 4);
    byte_order=ByteOrder::big_endian;
    return;
  case ECAT_I4_little_endian_data_type:
    type = NumericType("signed integer", 4);
    byte_order=ByteOrder::little_endian;
    return;
  case ECAT_I4_big_endian_data_type:
    type = NumericType("signed integer", 4);
    byte_order=ByteOrder::big_endian;
    return;    
  default:
    error("find_type_from_ecat_data_type: unsupported data_type: %d", data_type);
    // just to avoid compiler warnings
    return;
  }
}

short find_ECAT_data_type(const NumericType& type, const ByteOrder& byte_order)
{
  if (!type.signed_type())
    warning("find_ecat_data_type: ecat data support only signed types. Using the signed equivalent\n");
  if (type.integer_type())
  {
    switch(type.size_in_bytes())
    {
    case 1:
      return ECAT_Byte_data_type;
    case 2:
      return byte_order==ByteOrder::big_endian ? ECAT_I2_big_endian_data_type : ECAT_I2_little_endian_data_type;
    case 4:
      return byte_order==ByteOrder::big_endian ? ECAT_I4_big_endian_data_type : ECAT_I4_little_endian_data_type;
    default:
      {
        // write error message below
      }
    }
  }
  else
  {
    switch(type.size_in_bytes())
    {
    case 4:
      return byte_order==ByteOrder::big_endian ? ECAT_R4_IEEE_big_endian_data_type : ECAT_R4_VAX_data_type;
    default:
      {
        // write error message below
      }
    }
  }
  std::string number_format;
  std::size_t size_in_bytes;
  type.get_Interfile_info(number_format, size_in_bytes);
  error("find_ecat_data_type: ecat does not support data type '%s' of %d bytes.\n",
    number_format.c_str(), size_in_bytes);
  // just to satisfy compilers
  return short(0);
}

short find_ECAT_system_type(const Scanner& scanner)
{
  switch(scanner.get_type())
  {
  case Scanner::E921:
    return 921; 
  case Scanner::E925:
    return 925; 
    
  case Scanner::E931:
    return 931; 
    
  case Scanner::E951:
    return 951; 
    
  case Scanner::E953:
    return 953;

  case Scanner::E961:
    return 961;

  case Scanner::E962:
    return 962; 
    
  case Scanner::E966:
    return 966;

  case Scanner::RPT:
    return 128;

  case Scanner::RATPET:
    return 42;

  default:
    warning("\nfind_ecat_system_type: scanner \"%s\" currently unsupported. Returning 0.\n", 
      scanner.get_name().c_str());
    return 0;
  }
}

Scanner* find_scanner_from_ECAT_system_type(const short system_type)
{
  switch(system_type)
  {
  case 128 : 
    return new Scanner(Scanner::RPT);
  case 921 : 
    return new Scanner(Scanner::E921);
  case 925 : 
    return new Scanner(Scanner::E925);
  case 931 :
  case 12 : 
    return new Scanner(Scanner::E931);
  case 951 : 
    return new Scanner(Scanner::E951);
  case 953 : 
    return new Scanner(Scanner::E953);
  case 961 : 
    return new Scanner(Scanner::E961);
  case 962 : 
    return new Scanner(Scanner::E962);
  case 966 : 
    return new Scanner(Scanner::E966);
  case 42:
    return new Scanner(Scanner::RATPET);
  default :  
    return new Scanner(Scanner::Unknown_scanner);
  }
}

std::vector<int>
find_segment_sequence(const ProjDataInfo& pdi)
{
  const int max_segment_num = pdi.get_max_segment_num();
  std::vector<int> segment_sequence(2*max_segment_num+1);
  // KT 25/10/2000 swapped segment order
  // ECAT 7 always stores segments as 0, -1, +1, ...
  segment_sequence[0] = 0;
  for (int segment_num = 1; segment_num<=max_segment_num; ++segment_num)
  {
    segment_sequence[2*segment_num-1] = -segment_num;
    segment_sequence[2*segment_num] = segment_num;
  }
  return segment_sequence;
}

std::vector<int>
find_timing_poss_sequence(const ProjDataInfo& pdi)
{
  const int max_timing_pos_num = pdi.get_num_tof_poss()/2;
  std::vector<int> timing_pos_sequence(2*max_timing_pos_num+1);
  // Siemens always stores timing_poss as 0, -1, +1, ...
  timing_pos_sequence[0] = 0;
  for (int timing_pos_num = 1; timing_pos_num<=max_timing_pos_num; ++timing_pos_num)
  {
    timing_pos_sequence[2*timing_pos_num-1] = timing_pos_num;
    timing_pos_sequence[2*timing_pos_num] = -timing_pos_num;
  }
  return timing_pos_sequence;
}

END_NAMESPACE_ECAT
END_NAMESPACE_STIR
