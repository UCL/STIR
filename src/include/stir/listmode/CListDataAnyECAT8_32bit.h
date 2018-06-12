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
  \brief Definition of CListDataAnyECAT8_32bit for listmode events for the ECAT 8 format
    
  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#ifndef __stir_listmode_CListDataAnyECAT8_32bit_H__
#define __stir_listmode_CListDataAnyECAT8_32bit_H__

#include "stir/listmode/CListTimeDataECAT8_32bit.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR
namespace ecat {

    /*!
      \class
      \ingroup listmode
      \brief Definition of CListDataAnyECAT8_32bit for listmode events for the ECAT 8 format

      \author Kris Thielemans
    */
class CListDataAnyECAT8_32bit
{
public:
 Succeeded init_from_data_ptr(const void * const ptr)
    {
      const char * const data_ptr = reinterpret_cast<const char * const >(ptr);
      std::copy(data_ptr, data_ptr+sizeof(this->raw), reinterpret_cast<char *>(&this->raw));
      return Succeeded::yes;
    }
  bool is_time() const
  { return this->data.type == 1U && this->data.deadtimeetc == 0U; }
  bool is_other() const
  { return this->data.type == 1U && this->data.deadtimeetc != 0U; }
  bool is_event() const
  { return this->data.type == 0U; }


 private:
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT8_32bit)==4); 
  union 
  {
    CListTimeDataECAT8_32bit   data;
    boost::int32_t         raw;
  };
}; 

} // namespace ecat
END_NAMESPACE_STIR

#endif
