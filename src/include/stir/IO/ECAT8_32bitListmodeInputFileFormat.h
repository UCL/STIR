#ifndef __stir_IO_ECAT8_32bitListmodeInputFileFormat_h__
#define __stir_IO_ECAT8_32bitListmodeInputFileFormat_h__
/*
    Copyright (C) 2013 University College London
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
  \ingroup ECAT
  \brief Declaration of class stir::ecat::ECAT8_32bitListmodeInputFileFormat

  \author Kris Thielemans
*/
#include "stir/IO/InputFileFormat.h"
#include "stir/IO/interfile.h"
#include "stir/listmode/CListModeDataECAT8_32bit.h"
//#include "stir/listmode/CListRecordECAT8_32bit.h"

#include "stir/utilities.h"
#include "stir/error.h"
#include <string>

START_NAMESPACE_STIR
namespace ecat {

//! Class for being able to read list mode data from the ECAT 8_32bit scanner via the listmode-data registry.
/*! \ingroup ECAT
  \ingroup listmode
*/
class ECAT8_32bitListmodeInputFileFormat :
public InputFileFormat<CListModeData >
{
 public:
  virtual const std::string
    get_name() const
  {  return "ECAT8_32bit"; }

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const
  {
    // TODO need to do check that it's a siemens list file etc
    return is_interfile_signature(signature.get_signature());
  }
 public:
  virtual unique_ptr<data_type>
    read_from_file(std::istream& input) const
  {
    error("read_from_file for ECAT8_32bit listmode data with istream not implemented %s:%s. Sorry",
	  __FILE__, __LINE__);
    return
      unique_ptr<data_type>();
  }
  virtual unique_ptr<data_type>
    read_from_file(const std::string& filename) const
  {	
    return unique_ptr<data_type>(new CListModeDataECAT8_32bit(filename)); 
  }
};

} // namespace ecat
END_NAMESPACE_STIR

#endif
