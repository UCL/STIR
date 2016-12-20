//
//
#ifndef __stir_IO_ECAT962ListmodeInputFileFormat_h__
#define __stir_IO_ECAT962ListmodeInputFileFormat_h__
/*
    Copyright (C) 2006- 2013, Hammersmith Imanet Ltd
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
  \brief Declaration of class stir::ecat::ecat7::ECAT962ListmodeInputFileFormat

  \author Kris Thielemans

*/
#include "stir/IO/InputFileFormat.h"
#include "stir/listmode/CListModeDataECAT.h"
#include "stir/listmode/CListRecordECAT962.h"

#include "stir/error.h"
#include "stir/utilities.h"
#include <string>


#ifndef HAVE_LLN_MATRIX
#error HAVE_LLN_MATRIX not define: you need the lln ecat library.
#endif

#include "stir/IO/stir_ecat7.h"
START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! Class for reading list mode data from the ECAT 962 scanner
/*! \ingroup ECAT
  \ingroup listmode
*/
class ECAT962ListmodeInputFileFormat :
public InputFileFormat<CListModeData >
{
 public:
  virtual const std::string
    get_name() const
  {  return "ECAT962"; }

  virtual bool
    can_read(const FileSignature& signature,
	     std::istream& input) const
  {
    return this->actual_can_read(signature, input);
  }
  virtual bool 
    can_read(const FileSignature& signature,
	     const std::string&  listmode_filename_prefix) const
  {
    const std::string singles_filename = listmode_filename_prefix + "_1.sgl";
    std::ifstream singles_file(singles_filename.c_str(), std::ios::binary);
    char buffer[sizeof(Main_header)];
    Main_header singles_main_header;
    singles_file.read(buffer,
                      sizeof(singles_main_header));
    if (!singles_file)
        return false;
    unmap_main_header(buffer, &singles_main_header);
    shared_ptr<Scanner> scanner_sptr;
    ecat::ecat7::find_scanner(scanner_sptr, singles_main_header);
    if (scanner_sptr->get_type() == Scanner::E962)
      return false;

    return true;
  }

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const
  {
    warning("can_read for ECAT962 listmode data with istream not implemented %s:%d. Sorry",
	  __FILE__, __LINE__);
    return false;

    if (strncmp(signature.get_signature(), "MATRIX", 6) != 0)
      return false;

    // TODO
    // return (is_ECAT7_image_file(filename))
    return true;
  }
 public:
  virtual unique_ptr<data_type>
    read_from_file(std::istream& input) const
  {
    // cannot do this as need both .sgl and .lm
    error("read_from_file for ECAT962 listmode data with istream not implemented %s:%d. Sorry",
	  __FILE__, __LINE__);
    return
      unique_ptr<data_type>();
  }
  virtual unique_ptr<data_type>
    read_from_file(const std::string& filename) const
  {	
    return unique_ptr<data_type>(new ecat::ecat7::CListModeDataECAT<ecat::ecat7::CListRecordECAT962>(filename)); 
  }
};

END_NAMESPACE_ECAT
END_NAMESPACE_ECAT7
END_NAMESPACE_STIR

#endif
