#ifndef __stir_IO_ECAT966ListmodeInputFileFormat_h__
#define __stir_IO_ECAT966ListmodeInputFileFormat_h__
/*
    Copyright (C) 2011, Hammersmith Imanet Ltd
    Copyright (C) 2013-2014, University College London
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
  \brief Declaration of class stir::ecat::ecat7::ECAT966ListmodeInputFileFormat

  \author Kris Thielemans
*/
#include "stir/IO/InputFileFormat.h"
#include "stir/listmode/CListModeDataECAT.h"
#include "stir/listmode/CListRecordECAT966.h"

#include "stir/utilities.h"
#include "stir/info.h"
#include "stir/error.h"
#include <string>
#include <boost/format.hpp>

#ifndef HAVE_LLN_MATRIX
#error HAVE_LLN_MATRIX not define: you need the lln ecat library.
#endif

#include "stir/IO/stir_ecat7.h"
START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! Class for reading list mode data from the ECAT 966 scanner
/*! \ingroup ECAT
  \ingroup listmode

  ECAT7 list mode data are recorded in the following files:
  - <tt>PREFIX_1.sgl</tt>: contains an ECAT7 main header and the singles counts
  - <tt>PREFIX_1.lm</tt>: contains the coincidence events (max size  is 2GB)
  - <tt>PREFIX_1.sgl</tt>: contains the next chunk of coincidence events
  - ...

  This class expects to be passed the name of the .sgl file.
*/
class ECAT966ListmodeInputFileFormat :
public InputFileFormat<CListModeData >
{
 public:
  virtual const std::string
    get_name() const
  {  return "ECAT966"; }

  //! Always return false as ECAT7 IO cannot read from stream
  virtual bool
    can_read(const FileSignature& signature,
	     std::istream& input) const
  {
    return this->actual_can_read(signature, input);
  }

  //! Checks if it's an ECAT7 file by reading the main header and if the scanner is supported. */
  virtual bool 
    can_read(const FileSignature& signature,
	     const std::string&  singles_filename) const
  {
    if (strncmp(signature.get_signature(), "MATRIX", 6) != 0)
      return false;

    //const string singles_filename = listmode_filename_prefix + "_1.sgl";
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
    if (scanner_sptr->get_type() == Scanner::E966)
      return true;

    return false;
  }

 protected:
  //! Always return false as ECAT7 IO cannot read from stream
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const
  {
    warning("can_read for ECAT966 listmode data with istream not implemented %s:%d. Sorry",
	  __FILE__, __LINE__);
    return false;

    if (strncmp(signature.get_signature(), "MATRIX", 6) != 0)
      return false;

    // TODO
    // return (is_ECAT7_image_file(filename))
    return true;
  }
 public:
  virtual std::unique_ptr<data_type>
    read_from_file(std::istream& input) const
  {
    // cannot do this as need both .sgl and .lm
    error("read_from_file for ECAT966 listmode data with istream not implemented %s:%d. Sorry",
	  __FILE__, __LINE__);
    return
      std::unique_ptr<data_type>();
  }
  //! read the data via the .sgl file
  /*! We first remove the suffix (either .sgl or _1.sgl) and then call ecat::ecat7::CListModeDataECAT::CListModeDataECAT(const std::string&)
  */
  virtual std::unique_ptr<data_type>
    read_from_file(const std::string& filename) const
  {	
    // filename points to the .sgl file, but we need the prefix
    std::string::size_type pos = find_pos_of_extension(filename);
    // also remove _1 at the end (if present)
    if (pos != std::string::npos && pos>2 && filename.substr(pos-2,2)=="_1")
      {
        pos-=2;
      }
    const std::string filename_prefix = filename.substr(0, pos);
    info(boost::format("Reading ECAT listmode file with prefix %1%") % filename_prefix);

    return std::unique_ptr<data_type>(new ecat::ecat7::CListModeDataECAT<ecat::ecat7::CListRecordECAT966>(filename_prefix)); 
  }
};

END_NAMESPACE_ECAT
END_NAMESPACE_ECAT7
END_NAMESPACE_STIR

#endif
