//
// $Id: ECAT8_32bitListmodeInputFileFormat.h,v 1.1 2011-06-28 14:46:08 kris Exp $
//
#ifndef __UCL_IO_ECAT8_32bitListmodeInputFileFormat_h__
#define __UCL_IO_ECAT8_32bitListmodeInputFileFormat_h__
/*
    Copyright (C) 2013 University College London
*/
/*!

  \file
  \ingroup ECAT
  \brief Declaration of class stir::UCL::ECAT8_32bitListmodeInputFileFormat

  \author Kris Thielemans

  $Date: 2011-06-28 14:46:08 $
  $Revision: 1.1 $
*/
#include "stir/IO/InputFileFormat.h"
#include "stir/IO/interfile.h"
#include "UCL/listmode/CListModeDataECAT8_32bit.h"
//#include "UCL/listmode/CListRecordECAT8_32bit.h"

#include "stir/utilities.h"
#include <string>

START_NAMESPACE_STIR
namespace UCL {

//! Class for reading list mode data from the ECAT 8_32bit scanner
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
  virtual std::auto_ptr<data_type>
    read_from_file(std::istream& input) const
  {
    warning("read_from_file for ECAT8_32bit listmode data with istream not implemented %s:%s. Sorry",
	  __FILE__, __LINE__);
    return
      std::auto_ptr<data_type>
      (0);
  }
  virtual std::auto_ptr<data_type>
    read_from_file(const std::string& filename) const
  {	
    return std::auto_ptr<data_type>(new CListModeDataECAT8_32bit(filename)); 
  }
};
} // namespace UCL
END_NAMESPACE_STIR

#endif
