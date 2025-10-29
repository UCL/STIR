#ifndef __stir_IO_ECAT8_32bitListmodeInputFileFormat_h__
#define __stir_IO_ECAT8_32bitListmodeInputFileFormat_h__
/*
    Copyright (C) 2013 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
namespace ecat
{

//! Class for being able to read list mode data from the ECAT 8_32bit scanner via the listmode-data registry.
/*! \ingroup ECAT
  \ingroup listmode
*/
class ECAT8_32bitListmodeInputFileFormat : public InputFileFormat<ListModeData>
{
public:
  const std::string get_name() const override { return "ECAT8_32bit"; }

protected:
  bool actual_can_read(const FileSignature& signature, std::istream& input) const override
  {
    if (!is_interfile_signature(signature.get_signature()))
      return false;
    else
      {
        const std::string signature_as_string(signature.get_signature(), signature.size());
        return signature_as_string.find("PETLINK") != std::string::npos;
      }
  }

public:
  unique_ptr<data_type> read_from_file(std::istream& input) const override
  {
    error("read_from_file for ECAT8_32bit listmode data with istream not implemented %s:%s. Sorry", __FILE__, __LINE__);
    return unique_ptr<data_type>();
  }
  unique_ptr<data_type> read_from_file(const std::string& filename) const override
  {
    return unique_ptr<data_type>(new CListModeDataECAT8_32bit(filename));
  }
};

} // namespace ecat
END_NAMESPACE_STIR

#endif
