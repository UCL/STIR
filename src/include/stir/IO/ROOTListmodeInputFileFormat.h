/*
    Copyright (C) 2015-2016 University of Leeds
    Copyright (C) 2016 UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_IO_ROOTListmodeInputFileFormat_h__
#define __stir_IO_ROOTListmodeInputFileFormat_h__

#include "stir/IO/InputFileFormat.h"
#include "stir/listmode/CListModeDataROOT.h"
#include "stir/interfile_keyword_functions.h"
#include "stir/error.h"
#include "stir/utilities.h"
#include <string>

START_NAMESPACE_STIR

//!
//! \brief The ROOTListmodeInputFileFormat class
//! \details Class for being able to read list mode data from the ROOT via the listmode-data registry.
//! \author Nikos Efthimiou
//!
class ROOTListmodeInputFileFormat : public InputFileFormat<ListModeData>
{
public:
  const std::string get_name() const override { return "ROOT"; }

protected:
  bool actual_can_read(const FileSignature& signature, std::istream& input) const override
  {
    return this->is_root_signature(signature.get_signature());
  }

  bool is_root_signature(const char* const signature) const
  {
    // checking for "interfile :"
    const char* pos_of_colon = strchr(signature, ':');
    if (pos_of_colon == NULL)
      return false;
    std::string keyword(signature, pos_of_colon - signature);
    return (standardise_interfile_keyword(keyword) == standardise_interfile_keyword("ROOT header"));
  }

public:
  unique_ptr<data_type> read_from_file(std::istream& input) const override
  {
    error("read_from_file for ROOT listmode data with istream not implemented %s:%s. Sorry", __FILE__, __LINE__);
    return unique_ptr<data_type>();
  }

  unique_ptr<data_type> read_from_file(const std::string& filename) const override
  {
    return unique_ptr<data_type>(new CListModeDataROOT(filename));
  }
};

END_NAMESPACE_STIR

#endif
