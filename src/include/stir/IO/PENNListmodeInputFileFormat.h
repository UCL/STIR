/*
    Copyright (C) 2020-2022 University of Pennsylvania
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
#ifndef __stir_IO_PENNListmodeInputFileFormat_h__
#define __stir_IO_PENNListmodeInputFileFormat_h__


#include "stir/IO/InputFileFormat.h"
#include "stir/listmode/CListModeDataPENN.h"
#include "stir/error.h"
#include "stir/utilities.h"
#include <string>

START_NAMESPACE_STIR

//!
//! \brief The ROOTListmodeInputFileFormat class
//! \details Class for being able to read list mode data from the ROOT via the listmode-data registry.
//! \author Nikos Efthimiou
//!
class PENNListmodeInputFileFormat :
        public InputFileFormat<ListModeData >
{
public:
    virtual const std::string
    get_name() const
    {  return "PENN"; }

protected:

    virtual
    bool
    actual_can_read(const FileSignature& signature,
                    std::istream& input) const
    {
        return this->is_penn_signature(signature.get_signature());
    }

    bool is_penn_signature(const char* const signature) const
    {
        // checking for txt file
        const char * pos_of_colon = strchr(signature, ':');
        if (pos_of_colon == NULL)
          return false;
        std::string keyword(signature, pos_of_colon-signature);
        return (
            standardise_interfile_keyword(keyword) ==
            standardise_interfile_keyword("PENN header"));
    }

public:
    virtual unique_ptr<data_type>
    read_from_file(std::istream& input) const
    {
        error("read_from_file for PENN listmode data with istream not implemented %s:%s. Sorry",
                __FILE__, __LINE__);
        return unique_ptr<data_type>();
    }

    virtual unique_ptr<data_type>
    read_from_file(const std::string& filename) const
    {
        return unique_ptr<data_type>(new CListModeDataPENN(filename));
    }
};

END_NAMESPACE_STIR

#endif
