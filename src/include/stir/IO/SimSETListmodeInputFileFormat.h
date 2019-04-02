/*
    Copyright (C) 2019 University of Hull
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
#ifndef __stir_IO_SimSETListmodeInputFileFormat_h__
#define __stir_IO_SimSETListmodeInputFileFormat_h__


#include "stir/IO/InputFileFormat.h"
#include "stir/listmode/CListModeDataSimSET.h"
#include "stir/error.h"
#include <iostream>
#include <unistd.h>

START_NAMESPACE_STIR

//!
//! \brief The SimSETListmodeInputFileFormat class
//! \details Class to check the file signature of SiSET history files.
//! These are listmode files.
//!
//! \ingroup SimSET
//!
//! The phg parameter file should be used here.
//! However, STIR passes only 1024 bytes. The solution is that the
//! first line of your phg file should always write the line
//!
//! # Hello, I am a SimSET PHG file!
//!
//! This will let us know that this is a PHG file.
//!
//!
//! \author Nikos Efthimiou
//!
class SimSETListmodeInputFileFormat :
        public InputFileFormat<CListModeData >
{
public:
    virtual const std::string
    get_name() const
    {
        return "SimSET_Histrory_file";
    }

    virtual unique_ptr<data_type>
    read_from_file(std::istream& input) const
    {
        error("read_from_file for SimSET listmode data with istream not implemented %s:%s. Sorry",
              __FILE__, __LINE__);
        return unique_ptr<data_type>();
    }

    virtual unique_ptr<data_type>
    read_from_file(const std::string& filename) const
    {
        return unique_ptr<data_type>(new CListModeDataSimSET(filename));
    }

protected:

    virtual
    bool actual_can_read(const FileSignature& signature,
                         std::istream& input) const
    {
        return this->is_SimSET_signature(input);
    }

    //! This is a very dirty function. We have to replicate argv in
    //! a linux terminal fashion, as that is the acceptabel input to SimSET
    //! function LbEnGetOptions. I could rewrite that, but the effort might
    //! be too much.
    bool is_SimSET_signature(std::istream& signature) const
    {
        std::string check("# Hello, I am a SimSET PHG file!");
        std::string firstLine;
        getline(signature, firstLine);
        if (firstLine == check)
            return true;
        else
            return false;
    }
};

END_NAMESPACE_STIR

#endif

// This is an alternative version of the is_SimSET_signature.
//        std::string argv_str;
//        std::getline(signature, argv_str);
//        char fileName[1024];

//        // Remove everything prior to := from the line.
//        argv_str = argv_str.substr(argv_str.find("=") + 1);
//        strcpy(fileName, argv_str.c_str());

//        char** argv = new char*[3];
//        argv[0] = nullptr;
//        argv[1] = nullptr;
//        argv[2] = nullptr;


//        char* pseudo_binary = new char[7];
//        memset(pseudo_binary, 0, 7);
//        strcpy(pseudo_binary, "phgbin\0");
//        argv[0] = pseudo_binary;

//        char* flag = new char[4];
//        memset(flag, 0, 4);
//        strcpy(flag, "-p");
//        flag[3] = '\0';
//        argv[1] = flag;

//        char *argv_c = new char[argv_str.size() + 1];
//        memset(argv_c, 0, argv_str.size()+1);
//        argv_str.copy(argv_c, argv_str.size());
//        argv_c[argv_str.size()] = '\0';
//        argv[2] = argv_c;

//        char *knownOptions = new char[4];
//        strcpy(knownOptions, "pcd");
//        knownOptions[3] = '\0';

//        char optArgs[PHGRDHST_NumFlags][LBEnMxArgLen];
//        LbUsFourByte optArgFlags = (LBFlag0);
//        LbUsFourByte phgrdhstArgIndex;

//        /* Get our options */
//        if (!LbEnGetOptions(3, argv, &knownOptions,
//                            &PhgOptions, optArgs, optArgFlags, &phgrdhstArgIndex)) {

//            // Clean up.
//            delete [] argv;
//            delete [] pseudo_binary;
//            delete [] argv_c;
//            delete [] flag;
//            delete [] knownOptions;
//            return false;
//        }

//        /* Make sure the didn't specify more than one history file */
//        if (PHGRDHST_IsUsePHGHistory() && (PHGRDHST_IsUseColHistory() || PHGRDHST_IsUseDetHistory()))
//        {
//            warning("SimSETListmodeInputFileFormat: You can only specify one type of history file.");
//            // Clean up.
//            delete [] argv;
//            delete [] pseudo_binary;
//            delete [] argv_c;
//            delete [] flag;
//            delete [] knownOptions;
//            return false;
//        }

//        // Clean up.
//        delete [] argv;
//        delete [] pseudo_binary;
//        delete [] argv_c;
//        delete [] flag;
//        delete [] knownOptions;

