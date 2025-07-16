/* PETSIRDCListmodeInputFileFormat.h

 Class defining input file format for coincidence listmode data for PETSIRD.

        Copyright 2015 ETH Zurich, Institute of Particle Physics
        Copyright 2020 Positrigo AG, Zurich
        Copyright 2025 National Physical Laboratory

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
 */

/*!

  \file
  \ingroup listmode
  \brief Declaration of class stir::PETSIRDCListmodeInputFileFormat

  \author Jannis Fischer
  \author Markus Jehl, Positrigo
  \author Daniel Deidda

*/

#ifndef __stir_IO_PETSIRDCListmodeInputFileFormat_H__
#define __stir_IO_PETSIRDCListmodeInputFileFormat_H__

#include <cstring>
#include <string>
#include <iostream>

#include "boost/algorithm/string.hpp"

#include "stir/IO/InputFileFormat.h"
#include "stir/IO/InputFileFormat.h"
#include "stir/error.h"

// #include "stir/listmode/CListRecordPETSIRD.h"
#include "stir/listmode/CListModeDataPETSIRD.h"

START_NAMESPACE_STIR

/*! \brief Class for reading PETSIRD coincidence listmode data.

  The first 32 bytes of the binary file are interpreted as file signature and matched against the strings "MUPET CListModeData\0",
"PETSIRD". If either is successfull, the class claims it can read the file format. The
rest of the file is read as records, e.g. CListRecordPETSIRD.
*/

class PETSIRDCListmodeInputFileFormat : public InputFileFormat<ListModeData>
{
public:
  const std::string get_name() const override { return "PETSIRD"; }

  //! Checks in binary data file for correct signature.
  bool can_read(const FileSignature& signature, std::istream& input) const override
  {
    return false; // cannot read from istream
  }

  //! Checks in binary data file for correct signature (can be either "PETSIRD CListModeData", "NeuroLF CListModeData" or "MUPET
  //! CListModeData").
  bool can_read(const FileSignature& signature, const std::string& filename) const override
  {
    int nikos = 0;
    std::string d = filename;
    // PETSIRDReader ndn(d);
    // // Looking for the right key in the parameter file
    // std::ifstream par_file(filename.c_str());
    // std::string key;
    // std::getline(par_file, key, ':');
    // key = standardise_interfile_keyword(key);
    // if (key != std::string("clistmodedataPETSIRD parameters"))
    //   {
    //     return false;
    //   }
    // if (!actual_do_parsing(filename))
    //   return false;
    // std::ifstream data_file(listmode_filename.c_str(), std::ios::binary);
    // char* buffer = new char[32];
    // data_file.read(buffer, 32);
    // bool cr = false;
    // // depending on used template, check header of listmode file for correct format
    // if (std::is_same<EventDataType, CListEventDataPETSIRD>::value)
    //   {
    //     cr = (!strncmp(buffer, "MUPET CListModeData\0", 20) || !strncmp(buffer, "PETSIRD CListModeData\0", 20));
    //   }
    // else if (std::is_same<EventDataType, CListEventDataNeuroLF>::value)
    //   {
    //     cr = !strncmp(buffer, "NeuroLF CListModeData\0", 20);
    //   }
    // else
    //   {
    //     warning("PETSIRDCListModeInputFileFormat was initialised with an unexpected template.");
    //   }

    // if (!cr)
    //   {
    //     warning("PETSIRDCListModeInputFileFormat tried to read file " + listmode_filename
    //             + " but it seems to have the wrong signature.");
    //   }

    // delete[] buffer;
    // return cr;
    return true;
  }

  std::unique_ptr<data_type> read_from_file(std::istream& input) const override
  {
    error("read_from_file for PETSIRDCListmodeData with istream not implemented %s:%d. Sorry", __FILE__, __LINE__);
    return unique_ptr<data_type>();
  }

  std::unique_ptr<data_type> read_from_file(const std::string& filename) const override {}
};
END_NAMESPACE_STIR
#endif
