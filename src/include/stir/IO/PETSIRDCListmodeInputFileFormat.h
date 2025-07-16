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

// #include "boost/algorithm/string.hpp"

#include "stir/IO/InputFileFormat.h"
#include "stir/error.h"

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
  bool can_read(const FileSignature& signature, const std::string& filename) const override;

protected:
  bool actual_can_read(const FileSignature& signature, std::istream& input) const override
  {
    int nikos = 0;
    return true;
  }

public:
  unique_ptr<data_type> read_from_file(std::istream& input) const override
  {
    error("read_from_file for ROOT listmode data with istream not implemented %s:%s. Sorry", __FILE__, __LINE__);
    return unique_ptr<data_type>();
  }

  unique_ptr<data_type> read_from_file(const std::string& filename) const override
  {
    int nikos = 0;
    // return unique_ptr<data_type>(new CListModeDataPETSIRD(filename));
  }
};
END_NAMESPACE_STIR
#endif
