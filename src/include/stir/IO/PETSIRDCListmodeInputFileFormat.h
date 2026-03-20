/*
        Copyright 2025, 2026 UMCG
        Copyright 2025 National Physical Laboratory

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
 */

/*!

  \file
  \ingroup listmode
  \brief Class defining input file format for coincidence listmode data for PETSIRD.

  \author Nikos Efthimiou
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
 */

class PETSIRDCListmodeInputFileFormat : public InputFileFormat<ListModeData>
{
public:
  const std::string get_name() const override { return "PETSIRD"; }

  //! Checks in binary data file for correct signature.
  bool can_read(const FileSignature& signature, const std::string& filename) const override;

protected:
  bool actual_can_read(const FileSignature& signature, std::istream& input) const override { return false; }

  mutable bool use_hdf5 = false;

public:
  unique_ptr<data_type> read_from_file(std::istream& input) const override
  {
    error("read_from_file for ROOT listmode data with istream not implemented %s:%s. Sorry", __FILE__, __LINE__);
    return unique_ptr<data_type>();
  }

  unique_ptr<data_type> read_from_file(const std::string& filename) const override
  {
    info("PETSIRDCListmodeInputFileFormat: read_from_file(" + std::string(filename) + ")");
    return unique_ptr<data_type>(new CListModeDataPETSIRD(filename, use_hdf5));
  }
};
END_NAMESPACE_STIR
#endif
