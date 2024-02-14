#ifndef __stir_IO_GEHDF5ListmodeInputFileFormat_h__
#define __stir_IO_GEHDF5ListmodeInputFileFormat_h__
/*
    Copyright (C) 2016-2019 University College London
    Copyright (C) 2017-2019 University of Leeds
    Copyright (C) 2017-2019 University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup IO
  \ingroup GE
  \brief Declaration of class stir::GE::RDF_HDF5::IO::GEHDF5ListmodeInputFileFormat

  \author Kris Thielemans
  \author Ottavia Bertolli
  \author Palak Wadhwa
  \author Nikos Efthimiou
*/
#include "stir/IO/InputFileFormat.h"

START_NAMESPACE_STIR

namespace GE
{
namespace RDF_HDF5
{

//! Class for being able to read list mode data from the GE Signa PET/MR scanner via the listmode-data registry.
/*!
  \ingroup listmode
  \ingroup IO
  \ingroup GE
*/
class GEHDF5ListmodeInputFileFormat : public InputFileFormat<ListModeData>
{
public:
  const std::string get_name() const override { return "GEHDF5"; }

protected:
  bool actual_can_read(const FileSignature& signature, std::istream& input) const override;
  bool can_read(const FileSignature& signature, const std::string& filename) const override;

public:
  unique_ptr<data_type> read_from_file(std::istream& input) const override;
  unique_ptr<data_type> read_from_file(const std::string& filename) const override;
};

} // namespace RDF_HDF5
} // namespace GE
END_NAMESPACE_STIR

#endif
