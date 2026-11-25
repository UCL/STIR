/*
    Copyright (C) 2016-2019, 2023 University College London
    Copyright (C) 2017-2019 University of Leeds
    Copyright (C) 2017-2019 University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup listmode
  \ingroup IO
  \ingroup GE
  \brief Implementations of class stir::GE::RDF_HDF5::IO::GEHDF5ListmodeInputFileFormat

  \author Kris Thielemans
  \author Ottavia Bertolli
  \author Palak Wadhwa
  \author Nikos Efthimiou
*/
#include "stir/error.h"
#include "stir/warning.h"
#include "stir/listmode/CListModeDataGEHDF5.h"
#include "stir/IO/GEHDF5ListmodeInputFileFormat.h"
#include "stir/IO/GEHDF5Wrapper.h"

#include <string>
START_NAMESPACE_STIR

namespace GE
{
namespace RDF_HDF5
{
bool
GEHDF5ListmodeInputFileFormat::actual_can_read(const FileSignature& signature, std::istream& input) const
{
  error("Cannot read from stream");
  return false;
}

bool
GEHDF5ListmodeInputFileFormat::can_read(const FileSignature&, const std::string& filename) const
{
  return GEHDF5Wrapper::check_GE_signature(filename);
}

std::unique_ptr<ListModeData>
GEHDF5ListmodeInputFileFormat::read_from_file(std::istream& input) const
{
  warning("read_from_file for GEHDF5 listmode data with istream not implemented. Sorry");
  return std::unique_ptr<ListModeData>();
}
std::unique_ptr<ListModeData>
GEHDF5ListmodeInputFileFormat::read_from_file(const std::string& filename) const
{
  return std::unique_ptr<ListModeData>(new CListModeDataGEHDF5(filename));
}

} // namespace RDF_HDF5

} // namespace GE
END_NAMESPACE_STIR
