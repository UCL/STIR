/* PETSIRDCListmodeInputFileFormat.h

 Class defining input file format for coincidence listmode data for PETSIRD.

  Copyright 2025, 2026, University Medical Center Groningen
  Copyright 2025 National Physical Laboratory

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
  */
#include "stir/IO/PETSIRDCListmodeInputFileFormat.h"
#include "petsird/binary/protocols.h"
#include "petsird/hdf5/protocols.h"
#include "stir/error.h"
#include "stir/IO/HDF5_utilities.h"
#include "stir/format.h"
#include <array>

START_NAMESPACE_STIR

/*!

  \file
  \ingroup listmode
  \brief Implementation of class stir::PETSIRDCListmodeInputFileFormat

  \author Nikos Efthimiou
  \author Daniel Deidda

*/

bool
PETSIRDCListmodeInputFileFormat::can_read(const FileSignature& signature, const std::string& filename) const
{

  using HDF5_utilities::get_HDF5_group;
  using HDF5_utilities::is_valid;

  constexpr std::array<char, 5> binary_signature = { 'y', 'a', 'r', 'd', 'l' };

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    {
      error(format("Cannot open file: {}", filename));
      return false;
    }
  std::array<char, 5> signature_{};
  auto it = std::istreambuf_iterator<char>(file);
  auto end = std::istreambuf_iterator<char>();

  for (size_t i = 0; i < signature_.size() && it != end; ++i, ++it)
    {
      signature_[i] = *it;
    }

  if (!file)
    error("Stream error while reading file signature");

  if (signature_ == binary_signature)
    {
      use_hdf5 = false;
      return true;
    }

  // HDF5 signature can fail, so check reliably
  if (!H5::H5File::isHdf5(filename))
    return false;
  use_hdf5 = true;
  try
    {
      H5::H5File file;
      file.openFile(filename, H5F_ACC_RDONLY);
      auto group = get_HDF5_group(file, "PETSIRD");
      return is_valid(group) && group.attrExists("$yardl_format_version");
    }
  catch (...)
    {
      // it failed for some reason
      return false;
    }
}

END_NAMESPACE_STIR
