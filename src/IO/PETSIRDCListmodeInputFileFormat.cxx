/* PETSIRDCListmodeInputFileFormat.h

 Class defining input file format for coincidence listmode data for PETSIRD.

  Copyright 2025, 2026 UMCG
  Copyright 2025 National Physical Laboratory

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
/*!

  \file
  \ingroup listmode
  \brief Implementation of class stir::PETSIRDCListmodeInputFileFormat

  \author Nikos Efthimiou
  \author Daniel Deidda

*/

#include "stir/IO/PETSIRDCListmodeInputFileFormat.h"
#include "petsird/binary/protocols.h"
#include "petsird/hdf5/protocols.h"
#include "stir/error.h"
#include "stir/format.h"
#include <array>
// #include "../../PETSIRD/cpp/generated/types.h"
// #include "../../PETSIRD/cpp/helpers/include/petsird_helpers.h"

START_NAMESPACE_STIR

bool
PETSIRDCListmodeInputFileFormat::can_read(const FileSignature& signature, const std::string& filename) const
{

  std::array<char, 4> hdf5_signature = { 'H', 'D', 'F', '5' };
  std::array<char, 4> binary_signature = { 'y', 'a', 'r', 'd' };

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    {
      error(format("Cannot open file: {}", filename));
      return false;
    }
  std::array<char, 4> signature_{};
  auto it = std::istreambuf_iterator<char>(file);
  auto end = std::istreambuf_iterator<char>();

  for (size_t i = 0; i < signature_.size() && it != end; ++i, ++it)
    {
      signature_[i] = *it;
    }

  if (!file)
    error("Stream error while reading file signature");

  if (signature_ == hdf5_signature)
    {
      use_hdf5 = true;
      return use_hdf5;
    }

  if (signature_ == binary_signature)
    {
      use_hdf5 = false;
      return true;
    }

  // petsird::hdf5::PETSIRDReader* petsird_reader = new petsird::hdf5::PETSIRDReader(filename);

  // if (is_null_ptr(petsird_reader))
  //   {

  //     petsird::binary::PETSIRDReader* petsird_reader = new petsird::binary::PETSIRDReader(filename);
  //     if (is_null_ptr(petsird_reader))
  //       {
  //         return false;
  //       }
  //     return true;
  //   }

  return false;
}

END_NAMESPACE_STIR
