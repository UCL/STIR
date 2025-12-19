/* PETSIRDCListmodeInputFileFormat.h

 Class defining input file format for coincidence listmode data for PETSIRD.

        Copyright 2025, UMCG
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

  \author Nikos Efthimiou
  \author Daniel Deidda

*/

#include "stir/IO/PETSIRDCListmodeInputFileFormat.h"
#include "petsird/binary/protocols.h"
#include "petsird/hdf5/protocols.h"
// #include "../../PETSIRD/cpp/generated/types.h"
// #include "../../PETSIRD/cpp/helpers/include/petsird_helpers.h"

START_NAMESPACE_STIR

bool
PETSIRDCListmodeInputFileFormat::can_read(const FileSignature& signature, const std::string& filename)
{

  std::array<char, 4> hdf5_signature = { 'H', 'D', 'F', '5' };
  std::array<char, 4> binary_signature = { 'y', 'a', 'r', 'd' };

  if (signature_.empty())
    error("Internal error: signature buffer is empty");

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    {
      std::cerr << "Cannot open file: " << filename << std::endl;
      return false;
    }
  std::array<char, 4> signature_{};
  auto it = std::isstreambuf_iterator<char>(file);
  auto end = std::isstreambuf_iterator<char>();

  for (size_t i = 0; i < signature.size() && it != end; ++i, ++it)
    {
      if (it == end)
        error("Failed to read file signature: unexpected EOF");
      signature_[i] = *it;
    }

  // codacy:ignore CWE-120 CWE-20
  // Fixed-size preallocated buffer, bounded read, checked gcount()
  file.read(signature_.data(), static_cast<std::streamsize>(signature_.size()));

  if (!file || file.gcount() != static_cast<std::streamsize>(signature_.size()))
    {
      error("Failed to read file signature: unexpected EOF or read error");
    }

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
