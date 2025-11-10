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

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    {
      std::cerr << "Cannot open file: " << filename << std::endl;
      return false;
    }

  std::array<char, 4> signature_{};
  file.read(signature_.data(), signature_.size());

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
