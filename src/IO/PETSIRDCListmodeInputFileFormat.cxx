#include "stir/IO/PETSIRDCListmodeInputFileFormat.h"
#include "../../PETSIRD/cpp/generated/binary/protocols.h"
#include "../../PETSIRD/cpp/generated/hdf5/protocols.h"
// #include "../../PETSIRD/cpp/generated/types.h"
// #include "../../PETSIRD/cpp/helpers/include/petsird_helpers.h"

START_NAMESPACE_STIR

bool
PETSIRDCListmodeInputFileFormat::can_read(const FileSignature& signature, const std::string& filename) const
{

  petsird::hdf5::PETSIRDReader* petsird_reader = new petsird::hdf5::PETSIRDReader(filename);

  if(is_null_ptr(petsird_reader))
    {

      petsird::binary::PETSIRDReader* petsird_reader = new petsird::binary::PETSIRDReader(filename);
      if(is_null_ptr(petsird_reader))
        {
          return false;
        }
      return true;
    }

  return true;
}

END_NAMESPACE_STIR
