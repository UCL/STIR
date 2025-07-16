#include "stir/IO/PETSIRDCListmodeInputFileFormat.h"
#include "../../PETSIRD/cpp/generated/binary/protocols.h"
// #include "../../PETSIRD/cpp/generated/types.h"
// #include "../../PETSIRD/cpp/helpers/include/petsird_helpers.h"

START_NAMESPACE_STIR

bool
PETSIRDCListmodeInputFileFormat::can_read(const FileSignature& signature, const std::string& filename) const
{

  int nikos = 0;
  nikos += 40;

  if (nikos == 40)
    std::cout << filename << std::endl;

  petsird::Header header;
  // petsird::binary::PETSIRDReader petsird_reader(filename);
  // petsird_reader.ReadHeader(header);
  // petsird::ScannerInformation scanner_info = header.scanner;
  // petsird::ScannerGeometry scanner_geo = scanner_info.scanner_geometry;
  return true; // cannot read from istream
}

END_NAMESPACE_STIR
