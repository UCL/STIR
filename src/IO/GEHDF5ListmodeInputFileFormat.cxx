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

#include <string>
#include "H5Cpp.h"

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
GEHDF5ListmodeInputFileFormat::can_read(const FileSignature& signature, const std::string& filename) const
{
  // check that it's a GE HDF5 list file etc
  try
    {
      H5::H5File file;
      file.openFile(filename, H5F_ACC_RDONLY);
      /*
    std::string read_str_scanner;
  std::string read_str_manufacturer;


  H5::DataSet dataset=this->file.openDataSet("/HeaderData/ExamData/scannerDesc");
  H5::DataSet dataset2=this->file.openDataSet("/HeaderData/ExamData/manufacturer");

  dataset.read(read_str_scanner,vlst);
  std::cout << "\n Scanner :  " << read_str_scanner << "\n\n";

  dataset2.read(read_str_manufacturer,vlst);
  std::cout << "\n Manufacturer :  " << read_str_manufacturer << "\n\n";
      */
      return true;
    }
  catch (...)
    {
      // it failed for some reason
      return false;
    }
}

std::unique_ptr<ListModeData>
GEHDF5ListmodeInputFileFormat::read_from_file(std::istream& input) const
{
  warning("read_from_file for GEHDF5 listmode data with istream not implemented %s:%s. Sorry", __FILE__, __LINE__);
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
