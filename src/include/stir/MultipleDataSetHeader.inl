/*
    Copyright (C) 2018, 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup data_buildblock
  \brief Inline implementations of class stir::MultipleDataSetHeader
  \author Richard Brown
  \author Kris Thielemans
*/

#include <fstream>

START_NAMESPACE_STIR


template <class VectorOfStringsT>
void
MultipleDataSetHeader::
write_header(const std::string& filename, const VectorOfStringsT& individual_filenames)
{
  std::ofstream multi_file(filename.c_str());
  if (!multi_file.is_open())
    error("MultiDynamicDiscretisedDensity error: Failed to write \"" + filename + "\".\n");

  multi_file << "Multi :=\n";
  multi_file << "\ttotal number of data sets := " << individual_filenames.size() << "\n";
  int i=1;
  for (auto&& individual_filename : individual_filenames)
    {
      multi_file << "\tdata set["<<i<<"] := "<<individual_filename <<"\n";
      ++i;
    }
  multi_file << "end :=\n";
  multi_file.close();
}

END_NAMESPACE_STIR
