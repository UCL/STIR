/*
    Copyright (C) 2018, 2020, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

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
