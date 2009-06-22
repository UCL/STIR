//
// $Id$
//
/*!
  \file
  \ingroup utilities

  \brief A utility that creates a template projection data.
  \par Usage
  \verbatim
  create_projdata_template output_filename
  \endverbatim
  This will ask questions to the user about the scanner, the data size,
  etc. It will then output new projection data (in Interfile format).
  However, the binary file will not contain any data.

  This utility is mainly useful to create a template that can then
  be used for other STIR utilities (such as fwdtest, lm_to_projdata etc.).

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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

#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfo.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif


USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{ 
  
  if(argc!=2) 
  {
    cerr<<"Usage: " << argv[0] << " output_filename\n";
    return EXIT_FAILURE;
  }


  shared_ptr<ProjDataInfo> proj_data_info_ptr = 
    ProjDataInfo::ask_parameters();
  
  const string output_file_name = argv[1];
  shared_ptr<ProjData> proj_data_ptr =
    new ProjDataInterfile(proj_data_info_ptr, output_file_name);

  return EXIT_SUCCESS;
}
