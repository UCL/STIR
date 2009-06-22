//
// $Id$
//

/*!
  \file
  \ingroup utilities

  \brief Program that writes a projection matrix by bin to file

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

#include "stir/recon_buildblock/ProjMatrixByBinFromFile.h"
#include "stir/KeyParser.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjData.h"
// for ask_filename...
#include "stir/utilities.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/Coordinate3D.h"

#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cerr;
using std::endl;
#endif


int
main(int argc, char **argv)
{  
  USING_NAMESPACE_STIR
  if (argc==1 || argc>5)
  {
    cerr <<"Usage: " << argv[0] << " \\\n"
	 << "\toutput-filename [proj_data_file [projmatrixbybin-parfile [template-image]]]\n";
    exit(EXIT_FAILURE);
  }
  const string output_filename_prefix=
    argc>1? argv[1] : ask_string("Output filename prefix");
  
  shared_ptr<ProjDataInfo> proj_data_info_sptr;  
  if (argc>2)
    { 
      shared_ptr<ProjData> proj_data_sptr = ProjData::read_from_file(argv[2]);
      proj_data_info_sptr=proj_data_sptr->get_proj_data_info_ptr()->clone();
    }
  else
    {
      proj_data_info_sptr = ProjDataInfo::ask_parameters();
    }
  shared_ptr<ProjMatrixByBin> proj_matrix_sptr;

  if (argc>3)
    {
      KeyParser parser;
      parser.add_start_key("ProjMatrixByBin parameters");
      parser.add_parsing_key("type", &proj_matrix_sptr);
      parser.add_stop_key("END"); 
      parser.parse(argv[3]);
    }
 
  shared_ptr<DiscretisedDensity<3,float> > image_sptr;

  if (argc>4)
    {
      image_sptr = DiscretisedDensity<3,float>::read_from_file(argv[4]);
    }
  else
    {
      const float zoom = ask_num("Zoom factor (>1 means smaller voxels)",0.F,100.F,1.F);
      int xy_size = static_cast<int>(proj_data_info_sptr->get_num_tangential_poss()*zoom);
      xy_size = ask_num("Number of x,y pixels",3,xy_size*2,xy_size);
      int z_size = 2*proj_data_info_sptr->get_scanner_ptr()->get_num_rings()-1;
      z_size = ask_num("Number of z pixels",1,1000,z_size);
      VoxelsOnCartesianGrid<float> * vox_image_ptr =
	new VoxelsOnCartesianGrid<float>(*proj_data_info_sptr,
					 zoom,
					 Coordinate3D<float>(0,0,0),
					 Coordinate3D<int>(z_size,xy_size,xy_size));
      const float z_origin = 
	ask_num("Shift z-origin (in pixels)", 
		-vox_image_ptr->get_length()/2,
		vox_image_ptr->get_length()/2,
		0)
	*vox_image_ptr->get_voxel_size().z();
      vox_image_ptr->set_origin(Coordinate3D<float>(z_origin,0,0));

      image_sptr = vox_image_ptr;
    }

  while (is_null_ptr(proj_matrix_sptr))
    {
      proj_matrix_sptr =
	ProjMatrixByBin::ask_type_and_parameters();
    }

  proj_matrix_sptr->set_up(proj_data_info_sptr,
			   image_sptr);
 
  return
    ProjMatrixByBinFromFile::
    write_to_file(output_filename_prefix, 
		  *proj_matrix_sptr, 
		  proj_data_info_sptr,
		  *image_sptr) == Succeeded::yes ?
    EXIT_SUCCESS : EXIT_FAILURE;
}

//cache_proj_matrix_elems_for_one_bin
