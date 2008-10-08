//
// $Id$
//
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
/*!
  \file
  \ingroup projection

  \brief stir::ProjMatrixByBinFromFile's definition 

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#ifndef __stir_recon_buildblock_ProjMatrixByBinFromFile__
#define __stir_recon_buildblock_ProjMatrixByBinFromFile__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/ProjDataInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange.h"
#include "stir/shared_ptr.h"
#include <iostream>

 

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Bin;
/*!
  \ingroup projection
  \brief Reads/writes a projection matrix from/to file

  The only supported file format consists of an Interfile-type header
  and a binary file which stores the 'basic' elements in a sparse form, 
  i.e. only the elements that cannot by constructed via symmetries.

  \todo this class currently only works with VoxelsOnCartesianGrid. 
  To fix this, we would need a DiscretisedDensityInfo class, and be able
  to have constructed the appropriate symmetries object by parsing the
  .par file

  \par Example .par file
  \verbatim
    ProjMatrixByBinFromFile Parameters:=
      Version := 1.0
      symmetries type := PET_CartesianGrid
        PET_CartesianGrid symmetries parameters:=
	  do_symmetry_90degrees_min_phi:= <bool>
	  do_symmetry_180degrees_min_phi:=<bool>
	  do_symmetry_swap_segment:= <bool>
	  do_symmetry_swap_s:= <bool>
	  do_symmetry_shift_z:= <bool>
	End PET_CartesianGrid symmetries parameters:=
      ; example projection data of the same dimensions as used when constructing the matrix
      template proj data filename:= <filename>
      ; example image of the same dimensions as used when constructing the matrix
      template density filename:= <filename>
      ; binary data with projection matrix elements
      data_filename:=<filename> 
     End ProjMatrixByBinFromFile Parameters:=
  \endverbatim
*/

class ProjMatrixByBinFromFile : 
  public RegisteredParsingObject<
	      ProjMatrixByBinFromFile,
              ProjMatrixByBin,
              ProjMatrixByBin
	       >
{
public :
  //! Name which will be used when parsing a ProjMatrixByBin object
  static const char * const registered_name; 
 
  //! Writes a projection matrix to file in a format such that this class can read it back
  /*! Currently this will write an interfile-type header, a file with the binary data,
      a template image and template sinogram. You will need all 4 to be able to read the
      matrix back in.
  */
static Succeeded
  write_to_file(const string& output_filename_prefix, 
		const ProjMatrixByBin& proj_matrix,
		const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
		const DiscretisedDensity<3,float>& template_density);
 
  //! Default constructor (calls set_defaults())
  ProjMatrixByBinFromFile();

  //! Checks all necessary geometric info
  virtual void set_up(		 
		      const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );

private:

  string parsed_version;
  string template_density_filename;
  string template_proj_data_filename;
  string data_filename;
  
  string symmetries_type;
  // should be in symmetries
  bool do_symmetry_90degrees_min_phi;
  bool do_symmetry_180degrees_min_phi;
  bool do_symmetry_swap_segment;
  bool do_symmetry_swap_s;
  bool do_symmetry_shift_z;

  // TODO this only works as long as we only have VoxelsOnCartesianGrid
  // explicitly list necessary members for image details (should use an Info object instead)
  CartesianCoordinate3D<float> voxel_size;
  CartesianCoordinate3D<float> origin;  
  IndexRange<3> densel_range;


  shared_ptr<ProjDataInfo> proj_data_info_ptr;


  virtual void 
    calculate_proj_matrix_elems_for_one_bin(
                                            ProjMatrixElemsForOneBin&) const;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  Succeeded read_data();
    
};

END_NAMESPACE_STIR

#endif



