//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief ProjMatrixByBinFromFile's definition 

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
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
  \ingroup recon_buildblock
  \brief Reads a projection matrix from file
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
  static Succeeded
    write(std::ostream&fst, const ProjMatrixElemsForOneBin& lor );
  static Succeeded
    read(std::istream&fst, ProjMatrixElemsForOneBin& lor );

  string parsed_version;
  string template_density_filename;
  string template_proj_data_filename;
  string data_filename;
  
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



