//
//
/*!
  \file
  \ingroup recon_buildblock

  \brief ProjMatrixByBinWithPositronRange's definition 

  \author Kris 

*/
/*
    Copyright (C) 2000- 2004, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_ProjMatrixByBinWithPositronRange__
#define __stir_recon_buildblock_ProjMatrixByBinWithPositronRange__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/ProjDataInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/shared_ptr.h"

 

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup recon_buildblock
  \brief Computes projection matrix elements for VoxelsOnCartesianGrid images
  by using a Solid Angle model. 

*/

class ProjMatrixByBinWithPositronRange : 
  public RegisteredParsingObject<
	      ProjMatrixByBinWithPositronRange,
              ProjMatrixByBin,
              ProjMatrixByBin
	       >
{
public :
    //! Name which will be used when parsing a ProjMatrixByBin object
  static const char * const registered_name; 

  //! Default constructor (calls set_defaults())
  ProjMatrixByBinWithPositronRange();

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
		      const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );

private:

  // explicitly list necessary members for image details (should use an Info object instead)
  // ideally these should be const, but I have some trouble initialising them in that case
  CartesianCoordinate3D<float> voxel_size;
  CartesianCoordinate3D<float> origin;  
  CartesianCoordinate3D<int> min_index;
  CartesianCoordinate3D<int> max_index;

  double positron_range_C;
  double positron_range_k1;
  double positron_range_k2;
  shared_ptr<ProjMatrixByBin> post_proj_matrix_ptr;
  int positron_range_zoom;
  int positron_num_samples;


  shared_ptr<ProjDataInfo> proj_data_info_ptr;


  virtual void 
    calculate_proj_matrix_elems_for_one_bin(
                                            ProjMatrixElemsForOneBin&) const;

   virtual void set_defaults();
   virtual void initialise_keymap();
   virtual bool post_processing();
  
};

END_NAMESPACE_STIR

#endif



