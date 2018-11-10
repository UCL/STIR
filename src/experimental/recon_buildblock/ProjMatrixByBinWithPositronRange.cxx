//
//
/*!

  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for ProjMatrixByBinWithPositronRange

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2004, IRSL
    See STIR/LICENSE.txt for details
*/



#include "stir_experimental/recon_buildblock/ProjMatrixByBinWithPositronRange.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfo.h"
#include "stir/round.h"
#include "stir/is_null_ptr.h"
#include "stir/IndexRange.h"
#include <algorithm>
#include <math.h>
#include <boost/static_assert.hpp>

START_NAMESPACE_STIR


const char * const 
ProjMatrixByBinWithPositronRange::registered_name =
  "With Positron Range";

ProjMatrixByBinWithPositronRange::
ProjMatrixByBinWithPositronRange()
{
  set_defaults();
}

void 
ProjMatrixByBinWithPositronRange::initialise_keymap()
{
  ProjMatrixByBin::initialise_keymap();
  parser.add_start_key("With Positron Range Matrix Parameters");
  parser.add_parsing_key("matrix type to use after positron range blurring", 
			 &post_projmatrix_ptr);
  parser.add_key("C",  &positron_range_C);
  parser.add_key("k1", &positron_range_k1);
  parser.add_key("k2", &positron_range_k2);
  parser.add_key("zoom (odd integer)", &positron_range_zoom);
  parser.add_key("number of samples (odd integer)", &positron_range_num_samples);
  parser.add_stop_key("End With Positron Range Matrix Parameters");
}


void
ProjMatrixByBinWithPositronRange::set_defaults()
{
  ProjMatrixByBin::set_defaults();
  positron_range_C =-1;
  positron_range_k1=-1;
  positron_range_k2=-1;
  post_proj_matrix_ptr=0;
  positron_range_zoom=1;
  positron_num_samples=1;
}

bool
ProjMatrixByBinWithPositronRange::post_processing()
{
  if (ProjMatrixByBin::post_processing() == true)
    return true;
  if (positron_range_C < 0 || positron_range_C>1)
    { 
      warning("C has to be between 0 and 1 but is %g", positron_range_C); 
      return true;
    }
  if (positron_range_k1 < 0)
    { 
      warning("k1 has to be larger than 0 but is %g", positron_range_k1);
      return true;
    }
  if (positron_range_k2 < 0)
    { 
      warning("k2 has to be larger than 0 but is %g", positron_range_k2);
      return true;
    }
  if (positron_range_zoom < 1 || positron_range_zoom%2==0)
    { 
      warning("zoom has to be larger than 0 and odd but is %d", positron_range_zoom);
      return true;
    }
  if (positron_range_num_samples < 1 || positron_range_num_samples%2==0)
    { 
      warning("num_samples has to be larger than 0 and odd but is %d", positron_range_num_samples);
      return true;
    }
  if (is_null_ptr(post_projmatrix_ptr))
    { 
      warning("matrix has to be valid");
      return true;
    }
  return false;
}

void
ProjMatrixByBinWithPositronRange::
set_up(		 
       const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
       const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr  
       )
{
  proj_data_info_ptr= proj_data_info_ptr_v; 
  const VoxelsOnCartesianGrid<float> * image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByBinWithPositronRange initialised with a wrong type of DiscretisedDensity\n");

 
  voxel_size = image_info_ptr->get_voxel_size();
  origin = image_info_ptr->get_origin();
  image_info_ptr->get_regular_range(min_index, max_index);
  const CartesianCoordinate3D<float> zoomed_voxel_size = 
    voxel_size/zoom;
  const CartesianCoordinate3D<int> zoomed_max_index =
    max_index * zoom + (zoom-1)/2 + (positron_range_num_sample-1)/2;
  const CartesianCoordinate3D<int>zoomed_min_index = 
    min_index * zoom - (zoom-1)/2 - (positron_range_num_sample-1)/2;  
  
  shared_ptr<DiscretisedDensity<3,float> > zoomed_density_info_ptr =
    new VoxelsOnCartesianGrid<float>(IndexRange<3>(zoomed_min_index, zoomed_max_index),
				     origin,
				     zoomed_voxel_size);
  post_projmatrix_ptr->set_up(proj_data_info_ptr, zoomed_density_info_ptr);

  // TODO think about this. Should somehow depend on symmetries of underlying projmatrix
  symmetries_ptr = 
    new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_ptr,
                                                density_info_ptr);

};


//////////////////////////////////////
                               


void 
ProjMatrixByBinWithPositronRange::
calculate_proj_matrix_elems_for_one_bin(
                                        ProjMatrixElemsForOneBin& lor) const
{
  // This code is incomplete, and should not be used without major revision
  // and thorough testing. The reason for BOOST_STATIC_ASSERT is to enforce
  // this revision and testing before it gets used.
  BOOST_STATIC_ASSERT(false);
  const Bin bin = lor.get_bin();

  assert(lor.size() == 0);

  ProjMatrixElemsForOneBin zoomed_lor(bin);
  ProjMatrixElemsForOneBin zoomed_lor(bin);

  zoomed_projmatrix_ptr->get_proj_matrix_elems_for_one_bin(zoomed_lor);

  for (ProjMatrixElemsForOneBin::const_iterator iter = zoomed_lor.begin();
       iter != zoomed_lor.end();
       ++iter)
    {
      ProjMatrixElemsForOneBinValue
    }
}


END_NAMESPACE_STIR

