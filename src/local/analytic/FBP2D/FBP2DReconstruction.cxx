//
// $Id$
//
/*!
  \file
  \ingroup reconstructors

  \brief Implementation of class FBP2DReconstruction

  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/FBP2D/FBP2DReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Segment.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/Bin.h"


START_NAMESPACE_STIR

string FBP2DReconstruction::method_info() const
{
  return "2D FBP";
}

string FBP2DReconstruction::parameter_info()
{
  return parameters.parameter_info();
}

FBP2DReconstruction::
FBP2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr, const RampFilter& f)
: filter(f)
{
  parameters.proj_data_ptr = proj_data_ptr;
}

Succeeded 
FBP2DReconstruction::
reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & density_ptr)
{
  if (dynamic_cast<const ProjDataInfoCylindricalArcCorr*>
       (parameters.proj_data_ptr->get_proj_data_info_ptr()) == 0)
  {
    warning("Projection data has to be arc-corrected for FBP2D\n");
    return Succeeded::no;
  }

  VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);

  assert(fabs(parameters.proj_data_ptr->get_proj_data_info_ptr()->get_tantheta(Bin(0,0,0,0)) ) < 1.E-4);

  shared_ptr<BackProjectorByBin> back_projector_ptr =
    new BackProjectorByBinUsingInterpolation(parameters.proj_data_ptr->get_proj_data_info_ptr()->clone(), 
       density_ptr,
    /*use_piecewise_linear_interpolation = */true, 
    /*use_exact_Jacobian = */ false);

  density_ptr->fill(0);
  
  // TODO get boundaries from the symmetries ?
  for (int view=0; view <= parameters.proj_data_ptr->get_num_views() /4; view++) 
  {         
    RelatedViewgrams<float> viewgrams = 
      parameters.proj_data_ptr->get_related_viewgrams(ViewSegmentNumbers(view, 0),
                                                      back_projector_ptr->get_symmetries_used()->clone());   

    // now filter
    for (RelatedViewgrams<float>::iterator viewgram_iter = viewgrams.begin();
         viewgram_iter != viewgrams.end();
         ++viewgram_iter)
    {
      filter.apply(*viewgram_iter);
    }
    //  and backproject
    back_projector_ptr->back_project(*density_ptr, viewgrams);
  }  
  // Normalise the image
  // The binsize factor is only there when the forward projector
  // uses mm units, instead of pixel-size units.

  // KT & Darren Hogg 17/05/2000 finally found the scale factor!

  const ProjDataInfoCylindricalArcCorr& proj_data_info_cyl =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr&>
    (*parameters.proj_data_ptr->get_proj_data_info_ptr());
  // TODO remove magic, is a scale factor in the backprojector 
  const float magic_number=2*proj_data_info_cyl.get_ring_radius()*proj_data_info_cyl.get_num_views()/proj_data_info_cyl.get_ring_spacing();
#ifdef NEWSCALE
  // added binsize etc here to get units ok
  // only do this when the forward projector units are appropriate
  image *= magic_number / parameters.proj_data_ptr->get_num_views() *
    proj_data_info_cyl.get_bin_size()/
    (image.get_voxel_size().x()*image.get_voxel_size().y());
#else
  image *= magic_number / parameters.proj_data_ptr->get_num_views();
#endif

  return Succeeded::yes;
}


ReconstructionParameters& 
FBP2DReconstruction::
params()
{
  return parameters;
}
const ReconstructionParameters& 
FBP2DReconstruction::params() const
{
  return parameters;
}
 

END_NAMESPACE_STIR
