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
FBP2DReconstruction(const Segment<float>& direct_sinos, const RampFilter& f)
: filter(f),
  direct_sinos(direct_sinos)
{}

Succeeded 
FBP2DReconstruction::
reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & density_ptr)
{
  VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);

  assert(direct_sinos.get_proj_data_info_ptr()->get_tantheta(Bin(direct_sinos.get_segment_num(),0,0,0)) ==0);

  shared_ptr<BackProjectorByBin> back_projector_ptr =
    new BackProjectorByBinUsingInterpolation(direct_sinos.get_proj_data_info_ptr()->clone(), 
       density_ptr,
    /*use_piecewise_linear_interpolation = */true, 
    /*use_exact_Jacobian = */ false);

  density_ptr->fill(0);
  
  // TODO get boundaries from the symmetries ?
  for (int view=0; view <= direct_sinos.get_num_views() /4; view++) 
  {         
    // terrible trick to get a RelatedViewgrams object:
    // first get an empty one
    RelatedViewgrams<float> viewgrams = 
      direct_sinos.get_proj_data_info_ptr()->get_empty_related_viewgrams(ViewSegmentNumbers(view, 0),
      back_projector_ptr->get_symmetries_used()->clone());
    // now fill in with the actual data 
    for (RelatedViewgrams<float>::iterator viewgram_iter = viewgrams.begin();
         viewgram_iter != viewgrams.end();
         ++viewgram_iter)
    {
      *viewgram_iter = direct_sinos.get_viewgram(viewgram_iter->get_view_num());
    }


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
  // This starts from the following reasoning:
  //   integral(projection_data_for_one_view) == integral(image)
  // We take the average of the lhs over all views (better for noise).
  // We have to take proper units into account when replacing the 
  // integrals by discrete sums:
  //   sum_over_bins(projection_data_for_one_view) bin_size ==
  //   sum_over_pixels(image) voxel_size.x voxel_size.y
  // The binsize factor is only there when the forward projector
  // uses mm units, instead of pixel-size units.

  // KT & Darren Hogg 17/05/2000 finally found out the scale factor!
#if 1
  const ProjDataInfoCylindricalArcCorr& proj_data_info_cyl =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr&>
    (*direct_sinos.get_proj_data_info_ptr());
  // TODO remove magic, is a scale factor in the backprojector 
  const float magic_number=2*proj_data_info_cyl.get_ring_radius()*proj_data_info_cyl.get_num_views()/proj_data_info_cyl.get_ring_spacing();
#ifdef NEWSCALE
  // added binsize etc here to get units ok
  // only do this when the forward projector units are appropriate
  image *= magic_number / direct_sinos.get_num_views() *
    proj_data_info_cyl.get_bin_size()/
    (image.get_voxel_size().x*image.get_voxel_size().y);
#else
  image *= magic_number / direct_sinos.get_num_views();
#endif

#else // old way of scaling 

  const ProjDataInfoCylindricalArcCorr& proj_data_info_cyl =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr&>
    (*direct_sinos.get_proj_data_info_ptr());


#ifdef NEWSCALE
  // added binsize etc here to get units ok
  // only do this when the forward projector units are appropriate
#ifdef KTTEST  
  cerr << "Reconstruct2DFBP: applying scale factor " <<
     direct_sinos.sum()*proj_data_info_cyl.get_bin_size()/
       (direct_sinos.get_num_views()* image.sum_positive()*
        image.get_voxel_size().x*image.get_voxel_size().y) 
        << endl;
#endif
  
  
  if(image.sum_positive()==0)
    image.fill(0.F);
  else
    image*= direct_sinos.sum()*proj_data_info_cyl.get_bin_size()/
       (direct_sinos.get_num_views()* image.sum()*
        image.get_voxel_size().x*image.get_voxel_size().y);
#else
  // original code
  if(1){
      
      if(image.sum()==0)
          image.fill(0.F);
      else
      {
#ifdef KTTEST
	cerr << "Reconstruct2DFBP: applying scale factor " <<
	    direct_sinos.sum()/
              (direct_sinos.get_num_views()* image.sum()) << endl;
#endif
	  
           image*= direct_sinos.sum()/
              (direct_sinos.get_num_views()* image.sum());
          
	  
      }
}
 
#endif // !NEWSCALE

#endif // old way of scaling plane by plane

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
