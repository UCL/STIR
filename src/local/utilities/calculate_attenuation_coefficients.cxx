//
// $Id$
//

/*!
  \file
  \ingroup utilities

  \brief Calculates attenuation coefficients 

    Attenuation_image has to contain an estimate of the mu-map for the image. It will used
    to estimate attenuation factors as exp(-forw_proj(*attenuation_image_ptr)).

  \par Usage: 
  \code 
  calculate_attenuation_coefficients  <output filename > <input header file name> <template_proj_data>
  \endcode

  \warning This currently calculates the INVERSE of the attenuation correction factors!
  \warning attenuation image data are supposed to be in units cm^-1.
  Reference: water has mu .096 cm^-1.

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/ProjDataInterfile.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Viewgram.h"
#include "stir/ArrayFunction.h"
#if 1
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#else
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#endif
#include <iostream>
#include <list>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::list;
using std::find;
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

// The start..., end_... parameters could obviously be removed, as we're
// removing the defaults anyway. However, they're there now, so we can just
// as well leave them in
void
do_segments(const VoxelsOnCartesianGrid<float>& image, 
	    ProjData& proj_data,
	    const int start_segment_num, const int end_segment_num,
	    const int start_view, const int end_view,
	    const int start_tangential_pos_num, const int end_tangential_pos_num,
	    ForwardProjectorByBin& forw_projector)
{
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    forw_projector.get_symmetries_used()->clone();  
  
  list<ViewSegmentNumbers> already_processed;
  
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    for (int view= start_view; view<=end_view; view++)      
    {       
      ViewSegmentNumbers vs(view, segment_num);
      symmetries_sptr->find_basic_view_segment_numbers(vs);
      if (find(already_processed.begin(), already_processed.end(), vs)
	!= already_processed.end())
	continue;
      
      already_processed.push_back(vs);
      
      
      RelatedViewgrams<float> viewgrams = 
	proj_data.get_empty_related_viewgrams(vs, symmetries_sptr);

      forw_projector.forward_project(viewgrams, image,
				     viewgrams.get_min_axial_pos_num(),
				     viewgrams.get_max_axial_pos_num(),
				     start_tangential_pos_num, end_tangential_pos_num);
      
      // do the exp 
      for (RelatedViewgrams<float>::iterator viewgrams_iter= viewgrams.begin();
	   viewgrams_iter != viewgrams.end();
	   ++viewgrams_iter)
      {
	Viewgram<float>& viewgram = *viewgrams_iter;
	viewgram *= -1;
	in_place_exp(viewgram);
      }      
      
      if (!(proj_data.set_related_viewgrams(viewgrams) == Succeeded::yes))
	error("Error set_related_viewgrams\n");            
    }   
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int 
main (int argc, char * argv[])
{
  
  if (argc!=4)
  {
    cerr<<"\nUsage: calculate_attenuation_coefficients : <output filename > <input header file name> <template_proj_data>"
	<<"\nWARNING: This currently calculates the INVERSE of the attenuation correction factors! \n"<<endl;
    return EXIT_FAILURE;
  }
  
  shared_ptr <DiscretisedDensity<3,float> > attenuation_density_ptr =
    DiscretisedDensity<3,float>::read_from_file(argv[2]);
  VoxelsOnCartesianGrid<float> *  attenuation_image_ptr = 
    dynamic_cast<VoxelsOnCartesianGrid<float> *> (attenuation_density_ptr.get());

    cerr << "WARNING: attenuation image data are supposed to be in units cm^-1\n"
      "Reference: water has mu .096 cm^-1" << endl;
    cerr<< "Max in attenuation image:" 
      << attenuation_image_ptr->find_max() << endl;
#ifndef NORESCALE
    /*
      cerr << "WARNING: multiplying attenuation image by x-voxel size "
      << " to correct for scale factor in forward projectors...\n";
    */
    // projectors work in pixel units, so convert attenuation data 
    // from cm^-1 to pixel_units^-1
    const float rescale = attenuation_image_ptr->get_voxel_size().x()/10;
#else
    const float rescale = 
      10.F;
#endif
    *attenuation_image_ptr *= rescale;

  shared_ptr<ProjData> template_proj_data_ptr = 
    ProjData::read_from_file(argv[3]);

#if 0
  shared_ptr<ProjMatrixByBin> PM = 
    new  ProjMatrixByBinUsingRayTracing();
  shared_ptr<ForwardProjectorByBin> forw_projector_ptr =
    new ForwardProjectorByBinUsingProjMatrixByBin(PM); 
#else
  shared_ptr<ForwardProjectorByBin> forw_projector_ptr =
    new ForwardProjectorByBinUsingRayTracing();
#endif
  forw_projector_ptr->set_up(template_proj_data_ptr->get_proj_data_info_ptr()->clone(),
			       attenuation_density_ptr );
  cerr << "\n\nForward projector used:\n" << forw_projector_ptr->parameter_info();  

  const string output_file_name = argv[1];
  shared_ptr<ProjData> out_proj_data_ptr =
    new ProjDataInterfile(template_proj_data_ptr->get_proj_data_info_ptr()->clone(),
			  output_file_name);
  
  do_segments(*attenuation_image_ptr,*out_proj_data_ptr,
	      out_proj_data_ptr->get_min_segment_num(), out_proj_data_ptr->get_max_segment_num(), 
	      out_proj_data_ptr->get_min_view_num(), 
	      out_proj_data_ptr->get_max_view_num(),
	      out_proj_data_ptr->get_min_tangential_pos_num(), 
	      out_proj_data_ptr->get_max_tangential_pos_num(),
	      *forw_projector_ptr);  
  
  return EXIT_SUCCESS;
}

