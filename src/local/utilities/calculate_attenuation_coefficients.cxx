//
// $Id:
//

/*!
  \file
  \ingroup 

  \brief Calucaltes attenuation coefficients :
    Attenuation_image_ptr has to contain an estimate of the mu-map for the image. It will used
    to estimate attenuation factors as exp(-forw_proj(*attenuation_image_ptr)).

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Year:$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/utilities.h"
#include "stir/ProjDataInterfile.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Viewgram.h"
#include "stir/ArrayFunction.h"
#include "stir/recon_array_functions.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"

#include "stir/SegmentByView.h"

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

void
do_segments(const VoxelsOnCartesianGrid<float>& image, 
	    ProjData& proj_data,
	    const int start_segment_num, const int end_segment_num,
	    const int start_view, const int end_view,
	    const int start_tangential_pos_num, const int end_tangential_pos_num,
	    ForwardProjectorByBin& forw_projector,
	    const bool disp);


void
do_segments(const VoxelsOnCartesianGrid<float>& image, 
	    ProjData& proj_data,
	    const int start_segment_num, const int end_segment_num,
	    const int start_view, const int end_view,
	    const int start_tangential_pos_num, const int end_tangential_pos_num,
	    ForwardProjectorByBin& forw_projector,
	    const bool disp)
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
	proj_data.get_empty_related_viewgrams(vs, symmetries_sptr,false);

      forw_projector.forward_project(viewgrams, image,
	viewgrams.get_min_axial_pos_num(),
	viewgrams.get_max_axial_pos_num(),
	start_tangential_pos_num, end_tangential_pos_num);
      
      // do the exp 
      RelatedViewgrams<float>::iterator viewgrams_iter= 
	viewgrams.begin();
      for (; 
      viewgrams_iter != viewgrams.end();
      ++viewgrams_iter)
      {
	Viewgram<float>& viewgram = *viewgrams_iter;
	viewgram *= -1;
	in_place_exp(viewgram);
	// set rim_truncation_sino to 0 as there is no scatter and 
	// presumably there would not be any influance whatsoever
	int rim_truncation_sino = 0;
	truncate_rim(viewgram, rim_truncation_sino);
      }
      //if (multiplicative_binwise_correction_ptr != NULL)
      //{
      // KTXXX temporary (?) fix to divide instead of multiply
      //*viewgrams_ptr /= *multiplicative_binwise_correction_ptr;
      
      //}
      
      
      if (!(proj_data.set_related_viewgrams(viewgrams) == Succeeded::yes))
	error("Error set_related_viewgrams\n");            
    }   
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int 
main (int argc, char * argv[])
{
  
  if (argc!=3)
  {
    cerr<<"\nUsage: calculate attenuation coefficients : <output filename > <input header file name> <proj_data> \n"<<endl;
  }
  
  shared_ptr <DiscretisedDensity<3,float> > attenuation_density_ptr =
    DiscretisedDensity<3,float>::read_from_file(argv[2]);
  VoxelsOnCartesianGrid<float> *  attenuation_image_ptr = 
    dynamic_cast<VoxelsOnCartesianGrid<float> *> (attenuation_density_ptr.get());

  // projectors work in pixel units, so convert attenuation data 
  // from cm^-1 to pixel_units^-1 
  const float rescale = attenuation_image_ptr->get_voxel_size().x()/10;
  *attenuation_image_ptr *= rescale;      

  shared_ptr<ProjData> proj_data_ptr = 
    ProjData::read_from_file(argv[3]);

   ProjDataInfo* new_data_info_ptr;
   new_data_info_ptr= proj_data_ptr->get_proj_data_info_ptr()->clone();

  shared_ptr<ProjMatrixByBin> PM = 
    new  ProjMatrixByBinUsingRayTracing(); //attenuation_image_ptr , proj_data_ptr->get_proj_data_info_ptr()->clone()); 	
  ForwardProjectorByBin* forw_projector_ptr =
    new ForwardProjectorByBinUsingProjMatrixByBin(PM); 
  
  forw_projector_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
			       attenuation_density_ptr );
  cerr << forw_projector_ptr->parameter_info();  

  const string output_file_name = argv[1];
  shared_ptr<ProjData> out_proj_data_ptr =
    new ProjDataInterfile(new_data_info_ptr,
			  output_file_name);
  
  do_segments(*attenuation_image_ptr,*out_proj_data_ptr,
      proj_data_ptr->get_min_segment_num(), proj_data_ptr->get_max_segment_num(), 
      proj_data_ptr->get_min_view_num(), 
      proj_data_ptr->get_max_view_num(),
      proj_data_ptr->get_min_tangential_pos_num(), 
      proj_data_ptr->get_max_tangential_pos_num(),
      *forw_projector_ptr,
	      false);  
  
  return EXIT_SUCCESS;
}

