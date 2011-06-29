//
// $Id$
//

/*!
  \file
  \ingroup 

  \brief This programm was based on bck_project originnal code with the difference
   that here we normalize the final value of a single pixel with a sum of all values 
   in the corresponding LOR.

   BackProjectorByBinUsingSquareProjMatrixByBin was used which is modification of 
   the BackProjectorByBinUsingProjMatrixByBin with the difference in the sum where one has
   out(b)= sqrt (sum square(p(d,b))* in(d) / sum square(p(d,b)))


  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/BackProjectorByBinUsingSquareProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/IO/interfile.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
// for ask_filename...
#include "stir/utilities.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Viewgram.h"


#include <fstream>
#include <list>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::iostream;
using std::endl;
using std::list;
using std::find;
using std::cerr;
using std::endl;
#endif



START_NAMESPACE_STIR

void
do_segments(DiscretisedDensity<3,float>& image, 
            ProjData& proj_data_org,
	    const int start_segment_num, const int end_segment_num,
	    const int start_view, const int end_view,
	    BackProjectorByBin& back_projector,
	    bool fill_with_1)
{
  
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    back_projector.get_symmetries_used()->clone();
  
  
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
    
    cerr << "Processing view " << vs.view_num()
      << " of segment " <<vs.segment_num()
      << endl;
    
    if(fill_with_1 )
    {
      RelatedViewgrams<float> viewgrams_empty= 
	proj_data_org.get_empty_related_viewgrams(vs, symmetries_sptr);

      viewgrams_empty.fill(1.F);
      
      back_projector.back_project(image,viewgrams_empty);
    }
    else
    {
      RelatedViewgrams<float> viewgrams = 
	proj_data_org.get_related_viewgrams(vs,	symmetries_sptr);
      	
      back_projector.back_project(image,viewgrams);      
    } // fill
  } // for view_num, segment_num    
    
}


END_NAMESPACE_STIR



USING_NAMESPACE_STIR
int
main(int argc, char *argv[])
{  
  shared_ptr<ProjData> proj_data_ptr;
 
  bool do_denominator;
  bool do_sqrt;

  switch(argc)
  {
  case 3:
    { 
      proj_data_ptr = ProjData::read_from_file(argv[1]); 
      do_denominator = ask("Do you want to normalise the result ?", true);
      do_sqrt = ask("Do you want to take the sqrt ?", true);
      break;
    }
    /*
  case 2:
    {
      cerr << endl;
      cerr <<"Usage: " << argv[0] << "[proj_data_file] outputfile name\n";
      shared_ptr<ProjDataInfo> data_info= ProjDataInfo::ask_parameters();
      // create an empty ProjDataFromStream object
      // such that we don't have to differentiate between code later on
      proj_data_ptr = 
	new ProjDataFromStream (data_info,static_cast<iostream *>(NULL));
      fill = true;
      break;
    }
    */
  default:
    {
      cerr <<"Usage: " << argv[0] << "proj_data_file outputfile_name\n";
      return (EXIT_FAILURE);
    }

  }   



  bool projector_type = 
    ask ( " Which projector do you want to use Ray Tracing (Y) or Solid Angle (N) ", true) ;


    
   const ProjDataInfo * proj_data_info_ptr = 
    proj_data_ptr->get_proj_data_info_ptr();
  
  VoxelsOnCartesianGrid<float> * vox_image_ptr =
    new VoxelsOnCartesianGrid<float>(*proj_data_info_ptr); 

  shared_ptr<DiscretisedDensity<3,float> > image_sptr = vox_image_ptr;


  string name ;
  if ( projector_type )
   name = "Ray Tracing";
  else
    name = "Solid Angle";  
 
 shared_ptr<ProjMatrixByBin> PM = 
    ProjMatrixByBin :: read_registered_object(0, name);
 
  PM->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),image_sptr);
  shared_ptr<BackProjectorByBin>  bck_projector_ptr  =
    new BackProjectorByBinUsingSquareProjMatrixByBin(PM);   
 
  

  {    
    const int max_segment_num = 
      ask_num("Maximum absolute segment number to backproject",
        0,proj_data_info_ptr->get_max_segment_num(), 
        proj_data_info_ptr->get_max_segment_num());
        

    image_sptr->fill(0);
    
    CPUTimer timer;
    timer.reset();
    timer.start();
 
    do_segments(*image_sptr, 
      *proj_data_ptr, 
      -max_segment_num, max_segment_num,
      proj_data_info_ptr->get_min_view_num(), proj_data_info_ptr->get_max_view_num(),
      *bck_projector_ptr,
      false);  

    if (do_denominator)
    {
      shared_ptr<DiscretisedDensity<3,float> > denominator_ptr =
          image_sptr->get_empty_discretised_density();
      // set to non-zero value to avoid problems with division outside the FOV
      denominator_ptr->fill(image_sptr->find_max() * 1.E-10F);
      do_segments(*denominator_ptr, 
                  *proj_data_ptr, 
                  -max_segment_num, max_segment_num,
                  proj_data_info_ptr->get_min_view_num(), proj_data_info_ptr->get_max_view_num(),
                  *bck_projector_ptr,
                  true);  
      *image_sptr /= *denominator_ptr;
    }
    if (do_sqrt)
    {
    //in_place_apply_function(*image_sptr, &sqrt);
    for (DiscretisedDensity<3,float>::full_iterator iter = image_sptr->begin_all();
         iter != image_sptr->end_all();
	 ++iter)
	 *iter = sqrt(*iter);
    }

    
    timer.stop();
    cerr << timer.value() << " s CPU time"<<endl;
    cerr << "min and max in image " << image_sptr->find_min()
      << ", " << image_sptr->find_max() << endl;
    

    
    {
      char* file = argv[2];


      cerr <<"  - Saving " << file << endl;
      write_basic_interfile(file, *image_sptr);

      
    }

  }
  
  return  EXIT_SUCCESS;
}


