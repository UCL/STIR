//
// $Id$
//
/*!
  \file

  \brief non-inline implementations for IterativeReconstructionParameters

  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

// TODO get rid of restriction of subsets according to 'view45'
// it's not appropriate for general symmetries

#include "stir/recon_buildblock/IterativeReconstructionParameters.h" 
#include "stir/NumericInfo.h"
#include "stir/ImageProcessor.h"
#include "stir/utilities.h"
// for time(), used as seed for random stuff
#include <ctime>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ends;
#endif

START_NAMESPACE_STIR

IterativeReconstructionParameters::IterativeReconstructionParameters()
  : ReconstructionParameters()
{
}

void 
IterativeReconstructionParameters::set_defaults()
{
  ReconstructionParameters::set_defaults();
  num_subsets = 1;
  start_subset_num = 0;
  num_subiterations = 1;
  start_subiteration_num =1;
  zero_seg0_end_planes = 0;
  // default to all 1's
  initial_image_filename = "1";

  max_num_full_iterations=NumericInfo<int>().max_value();
  save_interval = 1;
  inter_iteration_filter_interval = 0;
  inter_iteration_filter_ptr = 0;
  post_filter_ptr = 0;
//MJ 02/08/99 added subset randomization
  // KT 05/07/2000 made int
  randomise_subset_order = 0;

}
void 
IterativeReconstructionParameters::initialise_keymap()
{
  ReconstructionParameters::initialise_keymap();
  parser.add_key("number of subiterations",  &num_subiterations);
  parser.add_key("start at subiteration number",  &start_subiteration_num);
  parser.add_key("save images at subiteration intervals",  &save_interval);
  parser.add_key("zero end planes of segment 0", &zero_seg0_end_planes);
  parser.add_key("initial image", &initial_image_filename);
  parser.add_key("number of subsets", &num_subsets);
  parser.add_key("start at subset", &start_subset_num);
  parser.add_key("inter-iteration filter subiteration interval",&inter_iteration_filter_interval);
  parser.add_parsing_key("inter-iteration filter type", &inter_iteration_filter_ptr);
  parser.add_parsing_key("post-filter type", &post_filter_ptr);
  
  //MJ 02/08/99 added subset randomization
  parser.add_key("uniformly randomise subset order", &randomise_subset_order);

}


void IterativeReconstructionParameters::ask_parameters()
{

  ReconstructionParameters::ask_parameters();
 

  char initial_image_filename_char[max_filename_length];



 
  const int view45 = proj_data_ptr->get_proj_data_info_ptr()->get_num_views()/4;
    
  // KT 15/10/98 correct formula for max
  max_segment_num_to_process=
    ask_num("Maximum absolute segment number to process: ",
	    0, proj_data_ptr->get_max_segment_num(), 0);
  
  // KT 05/07/2000 made int
  zero_seg0_end_planes =
    ask("Zero end planes of segment 0 ?", true) ? 1 : 0;
  
  
  
  // KT 21/10/98 use new order of arguments
  ask_filename_with_extension(initial_image_filename_char,
    "Get initial image from which file (1 = 1's): ", "");
  
  initial_image_filename=initial_image_filename_char;
  
  num_subsets= ask_num("Number of ordered sets: ", 1,view45,1);
  num_subiterations=ask_num("Number of subiterations",
    1,NumericInfo<int>().max_value(),num_subsets);
  
  start_subiteration_num=ask_num("Start at what subiteration number : ", 1,NumericInfo<int>().max_value(),1);
  
  start_subset_num=ask_num("Start with which ordered set : ",
    0,num_subsets-1,0);
  
  save_interval=ask_num("Save images at sub-iteration intervals of: ", 
    1,num_subiterations,num_subiterations);
  
  
  
  inter_iteration_filter_interval=
    ask_num("Do inter-iteration filtering at sub-iteration intervals of: ",              0, num_subiterations, 0);
  
  if(inter_iteration_filter_interval>0 )
  {
    cerr<<endl<<"Supply inter-iteration filter type:\nPossible values:\n";
    ImageProcessor<3,float>::list_registered_names(cerr);
    
    const string inter_iteration_filter_type = ask_string("");
    
    inter_iteration_filter_ptr = 
      ImageProcessor<3,float>::read_registered_object(0, inter_iteration_filter_type);      
  } 
  
  
  const bool  do_post_filtering=ask("Post-filter final image?", false);
  
  
  
  if(do_post_filtering)
  { 
    
    cerr<<endl<<"Supply post filter type:\nPossible values:\n";
    ImageProcessor<3,float>::list_registered_names(cerr);
    cerr<<endl;
    
    const string post_filter_type = ask_string("");
    
    post_filter_ptr = 
      ImageProcessor<3,float>::read_registered_object(0, post_filter_type);      	   
  }
  
  
  // KT 05/07/2000 made int
  randomise_subset_order=
    ask("Randomly generate subset order?", false) ? 1 : 0;
  
  
}



bool IterativeReconstructionParameters::post_processing() 
{
  if (ReconstructionParameters::post_processing())
    return true;

  if (initial_image_filename.length() == 0)
  { warning("You need to specify an initial image file\n"); return true; }

 if (num_subsets<1 )
  { warning("number of subsets should be positive\n"); return true; }
  if (num_subiterations<1)
  { warning("Range error in number of subiterations\n"); return true; }
  
  if(start_subset_num<0 || start_subset_num>=num_subsets) 
  { warning("Range error in starting subset\n"); return true; }

  if(save_interval<1 || save_interval>num_subiterations) 
  { warning("Range error in iteration save interval\n"); return true;}
 
  if (inter_iteration_filter_interval<0)
  { warning("Range error in inter-iteration filter interval \n"); return true; }

 if (start_subiteration_num<1)
   { warning("Range error in starting subiteration number\n"); return true; }

  ///////////////// consistency checks

 if(max_segment_num_to_process > proj_data_ptr->get_max_segment_num()) 
 { warning("Range error in number of segments\n"); return true;}
  
  if( num_subsets>proj_data_ptr->get_num_views()/4) 
  { warning("Range error in number of subsets\n"); return true;}
  
  ////////////////// subset order

  // KT 05/07/2000 made randomise_subset_order int
  if (randomise_subset_order!=0){
   srand((unsigned int) (time(NULL)) ); //seed the rand() function
   }

  return false;
}

END_NAMESPACE_STIR
