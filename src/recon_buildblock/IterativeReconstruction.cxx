//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief  implementation of the IterativeReconstruction class 
    
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


#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ImageProcessor.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include <iostream>
#include "stir/NumericInfo.h"
#include "stir/ImageProcessor.h"
#include "stir/utilities.h"
// for time(), used as seed for random stuff
#include <ctime>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ends;
#endif


// TODO get rid of restriction of subsets according to 'view45'
// it's not appropriate for general symmetries

START_NAMESPACE_STIR

//********* parameters ****************

void 
IterativeReconstruction::set_defaults()
{
  Reconstruction::set_defaults();
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
IterativeReconstruction::initialise_keymap()
{
  Reconstruction::initialise_keymap();
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


void IterativeReconstruction::ask_parameters()
{

  Reconstruction::ask_parameters();
 

  char initial_image_filename_char[max_filename_length];



 
  const int view45 = proj_data_ptr->get_proj_data_info_ptr()->get_num_views()/4;
    
  
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



bool IterativeReconstruction::post_processing() 
{
  if (Reconstruction::post_processing())
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

//************ other functions ****************

IterativeReconstruction::IterativeReconstruction()
{
  //initialize iteration loop terminator
  terminate_iterations=false;
}

DiscretisedDensity<3,float> *
IterativeReconstruction::get_initial_image_ptr() const
{
  if(get_parameters().initial_image_filename=="0")
  {
    return construct_target_image_ptr();    
  }
  else if(get_parameters().initial_image_filename=="1")
  {
    DiscretisedDensity<3,float> * target_image_ptr =
      construct_target_image_ptr();    
    target_image_ptr->fill(1.F);
    return target_image_ptr;
  }
  else
    {
      return 
        DiscretisedDensity<3,float>::read_from_file(get_parameters().initial_image_filename);
    }
}

// KT 10122001 new
Succeeded 
IterativeReconstruction::
reconstruct() 
{
  shared_ptr<DiscretisedDensity<3,float> > target_image_ptr =
    get_initial_image_ptr();
  return reconstruct(target_image_ptr);
}

Succeeded 
IterativeReconstruction::
reconstruct(shared_ptr<DiscretisedDensity<3,float> > const& target_image_ptr)
{

  start_timers();

  recon_set_up(target_image_ptr);

  for(subiteration_num=get_parameters().start_subiteration_num;subiteration_num<=get_parameters().num_subiterations && terminate_iterations==false; subiteration_num++)
  {
    update_image_estimate(*target_image_ptr);
    end_of_iteration_processing(*target_image_ptr);
  }

  stop_timers();

  cerr << "Total CPU Time " << get_CPU_timer_value() << "secs"<<endl;

  // currently, if there was something wrong, the programme is just aborted
  // so, if we get here, everything was fine
  return Succeeded::yes;

}


void 
IterativeReconstruction::
recon_set_up(shared_ptr<DiscretisedDensity<3,float> > const& target_image_ptr)
{
   
  // Building filters
  // This is not really necessary, as apply would call this anyway.
  // However, we have it here such that any errors in building the filters would
  // be caught before doing any projections or so done.
#if 0 
  /* 
  KT 04/06/2003 disabled the explicit calling of inter_iteration_filter_ptr->set_up()
  
  It was here to catch incompatibilities between the filter and the
  image early (i.e. before any real reconstruction stuff has been going on). Now
  this will only be caught when the inter_iteration_filter is called for the first time.

  The reason I disabled this is that OSMAPOSL::recon_setup (and presumably
  other algorithms that insist on non-negative data) chains the current
  inter_iteration_filter with a ThresholdMinToSmallPositiveValueImageProcessor. 
  This meant that the new image processor was not set-up yet, and resulted 
  in the current filter being set-up twice, which might potentially take a lot 
  of CPU time.
  */
  if(get_parameters().inter_iteration_filter_interval>0 && get_parameters().inter_iteration_filter_ptr != 0 )
    {
      cerr<<endl<<"Building inter-iteration filter kernel"<<endl;
      if (get_parameters().inter_iteration_filter_ptr->set_up(*target_image_ptr)
          == Succeeded::no)
	error("Error building inter iteration filter\n");
    }
#endif
 
  if(get_parameters().post_filter_ptr != 0) 
  {
    cerr<<endl<<"Building post filter kernel"<<endl;
    
    if (get_parameters().post_filter_ptr->set_up(*target_image_ptr)
          == Succeeded::no)
	error("Error building post filter\n");
  }

}


void IterativeReconstruction::end_of_iteration_processing(DiscretisedDensity<3,float> &current_image_estimate)
{


  
  if(get_parameters().inter_iteration_filter_interval>0 && 
     get_parameters().inter_iteration_filter_ptr != 0 &&
     subiteration_num%get_parameters().inter_iteration_filter_interval==0)
    {
      cerr<<endl<<"Applying inter-iteration filter"<<endl;
      get_parameters().inter_iteration_filter_ptr->apply(current_image_estimate);
    }
 

  cerr<< method_info()
      << " subiteration #"<<subiteration_num<<" completed"<<endl;
  cerr << "  min and max in image " << current_image_estimate.find_min() 
    << " " << current_image_estimate.find_max() << endl;
  
  if(subiteration_num==get_parameters().num_subiterations &&
     get_parameters().post_filter_ptr!=0 )
  {
    cerr<<endl<<"Applying post-filter"<<endl;
    get_parameters().post_filter_ptr->apply(current_image_estimate);
    
    cerr << "  min and max after post-filtering " << current_image_estimate.find_min() 
      << " " << current_image_estimate.find_max() << endl <<endl;
  }
  
    // Save intermediate (or last) iteration      
  if((!(subiteration_num%get_parameters().save_interval)) || subiteration_num==get_parameters().num_subiterations ) 
    {      	         
      // allocate space for the filename assuming that
      // we never have more than 10^49 subiterations ...
      char * fname = new char[get_parameters().output_filename_prefix.size() + 50];
      sprintf(fname, "%s_%d", get_parameters().output_filename_prefix.c_str(), subiteration_num);

     // Write it to file
      get_parameters().output_file_format_ptr->
	write_to_file(fname, current_image_estimate);
      delete fname; 
    }
}


VectorWithOffset<int> IterativeReconstruction::randomly_permute_subset_order()
{

  VectorWithOffset<int> temp_array(get_parameters().num_subsets),final_array(get_parameters().num_subsets);
  int index;

 for(int i=0;i<get_parameters().num_subsets;i++) temp_array[i]=i;

 for (int i=0;i<get_parameters().num_subsets;i++)
   {

   index = (int) (((float)rand()/(float)RAND_MAX)*(get_parameters().num_subsets-i));
   if (index==get_parameters().num_subsets-i) index--;
   final_array[i]=temp_array[index];
 

   for (int j=index;j<get_parameters().num_subsets-(i+1);j++) 
     temp_array[j]=temp_array[j+1];

   }

 cerr<<endl<<"Generating new subset sequence: ";
 for(int i=0;i<get_parameters().num_subsets;i++) cerr<<final_array[i]<<" ";

 return final_array;

}


END_NAMESPACE_STIR





