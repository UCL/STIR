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
      
  \date $Date$        
  \version $Revision$
*/


#include "recon_buildblock/IterativeReconstruction.h"
#include "DiscretisedDensity.h"
#include "tomo/ImageProcessor.h"
#include "interfile.h"
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_TOMO


IterativeReconstruction::IterativeReconstruction()
{
  //initialize iteration loop terminator
  terminate_iterations=false;
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
   
  //TODO use zoom factor et al
  if(get_parameters().zoom!=1)
    warning("Warning: No zoom factor will be used\n");
  if(get_parameters().Xoffset!=0)
    warning("Warning: Xoffset will not be used\n");
  if(get_parameters().Yoffset!=0)
    warning("Warning: Yoffset will not be used\n");
  if(get_parameters().output_image_size!=-1)
    warning("Warning: output_image_size will keep its default value\n");
  if(get_parameters().num_views_to_add!=1)
    warning("Warning: No mashing will be used\n");

  // Building filters
  // This is not really necessary, as build_and_filter would call this anyway.
  // However, we have it here such that any errors in building the filters would
  // be caught before doing any projections or so done.
  
  if(get_parameters().inter_iteration_filter_interval>0 && get_parameters().inter_iteration_filter_ptr != 0 )
    {
      cerr<<endl<<"Building inter-iteration filter kernel"<<endl;
      get_parameters().inter_iteration_filter_ptr->build_filter(*target_image_ptr);
    }

 
  if(get_parameters().post_filter_ptr != 0) 
  {
    cerr<<endl<<"Building post filter kernel"<<endl;
    
    get_parameters().post_filter_ptr->build_filter(*target_image_ptr);
  }

}


void IterativeReconstruction::end_of_iteration_processing(DiscretisedDensity<3,float> &current_image_estimate)
{


  
  if(get_parameters().inter_iteration_filter_interval>0 && 
     get_parameters().inter_iteration_filter_ptr != 0 &&
     subiteration_num%get_parameters().inter_iteration_filter_interval==0)
    {
      cerr<<endl<<"Applying inter-iteration filter"<<endl;
      get_parameters().inter_iteration_filter_ptr->build_and_filter(current_image_estimate);
    }
 

  cerr<< method_info()
      << " subiteration #"<<subiteration_num<<" completed"<<endl;
  cerr << "  min and max in image " << current_image_estimate.find_min() 
    << " " << current_image_estimate.find_max() << endl;
  
  if(subiteration_num==get_parameters().num_subiterations &&
     get_parameters().post_filter_ptr!=0 )
  {
    cerr<<endl<<"Applying post-filter"<<endl;
    get_parameters().post_filter_ptr->build_and_filter(current_image_estimate);
    
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
      write_basic_interfile(fname, current_image_estimate);
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


END_NAMESPACE_TOMO





