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

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR


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





