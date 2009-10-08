//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup recon_buildblock

  \brief  implementation of the stir::IterativeReconstruction class 
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
      
  $Date$        
  $Revision$
*/


#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include <iostream>
#include "stir/NumericInfo.h"
#include "stir/utilities.h"
#include "stir/is_null_ptr.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"
#include <algorithm>
// for time(), used as seed for random stuff
#include <ctime>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

//********* parameters ****************
template <typename TargetT>
void 
IterativeReconstruction<TargetT>::set_defaults()
{
  base_type::set_defaults();
  this->objective_function_sptr = 0;
  this->num_subsets = 1;
  this->start_subset_num = 0;
  this->num_subiterations = 1;
  this->start_subiteration_num =1;
  // default to all 1's
  this->initial_data_filename = "1";

  this->max_num_full_iterations=NumericInfo<int>().max_value();
  this->save_interval = 1;
  this->inter_iteration_filter_interval = 0;
  this->inter_iteration_filter_ptr = 0;
//MJ 02/08/99 added subset randomization
  this->randomise_subset_order = false;
  this->report_objective_function_values_interval = 0;

}

template <typename TargetT>
void 
IterativeReconstruction<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_parsing_key("objective function type", &objective_function_sptr);
  this->parser.add_key("number of subiterations",  &num_subiterations);
  this->parser.add_key("start at subiteration number",  &start_subiteration_num);
  this->parser.add_key("save estimates at subiteration intervals",  &save_interval);
  this->parser.add_key("initial estimate", &initial_data_filename);
  this->parser.add_key("number of subsets", &this->num_subsets);
  this->parser.add_key("start at subset", &start_subset_num);
  this->parser.add_key("uniformly randomise subset order", &randomise_subset_order);
  this->parser.add_key("inter-iteration filter subiteration interval",&inter_iteration_filter_interval);
  this->parser.add_parsing_key("inter-iteration filter type", &inter_iteration_filter_ptr);
  this->parser.add_key("report objective function values interval",
		       &this->report_objective_function_values_interval);
}

template <typename TargetT>
void IterativeReconstruction<TargetT>::
ask_parameters()
{

  base_type::ask_parameters();
 

  char initial_data_filename_char[max_filename_length];

  
  // KT 21/10/98 use new order of arguments
  ask_filename_with_extension(initial_data_filename_char,
    "Get initial estimate from which file (1 = 1's): ", "");
  
  this->initial_data_filename=initial_data_filename_char;
  
  this->num_subsets= ask_num("Number of ordered sets: ", 1,100000000,1);
  this->num_subiterations=ask_num("Number of subiterations",
				  1,NumericInfo<int>().max_value(),this->num_subsets);
  
  this->start_subiteration_num=ask_num("Start at what subiteration number : ", 1,NumericInfo<int>().max_value(),1);
  
  this->start_subset_num=ask_num("Start with which ordered set : ",
    0,this->num_subsets-1,0);
  
  this->save_interval=ask_num("Save estimates at sub-iteration intervals of: ", 
			      1,this->num_subiterations,this->num_subiterations);  
  
  
  this->inter_iteration_filter_interval=
    ask_num("Do inter-iteration filtering at sub-iteration intervals of: ",
	    0, this->num_subiterations, 0);
  
  if(this->inter_iteration_filter_interval>0 )
  {
    cerr<<endl<<"Supply inter-iteration filter type:\nPossible values:\n";
    DataProcessor<TargetT>::list_registered_names(cerr);
    
    const string inter_iteration_filter_type = ask_string("");
    
    this->inter_iteration_filter_ptr = 
      DataProcessor<TargetT>::read_registered_object(0, inter_iteration_filter_type);      
  } 
  
  
  
  this->randomise_subset_order=
    ask("Randomly generate subset order?", false);
  
  
}


template <typename TargetT>
bool IterativeReconstruction<TargetT>::
post_processing() 
{
  if (base_type::post_processing())
    return true;

  if (is_null_ptr(this->objective_function_sptr))
    { warning("You have to specify an objective function"); return true; }

  if (this->initial_data_filename.length() == 0)
  { warning("You need to specify an initial estimate file"); return true; }

  return false;
}

//************ get_ functions ****************
template <typename TargetT>
GeneralisedObjectiveFunction<TargetT> const&
IterativeReconstruction<TargetT>::
get_objective_function() const
{ return *this->objective_function_sptr; }

template <typename TargetT>
const int
IterativeReconstruction<TargetT>::
get_max_num_full_iterations() const
{ return this->max_num_full_iterations; }

template <typename TargetT>
const int 
IterativeReconstruction<TargetT>::
get_num_subsets() const
{ return this->num_subsets; }

template <typename TargetT>
const int 
IterativeReconstruction<TargetT>::
get_num_subiterations() const
{ return this->num_subiterations; }

template <typename TargetT>
const int
IterativeReconstruction<TargetT>::
get_start_subiteration_num() const
{ return this->start_subiteration_num; }

template <typename TargetT>
const int 
IterativeReconstruction<TargetT>::
get_start_subset_num() const
{ return this->start_subset_num; }

template <typename TargetT>
const int
IterativeReconstruction<TargetT>::
get_save_interval() const
{ return this->save_interval; }

template <typename TargetT>
const bool
IterativeReconstruction<TargetT>::
get_randomise_subset_order() const
{ return this->randomise_subset_order; }

template <typename TargetT>
const DataProcessor<TargetT>& 
IterativeReconstruction<TargetT>::
get_inter_iteration_filter() const
{ return *this->inter_iteration_filter_ptr; }

template <typename TargetT>
const int 
IterativeReconstruction<TargetT>::
get_inter_iteration_filter_interval() const
{ return this->inter_iteration_filter_interval; }

template <typename TargetT>
const int 
IterativeReconstruction<TargetT>::
get_report_objective_function_values_interval() const
{ return this->report_objective_function_values_interval; }

//************ set_ functions ****************
template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_objective_function_sptr(const shared_ptr<GeneralisedObjectiveFunction<TargetT > >& arg)
{
  this->objective_function_sptr  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_max_num_full_iterations(const int arg)
{
  this->max_num_full_iterations  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_num_subsets(const int arg)
{
  this->num_subsets  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_num_subiterations(const int arg)
{
  this->num_subiterations  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_start_subiteration_num(const int arg)
{
  this->start_subiteration_num  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_start_subset_num(const int arg)
{
  this->start_subset_num  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_save_interval(const int arg)
{
  this->save_interval  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_randomise_subset_order(const bool arg)
{
  this->randomise_subset_order  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_inter_iteration_filter_ptr(const shared_ptr<DataProcessor<TargetT> >& arg)
{
  this->inter_iteration_filter_ptr  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_inter_iteration_filter_interval(const int arg)
{
  this->inter_iteration_filter_interval  = arg;
}

template <typename TargetT>
void
IterativeReconstruction<TargetT>::
set_report_objective_function_values_interval(const int arg)
{
  this->report_objective_function_values_interval = arg;
}

//************ other functions ****************
template <typename TargetT>
IterativeReconstruction<TargetT>::
IterativeReconstruction()
{
}

template <typename TargetT>
std::string
IterativeReconstruction<TargetT>::
make_filename_prefix_subiteration_num(const std::string& filename_prefix) const
{
  char num[50];
  sprintf(num, "_%d", subiteration_num);
  return filename_prefix + num;
}

template <typename TargetT>
std::string
IterativeReconstruction<TargetT>::
make_filename_prefix_subiteration_num() const
{
  return 
    this->make_filename_prefix_subiteration_num(this->output_filename_prefix);
}

template <typename TargetT>
TargetT *
IterativeReconstruction<TargetT>::get_initial_data_ptr() const
{
  if(this->initial_data_filename=="0")
  {
    return this->objective_function_sptr->construct_target_ptr();
  }
  else if(this->initial_data_filename=="1")
  {
    TargetT * target_data_ptr =
      this->objective_function_sptr->construct_target_ptr();    
    std::fill(target_data_ptr->begin_all(), target_data_ptr->end_all(), 1.F);
    return target_data_ptr;
  }
  else
    {
      return 
        TargetT::read_from_file(this->initial_data_filename);
    }
}

// KT 10122001 new
template <typename TargetT>
Succeeded 
IterativeReconstruction<TargetT>::
reconstruct() 
{
  this->start_timers();

  shared_ptr<TargetT > target_data_sptr =
    this->get_initial_data_ptr();
  if (this->set_up(target_data_sptr) == Succeeded::no)
    {
      this->stop_timers();
      return Succeeded::no;
    }

  this->stop_timers();

  return this->reconstruct(target_data_sptr);
}

template <typename TargetT>
Succeeded 
IterativeReconstruction<TargetT>::
reconstruct(shared_ptr<TargetT > const& target_data_sptr)
{

  this->start_timers();
#if 0
  if (this->set_up(target_data_sptr) == Succeeded::no)
    {
      this->stop_timers();
      return Succeeded::no;
    }
#endif

  for(subiteration_num=start_subiteration_num;subiteration_num<=num_subiterations && this->terminate_iterations==false; subiteration_num++)
  {
    this->update_estimate(*target_data_sptr);
    this->end_of_iteration_processing(*target_data_sptr);
  }

  this->stop_timers();

  cerr << "Total CPU Time " << this->get_CPU_timer_value() << "secs"<<endl;

  // currently, if there was something wrong, the programme is just aborted
  // so, if we get here, everything was fine
  return Succeeded::yes;

}


template <typename TargetT>
Succeeded
IterativeReconstruction<TargetT>::
set_up(shared_ptr<TargetT > const& target_data_sptr)
{
  if (base_type::set_up(target_data_sptr) == Succeeded::no)
    return Succeeded::no;

  //initialize iteration loop terminator
  this->terminate_iterations=false;


  if (this->num_subsets<1 )
    { warning("number of subsets should be positive"); return Succeeded::no; }

  if (is_null_ptr(this->objective_function_sptr))
    { warning("You have to specify an objective function"); return Succeeded::no; }

  const int new_num_subsets =
    this->objective_function_sptr->set_num_subsets(this->num_subsets);
  if (new_num_subsets!=this->num_subsets)
    {
      warning("Number of subsets requested : %d, but actual number used is %d\n",
	      this->num_subsets, new_num_subsets);
      this->num_subsets = new_num_subsets;
    }

  if (this->objective_function_sptr->set_up(target_data_sptr) == Succeeded::no)
    return Succeeded::no;

  if (this->num_subiterations<1)
    { warning("Range error in number of subiterations"); return Succeeded::no; }
  
  if(this->start_subset_num<0 || this->start_subset_num>=this->num_subsets) 
    { warning("Range error in starting subset"); return Succeeded::no; }

  if(this->save_interval<1 || this->save_interval>this->num_subiterations) 
    { warning("Range error in iteration save interval"); return Succeeded::no;}
 
  if (this->inter_iteration_filter_interval<0)
    { warning("Range error in inter-iteration filter interval "); return Succeeded::no; }

  if (this->start_subiteration_num<1)
    { warning("Range error in starting subiteration number"); return Succeeded::no; }
  
  ////////////////// subset order

  // KT 05/07/2000 made randomise_subset_order int
  if (this->randomise_subset_order!=0){
    srand((unsigned int) (time(NULL)) ); //seed the rand() function
  }


  // Building filters
  // This is not really necessary, as apply would call this anyway.
  // However, we have it here such that any errors in building the filters would
  // be caught before doing any projections or so done.
#if 0 
  /* 
     KT 04/06/2003 disabled the explicit calling of inter_iteration_filter_ptr->set_up()
  
     It was here to catch incompatibilities between the filter and the
     estimate early (i.e. before any real reconstruction stuff has been going on). Now
     this will only be caught when the inter_iteration_filter is called for the first time.

     The reason I disabled this is that OSMAPOSL::set_up (and presumably
     other algorithms that insist on non-negative data) chains the current
     inter_iteration_filter with a ThresholdMinToSmallPositiveValueDataProcessor. 
     This meant that the new data-processor was not set-up yet, and resulted 
     in the current filter being set-up twice, which might potentially take a lot 
     of CPU time.
  */
  if(this->inter_iteration_filter_interval>0 && is_null_ptr(this->inter_iteration_filter_ptr) )
    {
      cerr<<endl<<"Building inter-iteration filter kernel"<<endl;
      if (this->inter_iteration_filter_ptr->set_up(*target_data_sptr)
          == Succeeded::no)
	error("Error building inter iteration filter\n");
    }
#endif
 
  return Succeeded::yes;
}

template <typename TargetT>
void IterativeReconstruction<TargetT>::
end_of_iteration_processing(TargetT &current_estimate)
{

  if (this->report_objective_function_values_interval>0 &&
      (this->subiteration_num%this->report_objective_function_values_interval == 0
       || this->subiteration_num==this->num_subiterations))      
    {
      std::cerr << "Objective function values (before any additional filtering):\n"
		<< this->objective_function_sptr->
	             get_objective_function_values_report(current_estimate);
    }
  
  if(this->inter_iteration_filter_interval>0 && 
     !is_null_ptr(this->inter_iteration_filter_ptr) &&
     this->subiteration_num%this->inter_iteration_filter_interval==0)
    {
      cerr<<endl<<"Applying inter-iteration filter"<<endl;
      this->inter_iteration_filter_ptr->apply(current_estimate);
    }
 

  cerr<< this->method_info()
      << " subiteration #"<<subiteration_num<<" completed"<<endl;
  cerr << "  min and max in current estimate " 
       <<    *std::min_element(current_estimate.begin_all(), current_estimate.end_all())
       << " " 
       <<     *std::max_element(current_estimate.begin_all(), current_estimate.end_all()) << endl;
  
  if(this->subiteration_num==this->num_subiterations &&
     !is_null_ptr(this->post_filter_sptr) )
  {
    cerr<<endl<<"Applying post-filter"<<endl;
    this->post_filter_sptr->apply(current_estimate);
    
    cerr << "  min and max after post-filtering " 
       <<    *std::min_element(current_estimate.begin_all(), current_estimate.end_all())
       << " " 
       <<    *std::max_element(current_estimate.begin_all(), current_estimate.end_all()) << endl;
  }
  
    // Save intermediate (or last) iteration      
  if((!(this->subiteration_num%this->save_interval)) || 
     this->subiteration_num==this->num_subiterations ) 
    {      	         
      this->output_file_format_ptr->
	write_to_file(this->make_filename_prefix_subiteration_num(), 
		      current_estimate);
    }
}

template <typename TargetT>
VectorWithOffset<int> 
IterativeReconstruction<TargetT>::
randomly_permute_subset_order()
{

  VectorWithOffset<int> temp_array(this->num_subsets),final_array(this->num_subsets);
  int index;

 for(int i=0;i<this->num_subsets;i++) temp_array[i]=i;

 for (int i=0;i<this->num_subsets;i++)
   {

   index = (int) (((float)rand()/(float)RAND_MAX)*(this->num_subsets-i));
   if (index==this->num_subsets-i) index--;
   final_array[i]=temp_array[index];
 

   for (int j=index;j<this->num_subsets-(i+1);j++) 
     temp_array[j]=temp_array[j+1];

   }

 cerr<<endl<<"Generating new subset sequence: ";
 for(int i=0;i<this->num_subsets;i++) cerr<<final_array[i]<<" ";

 return final_array;

}



#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

template class IterativeReconstruction<DiscretisedDensity<3,float> >;
template class IterativeReconstruction<ParametricVoxelsOnCartesianGrid >; 

END_NAMESPACE_STIR



