//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
\ingroup OSSPS  
\ingroup reconstructors
\brief  implementation of the OSSPSReconstruction class 

\author Sanida Mustafovic
\author Kris Thielemans
  
$Date$
$Revision$
*/

#include "local/stir/OSSPS/OSSPSReconstruction.h"
#include "stir/min_positive_element.h"
#include "stir/DiscretisedDensity.h"
#include "stir/LogLikBased/common.h"
#include "stir/recon_buildblock/PriorWithParabolicSurrogate.h"
#include "local/stir/recon_buildblock/QuadraticPrior.h" // necessary for recompute_penalty_term_in_denominator
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/Succeeded.h"
#include "stir/recon_array_functions.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/NumericInfo.h"
#include "stir/utilities.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/RelatedViewgrams.h"
#include <iostream>

#include <memory>
#include <iostream>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#ifndef STIR_NO_NAMESPACES
using std::auto_ptr;
using std::cerr;
using std::ends;
using std::endl;
#endif


START_NAMESPACE_STIR

//*************** parameters *************


void 
OSSPSReconstruction::set_defaults()
{
  LogLikelihoodBasedReconstruction::set_defaults();
  enforce_initial_positivity = 1;
  // KT 17/08/2000 3 new parameters
  maximum_relative_change = NumericInfo<float>().max_value();
  minimum_relative_change = 0;
  write_update_image = 0;
  precomputed_denominator_filename = "";
  forward_projection_of_all_ones_filename = "";

  normalisation_sptr = new TrivialBinNormalisation;

  //MAP_model="additive"; 
  prior_ptr = 0;
  relaxation_parameter = 0;
  relaxation_gamma = 0.1F;
}

void
OSSPSReconstruction::initialise_keymap()
{
  LogLikelihoodBasedReconstruction::initialise_keymap();
  parser.add_start_key("OSSPSParameters");
  parser.add_stop_key("End");
  
  parser.add_key("enforce initial positivity condition",&enforce_initial_positivity);
  // parser.add_key("inter-update filter subiteration interval",&inter_update_filter_interval);
  // //add_key("inter-update filter type", KeyArgument::ASCII, &inter_update_filter_type);
  // parser.add_parsing_key("inter-update filter type", &inter_update_filter_ptr);
  parser.add_parsing_key("Prior type", &prior_ptr);
  //parser.add_key("MAP_model", &MAP_model);
  parser.add_key("maximum relative change", &maximum_relative_change);
  parser.add_key("minimum relative change",&minimum_relative_change);
  parser.add_key("write update image",&write_update_image);   
  parser.add_key("precomputed denominator", &precomputed_denominator_filename);
  parser.add_key("forward_projection of all ones", &forward_projection_of_all_ones_filename);

  parser.add_parsing_key("Normalisation type", &normalisation_sptr);

  parser.add_key("relaxation parameter", &relaxation_parameter);
  parser.add_key("relaxation gamma", &relaxation_gamma);
  
}


void OSSPSReconstruction::ask_parameters()
{
  error("Currently incomplete code. Use a parameter file. Sorry.");

  LogLikelihoodBasedReconstruction::ask_parameters();
  
  // KT 05/07/2000 made enforce_initial_positivity int
  enforce_initial_positivity=
    ask("Enforce initial positivity condition?",true) ? 1 : 0;
  
  char precomputed_denominator_filename_char[max_filename_length];
  
  ask_filename_with_extension(precomputed_denominator_filename_char,"Enter file name of precomputed denominator  (1 = 1's): ", "");   
  
  precomputed_denominator_filename=precomputed_denominator_filename_char;
  


  if(ask("Include prior?",false))
  {       
    
    cerr<<endl<<"Supply prior type:\nPossible values:\n";
    GeneralisedPrior<float>::list_registered_names(cerr);
    
    const string prior_type = ask_string("");
    
    prior_ptr = 
      GeneralisedPrior<float>::read_registered_object(0, prior_type); 
    
  } 
  
  // KT 17/08/2000 3 new parameters
  const double max_in_double = static_cast<double>(NumericInfo<float>().max_value());
  maximum_relative_change = ask_num("maximum relative change",
    1.,max_in_double,max_in_double);
  minimum_relative_change = ask_num("minimum relative change",
    0.,1.,0.);
  
  write_update_image = ask_num("write update image", 0,1,0);

  // TODO some more parameters here (relaxation et al)
}




bool OSSPSReconstruction::post_processing()
{
  if (LogLikelihoodBasedReconstruction::post_processing())
    return true;
    
  // KT 09/12/2002 one more check
  if (!is_null_ptr(prior_ptr) && dynamic_cast<PriorWithParabolicSurrogate<float>*>(prior_ptr.get())==0)
  {
    warning("Prior must be of a type derived from PriorWithParabolicSurrogate\n");
    return true;
  }

  if (is_null_ptr(normalisation_sptr))
  {
    warning("bin normalisation object invalid");
    return true;
  }

  return false;
}

//*************** other functions *************

OSSPSReconstruction::
OSSPSReconstruction()
{  
  set_defaults();
}

OSSPSReconstruction::
OSSPSReconstruction(const string& parameter_filename)
{    
  initialise(parameter_filename);
  cerr<<parameter_info();
}

string OSSPSReconstruction::method_info() const
{
  
  // TODO add prior name?
  
#ifdef BOOST_NO_STRINGSTREAM
  char str[10000];
  ostrstream s(str, 10000);
#else
  std::ostringstream s;
#endif
  
  // if(inter_update_filter_interval>0) s<<"IUF-";
  if (!is_null_ptr(prior_ptr) && prior_ptr->get_penalisation_factor() != 0)
    s << "MAP-";
  if(num_subsets>1) 
    s<<"OS-";
  s << "SPS";
  if(inter_iteration_filter_interval>0) s<<"S";  

  return s.str();
}

Succeeded 
OSSPSReconstruction::
precompute_denominator_of_conditioner_without_penalty()
{
            
  CPUTimer timer;
  timer.reset();
  timer.start();

  assert(precomputed_denominator_ptr->find_min() == 0);
  assert(precomputed_denominator_ptr->find_max() == 0);

  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    projector_pair_ptr->get_symmetries_used()->clone();

  // TODO replace by boost::scoped_ptr
  std::auto_ptr<DiscretisedDensity<3,float> > image_full_of_ones_aptr;
  if (is_null_ptr(fwd_ones_sptr))
    {
      image_full_of_ones_aptr =
	std::auto_ptr<DiscretisedDensity<3,float> >
	( precomputed_denominator_ptr->clone());
      image_full_of_ones_aptr->fill(1);
    }

  for (int segment_num = -max_segment_num_to_process;
       segment_num<= max_segment_num_to_process;
       ++segment_num) 
    {      
      for (int view = proj_data_ptr->get_min_view_num(); 
	   view <= proj_data_ptr->get_max_view_num(); 
	   ++view)
	{
	  const ViewSegmentNumbers view_segment_num(view, segment_num);
	  
	  if (!symmetries_sptr->is_basic(view_segment_num))
	    continue;

	  // first compute data-term: y/norm^2
	  RelatedViewgrams<float> viewgrams =
	    proj_data_ptr->get_related_viewgrams(view_segment_num, symmetries_sptr);

	  // TODO insert sensible frame start and end times
	  normalisation_sptr->apply(viewgrams, 0,1);

	  // smooth TODO

	  // TODO insert sensible frame start and end times
	  normalisation_sptr->apply(viewgrams, 0,1);

	  RelatedViewgrams<float> tmp_viewgrams;
	  // set tmp_viewgrams to geometric forward projection of all ones
	  if (is_null_ptr(fwd_ones_sptr))
	    {
	      tmp_viewgrams = proj_data_ptr->get_empty_related_viewgrams(view_segment_num, symmetries_sptr);
	      projector_pair_ptr->get_forward_projector_sptr()->
		forward_project(tmp_viewgrams, *image_full_of_ones_aptr);
	    }
	  else
	    {
	      tmp_viewgrams = fwd_ones_sptr->get_related_viewgrams(view_segment_num, symmetries_sptr);
	    }
	  
	  // now divide by the data term
	  {
	    int tmp1=0, tmp2=0;// ignore counters returned by divide_and_truncate
	    divide_and_truncate(tmp_viewgrams, viewgrams, 0, tmp1, tmp2);
	  }

	  // back-project
	  projector_pair_ptr->get_back_projector_sptr()->back_project(*precomputed_denominator_ptr, tmp_viewgrams);
      }

  } // end of loop over segments

  timer.stop();
  cerr << "Precomputing denominator took " << timer.value() << " s CPU time\n";
  cerr << "min and max in precomputed denominator " << precomputed_denominator_ptr->find_min()
       << ", " << precomputed_denominator_ptr->find_max() << endl;
  

  // Write it to file
  {
    std::string fname =  
      get_parameters().output_filename_prefix +
      "_precomputed_denominator";
    
      cerr <<"  - Saving " << fname << endl;
      get_parameters().output_file_format_ptr->
	write_to_file(fname, *precomputed_denominator_ptr);
  }

  return Succeeded::yes;
}

void OSSPSReconstruction::recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr)
{
  LogLikelihoodBasedReconstruction::recon_set_up(target_image_ptr);
  
  if (max_segment_num_to_process==-1)
    max_segment_num_to_process =
    proj_data_ptr->get_max_segment_num();
  
  if(enforce_initial_positivity) 
    threshold_min_to_small_positive_value(target_image_ptr->begin_all(),
					  target_image_ptr->end_all(),
					  10.E-6F);
  
  if (get_parameters().forward_projection_of_all_ones_filename!="")
    fwd_ones_sptr = ProjData::read_from_file(forward_projection_of_all_ones_filename);

  if(get_parameters().precomputed_denominator_filename=="")
  {
    precomputed_denominator_ptr=target_image_ptr->get_empty_discretised_density();
    precompute_denominator_of_conditioner_without_penalty();
  }
  else if(get_parameters().precomputed_denominator_filename=="1")
  {
    precomputed_denominator_ptr=target_image_ptr->get_empty_discretised_density();
    precomputed_denominator_ptr->fill(1.0);  
  }
  else
  {       
    precomputed_denominator_ptr = 
      DiscretisedDensity<3,float>::read_from_file(get_parameters().precomputed_denominator_filename);   
    if (precomputed_denominator_ptr->get_index_range() != 
        target_image_ptr->get_index_range())
    {
      error("OSSPS: precomputed_denominator should have same index range as target image.");
      // TODO return Succeeded::no;
    }    
  }  
}



/*! \brief OSSPS additive update at every subiteration
  \warning This modifies *precomputed_denominator_ptr. So, you <strong>have to</strong>
  call recon_set_up() before running a new reconstruction.
  */
void OSSPSReconstruction::update_image_estimate(DiscretisedDensity<3,float> &current_image_estimate)
{
  static int count=0;
  // every time it's called, counter is incremented
  count++;
  
  // KT 09/12/2202 new variable
  // Check if we need to recompute the penalty yerm in the denominator.
  // For the quadratic prior, this is independent of the image (only on kappa's)
  // And of course, it's also independent when there is no prior
  const bool recompute_penalty_term_in_denominator =
    relaxation_parameter>0 && !is_null_ptr(prior_ptr) &&
    prior_ptr->get_penalisation_factor()>0 &&
    is_null_ptr(dynamic_cast<QuadraticPrior<float> const* >(get_parameters().prior_ptr.get())) ;
#ifndef PARALLEL
  //CPUTimer subset_timer;
  //subset_timer.start();
#else // PARALLEL
  PTimer timerSubset;
  timerSubset.Start();
#endif // PARALLEL
  
  //For ordered set processing  
  static VectorWithOffset<int> subset_array(get_parameters().num_subsets);  
  if(randomise_subset_order && (subiteration_num-1)%num_subsets==0)
  {
    subset_array = randomly_permute_subset_order();    
    cerr<<endl<<"current subset order:"<<endl;    
    for(int i=subset_array.get_min_index();i<=subset_array.get_max_index();i++) 
      cerr<<subset_array[i]<<" ";
  };
  
  const int subset_num=randomise_subset_order 
    ? subset_array[(subiteration_num-1)%num_subsets] 
    : (subiteration_num+start_subset_num-1)%num_subsets;
  
  cerr<<endl<<"Now processing subset #: "<<subset_num<<endl;
    
  // TODO make member or static parameter to avoid reallocation all the time
  auto_ptr< DiscretisedDensity<3,float> > numerator_ptr =
    auto_ptr< DiscretisedDensity<3,float> >(current_image_estimate.get_empty_discretised_density());
  

  // KT 05/07/2000 made zero_seg0_end_planes int
  // This is L' term (upto the (sub)sensitivity)
  distributable_compute_gradient(*numerator_ptr, 
				 current_image_estimate, 
				 proj_data_ptr, 
				 subset_num, 
				 num_subsets, 
				 -max_segment_num_to_process, // KT 30/05/2002 use new convention of distributable_* functions
				 max_segment_num_to_process, 
				 zero_seg0_end_planes!=0, 
				 NULL, 
				 additive_projection_data_ptr);
  
  // subtract sensitivity image
  *numerator_ptr *= num_subsets;
  *numerator_ptr -= *sensitivity_image_ptr;
  
  cerr << "num_subsets*subgradient L : max " << numerator_ptr->find_max();
  cerr << ", min " << numerator_ptr->find_min() << endl;

  
  auto_ptr< DiscretisedDensity<3,float> > work_image_ptr = 
    auto_ptr< DiscretisedDensity<3,float> >(current_image_estimate.get_empty_discretised_density());
      
  //relaxation_parameter ~1/n where n is iteration number 
  // KT 09/12/2002 added brackets around subiteration_num/num_subsets to force integer division
  const float relaxation_parameter = get_parameters().relaxation_parameter/
	      (1+get_parameters().relaxation_gamma*(subiteration_num/num_subsets));
    
  // subtract gradient of penalty
  // KT 09/12/2002 avoid work (or crash) when penalty is 0
  if (relaxation_parameter>0 && !is_null_ptr(prior_ptr)
      && prior_ptr->get_penalisation_factor()>0)
  {
    prior_ptr->compute_gradient(*work_image_ptr, current_image_estimate); 
    *numerator_ptr -= *work_image_ptr;   

    cerr << "total gradient max " << numerator_ptr->find_max();
    cerr << ", min " << numerator_ptr->find_min() << endl;
  }
  
  // now divide by denominator

  // KT 09/12/2002 only recompute it when necessary
  if (recompute_penalty_term_in_denominator || count==1)
  {
    // KT 09/12/2002 avoid work (or crash) when penalty is 0
    if (relaxation_parameter>0 && !is_null_ptr(prior_ptr)
      && prior_ptr->get_penalisation_factor()>0)
    {
      static_cast<PriorWithParabolicSurrogate<float>&>(*prior_ptr).
        parabolic_surrogate_curvature(*work_image_ptr, current_image_estimate);   
      *work_image_ptr *= 2;
      *work_image_ptr += *precomputed_denominator_ptr ;
    }
    else
      *work_image_ptr = *precomputed_denominator_ptr ;
    
    // KT 09/12/2002 new
    // avoid division by 0 by thresholding the denominator to be strictly positive
    // note that zeroes should really only occur where the sensitivity is 0
    threshold_min_to_small_positive_value(work_image_ptr->begin_all(),
					  work_image_ptr->end_all(),
					  10.E-6F);
    cerr << " denominator max " << work_image_ptr->find_max();
    cerr << ", min " << work_image_ptr->find_min() << endl;

    if (!recompute_penalty_term_in_denominator)
      {
	// store for future use
	*precomputed_denominator_ptr = *work_image_ptr;
      }
    
    *numerator_ptr /= *work_image_ptr;
  }
  else
  {
    // we have computed the denominator already 
    *numerator_ptr /= *precomputed_denominator_ptr;
  }

  if ( relaxation_parameter>0)
   *numerator_ptr *= relaxation_parameter;  
  
  // TODO move below thresholding?
  if (write_update_image)
  {
    // allocate space for the filename assuming that
    // we never have more than 10^49 subiterations ...
    char * fname = new char[output_filename_prefix.size() + 60];
    sprintf(fname, "%s_update_%d", get_parameters().output_filename_prefix.c_str(), subiteration_num);
    
    // Write it to file
    output_file_format_ptr->write_to_file(fname, *numerator_ptr);
    delete fname;
  }
  
  {
    cerr << "additive update image min,max: " 
	 << numerator_ptr->find_min()
	 << ", " 
	 << numerator_ptr->find_max()
	 << endl;
  }  
  current_image_estimate += *numerator_ptr; 
 
  // set all voxels to 0 for which the sensitivity is 0. These cannot be estimated.
  // Any such nonzero voxel results from a 0 backprojection, but non-zero prior gradient.
  {
    DiscretisedDensity<3,float>::full_iterator image_iter = current_image_estimate.begin_all();
    DiscretisedDensity<3,float>::const_full_iterator sens_iter = sensitivity_image_ptr->begin_all_const();
       
    for (;
       image_iter != current_image_estimate.end_all();
       ++image_iter, ++sens_iter)
      if (*sens_iter == 0)
        *image_iter = 0;
    assert(sens_iter == sensitivity_image_ptr->end_all_const());
  }
  // now threshold image
  {
    const float current_min =
      current_image_estimate.find_min();
    const float current_max = 
      current_image_estimate.find_max();
    const float new_min = 0.F;
    const float new_max = 
      static_cast<float>(maximum_relative_change);
    cerr << "current image old min,max: " 
      << current_min
      << ", " 
      << current_max
      << ", new min,max " 
      << max(current_min, new_min) << ", " << min(current_max, new_max)
      << endl;
    
    threshold_upper_lower(current_image_estimate.begin_all(),
			  current_image_estimate.end_all(), 
			  new_min, new_max);      
  }  
  
#ifndef PARALLEL
  //cerr << "Subset : " << subset_timer.value() << "secs " <<endl;
#else // PARALLEL
  timerSubset.Stop();
  cerr << "Subset: " << timerSubset.GetTime() << "secs" << endl;

#endif
  
}


END_NAMESPACE_STIR
