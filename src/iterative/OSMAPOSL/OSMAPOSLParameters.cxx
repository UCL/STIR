//
// $Id$
//
/*!
  \file
  \ingroup OSMAPOSL
  \brief Implementations for OSMAPOSLParameters.cxx

  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  \date $Date$
  \version $Revision $
*/


#include "OSMAPOSL/OSMAPOSLParameters.h"

#include "NumericInfo.h"
#include "utilities.h"
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ends;
#endif

START_NAMESPACE_TOMO


OSMAPOSLParameters::OSMAPOSLParameters(const string& parameter_filename)
     : LogLikelihoodBasedAlgorithmParameters()
{
  // do parsing
  initialise(parameter_filename);

}

void 
OSMAPOSLParameters::set_defaults()
{
  LogLikelihoodBasedAlgorithmParameters::set_defaults();
  enforce_initial_positivity = 1;
  // KT 17/08/2000 3 new parameters
  maximum_relative_change = NumericInfo<float>().max_value();
  minimum_relative_change = 0;
  write_update_image = 0;
  inter_update_filter_interval = 0;
  inter_update_filter_ptr = 0;
  MAP_model="additive"; 
  prior_ptr = 0;
}

void
OSMAPOSLParameters::initialise_keymap()
{
  LogLikelihoodBasedAlgorithmParameters::initialise_keymap();
  parser.add_start_key("OSMAPOSLParameters");
  parser.add_stop_key("End");
  
  parser.add_key("enforce initial positivity condition",&enforce_initial_positivity);
  parser.add_key("inter-update filter subiteration interval",&inter_update_filter_interval);
  //add_key("inter-update filter type", KeyArgument::ASCII, &inter_update_filter_type);
  parser.add_parsing_key("inter-update filter type", &inter_update_filter_ptr);
  parser.add_parsing_key("Prior type", &prior_ptr);
  parser.add_key("MAP_model", &MAP_model);
  parser.add_key("maximum relative change", &maximum_relative_change);
  parser.add_key("minimum relative change",&minimum_relative_change);
  parser.add_key("write update image",&write_update_image);

   
}




void OSMAPOSLParameters::ask_parameters()
{

  LogLikelihoodBasedAlgorithmParameters::ask_parameters();

  // KT 05/07/2000 made enforce_initial_positivity int
  enforce_initial_positivity=
    ask("Enforce initial positivity condition?",true) ? 1 : 0;

  inter_update_filter_interval=
    ask_num("Do inter-update filtering at sub-iteration intervals of: ",0, num_subiterations, 0);
     
  if(inter_update_filter_interval>0)
  {       
    
    cerr<<endl<<"Supply inter-update filter type:\nPossible values:\n";
    ImageProcessor<3,float>::list_registered_names(cerr);
    
    const string inter_update_filter_type = ask_string("");
    
    inter_update_filter_ptr = 
      ImageProcessor<3,float>::read_registered_object(0, inter_update_filter_type);      
    
  } 

 if(ask("Include prior?",false))
  {       
    
    cerr<<endl<<"Supply prior type:\nPossible values:\n";
    GeneralisedPrior<float>::list_registered_names(cerr);
    
    const string prior_type = ask_string("");
    
    prior_ptr = 
      GeneralisedPrior<float>::read_registered_object(0, prior_type); 
    
    MAP_model = 
      ask_string("Use additive or multiplicative form of MAP-OSL ('additive' or 'multiplicative')","additive");

    
  } 
  
  // KT 17/08/2000 3 new parameters
  const double max_in_double = static_cast<double>(NumericInfo<float>().max_value());
  maximum_relative_change = ask_num("maximum relative change",
      1.,max_in_double,max_in_double);
  minimum_relative_change = ask_num("minimum relative change",
      0.,1.,0.);
  
  write_update_image = ask_num("write update image", 0,1,0);

}




bool OSMAPOSLParameters::post_processing()
{
  if (LogLikelihoodBasedAlgorithmParameters::post_processing())
    return true;


  if (inter_update_filter_interval<0)
    { warning("Range error in inter-update filter interval \n"); return true; }
  
  if (prior_ptr != 0)
  {
    if (MAP_model != "additive" && MAP_model != "multiplicative")
    {
      warning("MAP model should have as value 'additive' or 'multiplicative', while it is '%s'\n",
	MAP_model.c_str());
      return true;
    }
  }
  return false;
}

END_NAMESPACE_TOMO
