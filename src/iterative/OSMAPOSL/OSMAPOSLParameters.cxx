//
// $Id$
//
/*!

  \file
  \ingroup OSMAPOSL
  
  \brief  implementation of the OSMAPOSLParameters class 
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
      
  \date $Date$
        
  \version $Revision$
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
     : MAPParameters()
{

  enforce_initial_positivity = 1;
  inter_update_filter_interval = 0;
  inter_update_filter_type = "separable cartesian metz";
  inter_update_filter_fwhmxy_dir = 0.0;
  inter_update_filter_fwhmz_dir = 0.0;
  inter_update_filter_Nxy_dir = 0.0;
  inter_update_filter_Nz_dir = 0.0;

  add_key("OSMAPOSLParameters", 
    KeyArgument::NONE, &KeyParser::start_parsing);
  
  add_key("enforce initial positivity condition", KeyArgument::INT, &enforce_initial_positivity);
  add_key("inter-update filter subiteration interval", KeyArgument::INT, &inter_update_filter_interval);
  add_key("inter-update filter type", KeyArgument::ASCII, &inter_update_filter_type);
  add_key("inter-update xy-dir filter FWHM (in mm)", KeyArgument::DOUBLE, &inter_update_filter_fwhmxy_dir);
  add_key("inter-update z-dir filter FWHM (in mm)", KeyArgument::DOUBLE, &inter_update_filter_fwhmz_dir);
  
  add_key("inter-update xy-dir filter Metz power", KeyArgument::DOUBLE, &inter_update_filter_Nxy_dir);
  add_key("inter-update z-dir filter Metz power", KeyArgument::DOUBLE, &inter_update_filter_Nz_dir);
  
  
  // do parsing
  initialise(parameter_filename);

}



void OSMAPOSLParameters::ask_parameters()
{

  MAPParameters::ask_parameters();


  enforce_initial_positivity=ask("Enforce initial positivity condition?",true);

  inter_update_filter_interval=
    ask_num("Do inter-update filtering at sub-iteration intervals of: ",0, num_subiterations, 0);
     
  if(inter_update_filter_interval>0)
    {       
	

      cerr<<endl<<"Supply inter-update filter parameters:"<<endl;

      inter_update_filter_fwhmxy_dir= ask_num(" Full-width half-maximum (xy-dir) in mm ?", 0.0,NumericInfo<double>().max_value(),0.0);
	
      inter_update_filter_fwhmz_dir=ask_num(" Full-width half-maximum (z-dir) in mm?", 0.0,NumericInfo<double>().max_value(),0.0);
	
      inter_update_filter_Nxy_dir= ask_num(" Metz power (xy-dir)?", 0.0,NumericInfo<double>().max_value(),0.0);
	
      inter_update_filter_Nz_dir= ask_num(" Metz power (z-dir)?", 0.0,NumericInfo<double>().max_value(),0.0);
	
    } 

}




bool OSMAPOSLParameters::post_processing()
{
  if (MAPParameters::post_processing())
    return true;


  if (inter_update_filter_interval<0)
    { warning("Range error in inter-update filter interval \n"); return true; }

  return false;
}


// KT&MJ 230899 changed return-type to string
string OSMAPOSLParameters::parameter_info() const
{
  

  // TODO dangerous for out-of-range, but 'old-style' ostrstream seems to need this

  char str[10000];
  ostrstream s(str, 10000);

  s << MAPParameters::parameter_info();

  s << "enforce initial positivity condition :="
    << enforce_initial_positivity<<endl;
  s << "inter-update filter subiteration interval := " 
    << inter_update_filter_interval << endl;
  s << "inter-update filter type := " 
    << inter_update_filter_type << endl;
  s << "inter-update xy-dir filter FWHM (in mm) := " 
    << inter_update_filter_fwhmxy_dir << endl;
  s << "inter-update z-dir filter FWHM (in mm) := "
    << inter_update_filter_fwhmz_dir << endl;
  s << "inter-update xy-dir filter Metz power := " 
    << inter_update_filter_Nxy_dir << endl;
  s << "inter-update z-dir filter Metz power := "
    << inter_update_filter_Nz_dir << endl<<endl;  

  s<<ends;

  return s.str();

}

END_NAMESPACE_TOMO
