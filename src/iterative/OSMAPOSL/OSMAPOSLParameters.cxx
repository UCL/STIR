//
// $Id$: $Date$
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
  
  // KT 17/08/2000 3 new parameters

  maximum_relative_change = NumericInfo<float>().max_value();
  minimum_relative_change = 1/ NumericInfo<float>().max_value();
  add_key("maximum relative change", KeyArgument::DOUBLE, &maximum_relative_change);
  add_key("minimum relative change", KeyArgument::DOUBLE, &minimum_relative_change);
 
  write_update_image = 0;
  add_key("write update image", KeyArgument::INT, &write_update_image);

  // do parsing
  initialise(parameter_filename);

}



void OSMAPOSLParameters::ask_parameters()
{

  MAPParameters::ask_parameters();

  // KT 05/07/2000 made enforce_initial_positivity int
  enforce_initial_positivity=
    ask("Enforce initial positivity condition?",true) ? 1 : 0;

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

  // KT 17/08/2000 3 new parameters
  const double max_in_double = static_cast<double>(NumericInfo<float>().max_value());
  maximum_relative_change = ask_num("maximum relative change",
      1.,max_in_double,max_in_double);
  minimum_relative_change = ask_num("minimum relative change",
      1/max_in_double,1.,1/max_in_double);
  
  write_update_image = ask_num("write update image", 0,1,0);

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

  // KT 17/08/2000 3 new parameters

  s << "maximum relative change := "
    << maximum_relative_change << endl;
  s << "minimum relative change := "
    << minimum_relative_change << endl;

  s << "write update image := "
    << write_update_image << endl;

  s<<ends;

  return s.str();

}

END_NAMESPACE_TOMO
