//
// $Id$
//
/*!
  \file

  \brief non-inline implementations for IterativeReconstructionParameters

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
  
  \date $Date$

  \version $Revision$
*/

// TODO get rid of restriction of subsets according to 'view45'
// it's not appropriate for general symmetries

#include "recon_buildblock/IterativeReconstructionParameters.h" 
#include "NumericInfo.h"
#include "ImageFilter.h"
#include "utilities.h"
// for time(), used as seed for random stuff
#include <ctime>
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ends;
#endif

START_NAMESPACE_TOMO

IterativeReconstructionParameters::IterativeReconstructionParameters()
  : ReconstructionParameters()
{

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
  inter_iteration_filter_type = "separable cartesian metz";
  inter_iteration_filter_fwhmxy_dir = 0.0;
  inter_iteration_filter_fwhmz_dir = 0.0;

//MJ 22/02/99 added Metz filter capability
  inter_iteration_filter_Nxy_dir = 0.0;
  inter_iteration_filter_Nz_dir = 0.0;

  do_post_filtering = 0;
  post_filter_type = "separable cartesian metz";
  post_filter_fwhmxy_dir = 0.0;
  post_filter_fwhmz_dir = 0.0;

//MJ 22/02/99 added Metz filter capability
  post_filter_Nxy_dir = 0.0;
  post_filter_Nz_dir = 0.0;

//MJ 02/08/99 added subset randomization
  // KT 05/07/2000 made int
  randomise_subset_order = 0;


  add_key("number of subiterations", KeyArgument::INT, &num_subiterations);
  add_key("start at subiteration number", KeyArgument::INT, &start_subiteration_num);
  add_key("save images at subiteration intervals", KeyArgument::INT, &save_interval);
  add_key("zero end planes of segment 0", KeyArgument::INT, &zero_seg0_end_planes);
  add_key("initial image", KeyArgument::ASCII, &initial_image_filename);
  add_key("number of subsets", KeyArgument::INT, &num_subsets);
  add_key("start at subset", KeyArgument::INT, &start_subset_num);
  add_key("inter-iteration filter subiteration interval", KeyArgument::INT, &inter_iteration_filter_interval);
  add_key("inter-iteration filter type", KeyArgument::ASCII, &inter_iteration_filter_type);
  add_key("inter-iteration xy-dir filter FWHM (in mm)", KeyArgument::DOUBLE, &inter_iteration_filter_fwhmxy_dir);
  add_key("inter-iteration z-dir filter FWHM (in mm)", KeyArgument::DOUBLE, &inter_iteration_filter_fwhmz_dir);
  
  //MJ 22/02/99 added for Metz filtering
  add_key("inter-iteration xy-dir filter Metz power", KeyArgument::DOUBLE, &inter_iteration_filter_Nxy_dir);
  add_key("inter-iteration z-dir filter Metz power", KeyArgument::DOUBLE, &inter_iteration_filter_Nz_dir);
  
  add_key("do post-filtering", KeyArgument::INT,&do_post_filtering);
  // KT 25/05/2000 'post filter' -> post-filter
  add_key("post-filter type", KeyArgument::ASCII, &post_filter_type);
  add_key("post-filter xy-dir FWHM (in mm)", KeyArgument::DOUBLE, &post_filter_fwhmxy_dir);
  add_key("post-filter z-dir FWHM (in mm)", KeyArgument::DOUBLE, &post_filter_fwhmz_dir);
  //MJ 22/02/99 added for Metz filtering
  add_key("post-filter xy-dir Metz power", KeyArgument::DOUBLE, &post_filter_Nxy_dir);
  add_key("post-filter z-dir Metz power", KeyArgument::DOUBLE, &post_filter_Nz_dir);
  
  //MJ 02/08/99 added subset randomization
  add_key("uniformly randomise subset order", KeyArgument::INT, &randomise_subset_order);

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
      
    if(inter_iteration_filter_interval>0){  

      cerr<<endl<<"Supply inter-iteration filter parameters:"<<endl;
	
      //MJ 21/02/99 extended filter range & enabled Metz filtering
      //MJ 01/06/99 extended it again

	inter_iteration_filter_fwhmxy_dir= ask_num(" Full-width half-maximum (xy-dir) in mm?", 0.0,NumericInfo<double>().max_value(),0.0);
	
	inter_iteration_filter_fwhmz_dir=ask_num(" Full-width half-maximum (z-dir) in mm?", 0.0,NumericInfo<double>().max_value(),0.0);
	
	inter_iteration_filter_Nxy_dir= ask_num(" Metz power (xy-dir) ?", 0.0,NumericInfo<double>().max_value(),0.0);
	
	inter_iteration_filter_Nz_dir= ask_num(" Metz power (z-dir) ?", 0.0,NumericInfo<double>().max_value(),0.0);
	
    } 
    
    
    do_post_filtering=ask_num("Post-filter final image?", 0,1,0);
    
    
    
    if(do_post_filtering){ 
      
      cerr<<endl<<"Supply post-filter parameters:"<<endl;
	
	post_filter_fwhmxy_dir=
	 ask_num(" Full-width half-maximum (xy-dir) in mm?", 0.0,NumericInfo<double>().max_value(),0.0);
	
	post_filter_fwhmz_dir=
	 ask_num(" Full-width half-maximum (z-dir) in mm?", 0.0,NumericInfo<double>().max_value(),0.0);
	
	post_filter_Nxy_dir= ask_num(" Metz power (xy-dir) ?", 0.0,NumericInfo<double>().max_value(),0.0);
	
	post_filter_Nz_dir= ask_num(" Metz power (z-dir) ?", 0.0,NumericInfo<double>().max_value(),0.0);

   
    }


    // KT 05/07/2000 made int
    randomise_subset_order=
      ask("Randomly generate subset order?", false) ? 1 : 0;


}



bool IterativeReconstructionParameters::post_processing() 
{
  if (ReconstructionParameters::post_processing())
    return true;

  // KT&MJ 23/05/2000 PETerror->warning
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

  //TODO add more checking for filters

 if (start_subiteration_num<1)
   { warning("Range error in starting subiteration number\n"); return true; }

  ///////////////// consistency checks

 if(max_segment_num_to_process > proj_data_ptr->get_max_segment_num()) 
 { warning("Range error in number of segments\n"); return true;}
  
  if( num_subsets>proj_data_ptr->get_num_views()/4) 
  { warning("Range error in number of subsets\n"); return true;}
  
#if 0
  // requires target_image
  // TODO
  /////////////////// initialise filters
  
  if(inter_iteration_filter_interval>0 && !inter_iteration_filter.kernels_built )
    {
      cerr<<endl<<"Building inter-iteration filter kernel"<<endl;

      inter_iteration_filter.build(target_image,
					    inter_iteration_filter_fwhmxy_dir, 
					    inter_iteration_filter_fwhmz_dir,
					    (float) inter_iteration_filter_Nxy_dir,
					    (float) inter_iteration_filter_Nz_dir);
    }

 
  if(do_post_filtering && !post_filter.kernels_built)
    {
    cerr<<endl<<"Building post filter kernel"<<endl;

    post_filter.build(target_image,post_filter_fwhmxy_dir,
			       post_filter_fwhmz_dir,
			       (float) post_filter_Nxy_dir,
			       (float) post_filter_Nz_dir);
    }

#endif
  ////////////////// subset order

  // KT 05/07/2000 made randomise_subset_order int
  if (randomise_subset_order!=0){
   srand((unsigned int) (time(NULL)) ); //seed the rand() function
   }

  return false;
}


// KT&MJ 230899 changed return-type to string
string IterativeReconstructionParameters::parameter_info() const
{
  
  //cout << "\n\nOSEMParameters := " << endl;

  //ReconstructionParameters::parameter_info();
  // TODO dangerous for out-of-range, but 'old-style' ostrstream seems to need this

  char str[10000];
  ostrstream s(str, 10000);

  // KT 160899 do not write to cout, but append
  //string a = ReconstructionParameters::parameter_info();
  //cout << a << endl;
  s << ReconstructionParameters::parameter_info();
    
  s << "number of subsets := "
        << num_subsets << endl;
  s << "start at subset := "
        << start_subset_num << endl;
  s << "number of subiterations := "
        << num_subiterations << endl;
  // KT 25/05/2000 'starting' -> 'start at'
  s << "start at subiteration number := "
     << start_subiteration_num << endl;
  s << "save images at subiteration intervals := "
        << save_interval << endl;
  s << "zero end planes of segment 0 := "
        << zero_seg0_end_planes << endl;
  s << "initial image := "
        << initial_image_filename << endl;
  s << "inter-iteration filter subiteration interval := " 
    << inter_iteration_filter_interval << endl;
  s << "inter-iteration filter type := " 
    << inter_iteration_filter_type << endl;
  s << "inter-iteration xy-dir filter FWHM (in mm) := " 
    << inter_iteration_filter_fwhmxy_dir << endl;
  s << "inter-iteration z-dir filter FWHM (in mm) := "
    << inter_iteration_filter_fwhmz_dir << endl;

//MJ 22/02/99 enabled Metz
  s << "inter-iteration xy-dir filter Metz power := " 
    << inter_iteration_filter_Nxy_dir << endl;
  s << "inter-iteration z-dir filter Metz power := "
    << inter_iteration_filter_Nz_dir << endl;

  s << "do post-filtering := " 
    << do_post_filtering << endl;
  s << "post-filter type := " 
    << post_filter_type << endl;
  s << "post-filter xy-dir FWHM (in mm) := " 
    << post_filter_fwhmxy_dir << endl;
  s << "post-filter z-dir FWHM (in mm) := "
    << post_filter_fwhmz_dir << endl;

//MJ 22/02/99 enabled Metz
  s << "post-filter xy-dir Metz power := " 
    << post_filter_Nxy_dir << endl;
  s << "post-filter z-dir Metz power := "
    << post_filter_Nz_dir << endl;

//MJ 02/08/99 added subset randomization
  s << "uniformly randomise subset order := "
    << randomise_subset_order<<endl<<endl;

  s<<ends;
  
  return s.str();
}


END_NAMESPACE_TOMO
