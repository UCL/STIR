//
// $Id$
//



#include "recon_buildblock/IterativeReconstruction.h"
#include "interfile.h"
#include "recon_buildblock/timers.h"
#include "PTimer.h"
#include "display.h"
// for time(), used as seed for random stuff
#include <ctime>

//
//
//---------------IterativeReconstruction definitions-----------------
//
//

//IterativeReconstruction::IterativeReconstruction(char* parameter_filename)
void IterativeReconstruction::IterativeReconstruction_ctor(char* parameter_filename)
{

 

if(strlen(parameter_filename)==0)
  {
    cerr << "Next time, try passing the executable a parameter file"
	 << endl;

     get_parameters().ask_parameters();

  }

else
  {

  if (!get_parameters().parse(parameter_filename))
    {
      PETerror("Error parsing input file %s, exiting\n", parameter_filename);
      exit(1);
    }

   get_parameters().proj_data_ptr= new PETSinogramOfVolume(read_interfile_PSOV(get_parameters().input_filename.c_str()));
    cerr<<endl;
  }

 ///////////////// consistency checks

 if(get_parameters().max_segment_num_to_process > get_parameters().proj_data_ptr->get_max_segment()) 
  {cerr<<"Range error in number of segments"<<endl; exit(1);}
  
  if( get_parameters().num_subsets>get_parameters().proj_data_ptr->get_num_views()/4) 
  {cerr<<"Range error in number of subsets"<<endl; exit(1);}
  

  //initialize iteration loop terminator
  terminate_iterations=false;

 
}

//IterativeReconstruction::~IterativeReconstruction()
void IterativeReconstruction::IterativeReconstruction_dtor()
{

  delete get_parameters().proj_data_ptr;


}

//MJ possibly we'll have to insert another level into
//the hierarchy between IterativeReconstruction and LogLikelihoodBasedReconstruction
// and move the code below there.
void IterativeReconstruction::reconstruct(PETImageOfVolume &target_image)
{

  // KTyyy
  start_timer_tot;
  Start_PTimer(tot);

  recon_set_up(target_image);

  for(subiteration_num=get_parameters().start_subiteration_num;subiteration_num<=get_parameters().num_subiterations && terminate_iterations==false; subiteration_num++)
  {

    update_image_estimate(target_image);
    end_of_iteration_processing(target_image);

  }




  // KTyyy
  stop_timer_tot;
  Stop_PTimer(tot);  // Real Time 


  print_timer_tot(cout);
  cout << " Real Time : \n";  
  Print_PTimer_tot(cout);


  if (get_parameters().disp)
    // KTxxx forget about conversion to Tensor3D
    display(target_image, target_image.find_max());


}

void IterativeReconstruction::iterative_common_recon_set_up(PETImageOfVolume &target_image)
{

  //MJ The following could probably go in the 
  //constructor, but KT wants to include it in timing measurements
   

  if(get_parameters().inter_iteration_filter_interval>0 && !get_parameters().inter_iteration_filter.kernels_built )
    {
      cerr<<endl<<"Building inter-iteration filter kernel"<<endl;

      get_parameters().inter_iteration_filter.build(target_image,
					    get_parameters().inter_iteration_filter_fwhmxy_dir, 
					    get_parameters().inter_iteration_filter_fwhmz_dir,
					    (float) get_parameters().inter_iteration_filter_Nxy_dir,
					    (float) get_parameters().inter_iteration_filter_Nz_dir);
    }

 
  if(get_parameters().do_post_filtering && !get_parameters().post_filter.kernels_built)
    {
    cerr<<endl<<"Building post filter kernel"<<endl;

    get_parameters().post_filter.build(target_image,get_parameters().post_filter_fwhmxy_dir,
			       get_parameters().post_filter_fwhmz_dir,
			       (float) get_parameters().post_filter_Nxy_dir,
			       (float) get_parameters().post_filter_Nz_dir);
    }

  if (get_parameters().randomise_subset_order){
   srand((unsigned int) (time(NULL)) ); //seed the rand() function
   }

}

//void IterativeReconstruction::end_of_iteration_processing(PETImageOfVolume &current_image_estimate)
void IterativeReconstruction::iterative_common_end_of_iteration_processing(PETImageOfVolume &current_image_estimate)
{


  
  if(get_parameters().inter_iteration_filter_interval>0 && subiteration_num%get_parameters().inter_iteration_filter_interval==0)
    {
      cerr<<endl<<"Applying inter-iteration filter"<<endl;
      get_parameters().inter_iteration_filter.apply(current_image_estimate);
    }


 

  cerr<< method_info()
      << " subiteration #"<<subiteration_num<<" completed"<<endl;
 
  cerr << "  min and max in image " << current_image_estimate.find_min() 
       << " " << current_image_estimate.find_max() << endl;

}

VectorWithOffset<int> IterativeReconstruction::randomly_permute_subset_order()
{

// KTxxx
//int temp_array[get_parameters().num_subsets];
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








