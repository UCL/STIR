#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"

#include <iostream>
#include <fstream>


#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::ifstream;
using std::endl;
using std::find;
using std::cerr;
#endif

USING_NAMESPACE_STIR

bool is_inside_range (int lower, int upper, int num)
{
  if ( num >= lower && num <= upper)
    return true;
  else
    return false;
}



 
int main(int argc, char *argv[])
{ 
  
  if (argc!= 2)
  {
    cerr << "Usage:" << argv[0] << " input image"<<endl<<endl;    
    return (EXIT_FAILURE);
  }

  ifstream in (argv[1]);  

 shared_ptr< DiscretisedDensity<3,float> >  input_image = 
   DiscretisedDensity<3,float>::read_from_file(argv[1]);

 VoxelsOnCartesianGrid <float>*  input_image_vox = 0;
   input_image_vox = 
     dynamic_cast< VoxelsOnCartesianGrid<float>*  > (input_image.get());

  float first_maximum = input_image_vox->find_max();
  int max_k, max_j,max_i;
  max_k = max_j = max_i =0;
  for ( int k = input_image_vox->get_min_index(); k<= input_image_vox->get_max_index();k++)
    for ( int j = (*input_image_vox)[k].get_min_index(); j<= (*input_image_vox)[k].get_max_index();j++)
      for ( int i = (*input_image_vox)[k][j].get_min_index(); i<= (*input_image_vox)[k][j].get_max_index();i++)
      {
	if ( (*input_image_vox)[k][j][i] == first_maximum)
	{
	  max_k = k;
	  max_j = j;
	  max_i = i;
	}
	else
	  continue;
      }
#if 1
      float second_maximum =-1.0;
      int max_k_s;
      int max_j_s;
      int max_i_s;

    // now find the other maximum 
  for ( int k = input_image_vox->get_min_index(); k<= input_image_vox->get_max_index();k++)
    for ( int j = (*input_image_vox)[k].get_min_index(); j<= (*input_image_vox)[k].get_max_index();j++)
      for ( int i = (*input_image_vox)[k][j].get_min_index(); i<= (*input_image_vox)[k][j].get_max_index();i++)
         {
         if (is_inside_range (max_k-7, max_k+7, k) && is_inside_range (max_j-7, max_j+7, j)
	   && is_inside_range (max_i-7, max_i+7, i))
	   continue;
	 else
	   if ((*input_image_vox)[k][j][i] > second_maximum)
	   {
	   second_maximum = (*input_image_vox)[k][j][i];
	   max_k_s=k;
	   max_j_s=j;
	   max_i_s=i;
	   }
	   else
	     continue;
      }
#endif
      cerr << "The  first maximum is:   " <<  first_maximum << "    " << " at position:  " << max_k <<"   " << max_j <<"   " << max_i << "   ";
      cerr << "The  second maximum is:   " <<  second_maximum << "    " << " at position:  " << max_k_s <<"   " << max_j_s <<"   " << max_i_s << "   ";




    return EXIT_SUCCESS;
}
  