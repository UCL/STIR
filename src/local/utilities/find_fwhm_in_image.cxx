//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief List coordinates of maxima in the image

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Charalampos Tsoumpas

  $Date$
  $Revision$

  \par Usage:
   \code
   find_fwhm_in_image filename [num_maxima] [level] [dimension]
   \endcode
   \param num_maxima defaults to 1
   \if num_maxima more than 10, the [num_maxima] slices are printed into a file
   \param level defaults to 2   
   \param dimension can take the values of the axis at which the line source is along to.
   \along z axis : 1 (default)
   \along y axis : 2
   \along x axis : 3          
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "local/stir/find_fwhm_in_image.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <list>
#include <algorithm>  
#include <string>
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cin;
using std::cerr;
using std::min;
using std::max;
using std::setw;
#endif

/***********************************************************/     
int main(int argc, char *argv[])                                  
{         
  USING_NAMESPACE_STIR
  using namespace std;                                                                
  if (argc< 2 || argc>5)
  {
    cerr << "Usage:" << argv[0] << " input_image [num_maxima] [level] [dimension]\n"
	       << "\tnum_maxima defaults to 1\n" 
         << "\tlevel of maximum defaults at half maximum: 2\n"  
         << "\tfor point sources print less than 11 num_maxima\n"
         << "\tfor line sources print more than 11 num_maxima\n"
         << "\tline source along z-axis dimension=1 (default)\n"  
         << "\tline source along y-axis dimension=2 \n"
         << "\tline source along x-axis dimension=3 \n" ;         
    return EXIT_FAILURE;            
  } 
  const unsigned short num_maxima = argc>=3 ? atoi(argv[2]) : 1 ;
  const float level = argc>=4 ? atoi(argv[3]) : 2 ;  
  const int dimension = argc>=5 ? atoi(argv[4]) : 1 ;  
  cerr << "Finding " << num_maxima << " maxima\n" ;    
  shared_ptr< DiscretisedDensity<3,float> >  input_image_sptr = 
  DiscretisedDensity<3,float>::read_from_file(argv[1]);
  DiscretisedDensity<3,float>& input_image = *input_image_sptr;      
  const DiscretisedDensityOnCartesianGrid <3,float>*  input_image_cartesian_ptr =
  dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*> (input_image_sptr.get());  
  const bool is_cartesian_grid = input_image_cartesian_ptr!=0; 
  CartesianCoordinate3D<float> grid_spacing;     
  if (is_cartesian_grid)   																					
  {
    grid_spacing = input_image_cartesian_ptr->get_grid_spacing(); 
    list<ResolutionIndex<3,float> > list_res_index = 
    find_fwhm_in_image(input_image,num_maxima,level,dimension);    
    list<ResolutionIndex<3,float> >:: iterator current_iter=list_res_index.begin();
  
    if (num_maxima>10)    
    {
      string output_string;
      string input_string(argv[1]);
      string slices_string(argv[2]);             
      string:: iterator string_iter;
      for(string_iter=input_string.begin(); 
          string_iter!=input_string.end() && *string_iter!='.' ;
          ++string_iter)  
      output_string.push_back(*string_iter);     

      if (argc>=4)
      {
         string level_string(argv[3]);
         output_string +=  '_' + slices_string + "_slices_FW" + level_string + 'M' ;
      }
      else
      output_string +=  '_' + slices_string + "_slices_FWHM" ;  
      
      ofstream out(output_string.c_str()); //output file //
      if(!out)
      {
        cout << "Cannot open text file.\n" ;
        return EXIT_FAILURE;
      }       
      out << "Slice\t Z\tY\t X\tResZ(mm) ResY(mm) ResX(mm) Value\n"; //t Fitted Maximum Values \n";
      for (short counter=0 ; counter!=num_maxima ; ++counter)
      { 
        out << setw(3) << counter+1 << "\t" 
            << setw(3) << current_iter->voxel_location[1] << "\t"
		        << setw(3) << current_iter->voxel_location[2] << "\t"
            << setw(3) << current_iter->voxel_location[3] << "\t" 
  	        << setw(6) << current_iter->resolution[1]*grid_spacing[1]<< "\t"
        	  << setw(6) << current_iter->resolution[2]*grid_spacing[2]<< "\t"
       	    << setw(6) << current_iter->resolution[3]*grid_spacing[3]<< "\t"
            << setw(9) << current_iter->voxel_value << "\n" ;
        //   << setw(9) << current_iter->real_maximum_value[2] << "\t"
        //   << setw(9) << current_iter->real_maximum_value[3] << "\n";      
        ++current_iter;         
      }
      out.close();  
      return EXIT_SUCCESS;
    } 
    else 
    for (short counter=0 ; counter!=num_maxima ; ++counter)
    {
      cout << counter+1 << ". max: " << setw(6)	<<  current_iter->voxel_value	  
	         << " at: "  << setw(3) << current_iter->voxel_location[1]
		       << " (Z) ," << setw(3) << current_iter->voxel_location[2]
           << " (Y) ," << setw(3) << current_iter->voxel_location[3] << " (X) \n" ;
		  cout << "which is "
		       << ':' << setw(6) << (current_iter->voxel_location[1])*grid_spacing[1]
			     << ',' << setw(6) << (current_iter->voxel_location[2])*grid_spacing[2]
				   << ',' << setw(6) << (current_iter->voxel_location[3])*grid_spacing[3]
					 << "  in mm relative to origin";
  	  cout << "  \n The resolution in z axis is "
           << setw(6) << (current_iter->resolution[1])*grid_spacing[1]
       	   << ", \n The resolution in y axis is "
     	     << setw(6) << (current_iter->resolution[2])*grid_spacing[2]
           << ", \n The resolution in x axis is "
     	     << setw(6) << (current_iter->resolution[3])*grid_spacing[3]
           << ", in mm relative to origin. \n \n";  
      ++current_iter;         
    }        
  }      
  else
  { 
    cout << "THERE IS NOT A CARTESIAN GRID\n";
    return EXIT_FAILURE;  
  }
}                 
