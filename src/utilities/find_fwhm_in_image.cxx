//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief List of FWHM and its location in the image
  

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$

  \par Usage:
   \code
   find_fwhm_in_image filename [num_maxima][level][dimension][nema]
   \endcode
   \param num_maxima defaults to 1
   \param level defaults to 2 (half maximum)  
   \param dimension:
   \for point sources (default) set to 0
   \for line source along z, y, x -axis,
   \set to: 1, 2, 3 respectively
   \param NEMA defaults to 1 
   
  
  If you have point sources, it prints the value of the [num_maxima] maximum point source with its location 
  and the resolution at the three dimensions, sorting from the maximum to the minimum. If you have a line 
  source, a text file is returned that it contains the maximum value at its one slice, which is sorted from
  the mimimum to maximum slice index, at the wanted direction, with its location and the resolution at the 
  three dimensions. The resolution at the axis of the line is set to be 0. If given as [num_maxima] less 
  than the total slices it returns the results for some slices by sampling with the same step the total 
  slices of the wanted dimension. The [nema] parameter enables the oportunity of using the NEMA Standards 
  Publication NU 2-2001. If it is set to 0 the function estimates the FWHM using 3D interpolation, for 
  closer approximation. 
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/find_fwhm_in_image.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <list>
#include <algorithm>  
#include <string>
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cerr;
using std::setw;
#endif

/***********************************************************/     
int main(int argc, char *argv[])                                  
{         
  USING_NAMESPACE_STIR
  using namespace std;                                                                
  if (argc< 2 || argc>6)
  {
    cerr << "Usage:" << argv[0] << " input_image [num_maxima][level][dimension][nema]\n"
	       << "\tnum_maxima defaults to 1\n" 
         << "\tlevel of maximum defaults at half maximum: 2\n"  
         << "\tfor point sources dimension set to 0 (default)\n"
         << "\tfor line source along z, y, x -axis dimension set to 1, 2, 3 respectively:\n"
         << "\tnema defaults to 1, NEMA Standards NU 2-2001 is enabled\n"
         << "returns a file containing the resolutions\n\n";

    return EXIT_FAILURE;            
  } 
  const unsigned short num_maxima = argc>=3 ? atoi(argv[2]) : 1 ;
  const float level = argc>=4 ? static_cast<float>(atof(argv[3])) : 2 ;  
  const int dimension = argc>=5 ? atoi(argv[4]) : 0 ; 
  const bool nema = argc>=6 ? atoi(argv[5]) : 1 ; 
  cerr << "Finding " << num_maxima << " maxima\n" ;    
  shared_ptr< DiscretisedDensity<3,float> >  input_image_sptr= 
  DiscretisedDensity<3,float>::read_from_file(argv[1]);
  DiscretisedDensity<3,float>& input_image = *input_image_sptr;  
  list<ResolutionIndex<3,float> > list_res_index = 
  find_fwhm_in_image(input_image,num_maxima,level,dimension,nema);    
  list<ResolutionIndex<3,float> >:: iterator current_iter=list_res_index.begin();    
  if (dimension!=0)    
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
    out << "Slice\t Z\tY\t X\tResZ(mm) ResY(mm) ResX(mm) Value\n"; 
    for (short counter=0 ; counter!=num_maxima ; ++counter)
    { 
      out << setw(3) << counter+1 << "\t" 
          << setw(3) << current_iter->voxel_location[1] << "\t"
          << setw(3) << current_iter->voxel_location[2] << "\t"
          << setw(3) << current_iter->voxel_location[3] << "\t" 
 	        << setw(6) << current_iter->resolution[1]<< "\t"
       	  << setw(6) << current_iter->resolution[2]<< "\t"
     	    << setw(6) << current_iter->resolution[3]<< "\t"
          << setw(9) << current_iter->voxel_value << "\n" ;
      ++current_iter;         
    }
    out.close();       
  } 
  else 
    for (short counter=0 ; counter!=num_maxima ; ++counter)
    {
      cout << counter+1 << ". max: " << setw(6)	<<  current_iter->voxel_value	  
	         << " at: "  << setw(3) << current_iter->voxel_location[1]
		       << " (Z) ," << setw(3) << current_iter->voxel_location[2]
           << " (Y) ," << setw(3) << current_iter->voxel_location[3] << " (X) \n" ;
  	  cout << "  \n The resolution in z axis is "
           << setw(6) << (current_iter->resolution[1])
       	   << ", \n The resolution in y axis is "
     	     << setw(6) << (current_iter->resolution[2])
           << ", \n The resolution in x axis is "
     	     << setw(6) << (current_iter->resolution[3])
           << ", in mm relative to origin. \n \n";  
      ++current_iter;         
    }                
  return EXIT_SUCCESS;
}                 
