//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief List coordinates of maxima in the image

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Harry

  $Date$
  $Revision$

  \par Usage:
   \code
   find_fwhm_in_image filename [num_maxima [ mask_size_xy [mask_size z]] ]
   \endcode
   \param num_maxima defaults to 1
   \param mask_size_xy defaults to 1
   \param mask_size_z defaults to \a mask_size_xy
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;

#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cin;
using std::cerr;
using std::min;
using std::max;
using std::setw;
#endif

USING_NAMESPACE_STIR

/***********************************************************/
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType begin_iterator,
                       const RandomAccessIterType current_max_iterator,
                       const RandomAccessIterType end_iterator,
                       const float level)  ;  
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType begin_iterator,
                       const RandomAccessIterType end_iterator,
                       const float level)  ;                                                 
template <int num_dimensions, class elemT>
BasicCoordinate<num_dimensions,int>                        
maximum_location(const Array<num_dimensions,elemT>&);   
/*
template <int num_dimensions, class elemT>
Array<1,elemT>
extract_line(const Array<num_dimensions,elemT>& ,    
             const BasicCoordinate<num_dimensions,int>& , 
             int dimension);       
*/
template <class RandomAccessIterType>
float parabolic_3points_fit(const RandomAccessIterType begin_iter,const RandomAccessIterType end_iter);

/***********************************************************/

int main(int argc, char *argv[])
{ 

  if (argc< 2 || argc>5)
  {
    cerr << "Usage:" << argv[0] << " input_image [num_maxima [ mask_size_xy [mask_size z]] ]\n"
	       << "\tnum_maxima defaults to 1\n"
				 << "\tmask_size_xy defaults to 1\n"
				 << "\tmask_size_z defaults to mask_size_xy" <<endl;        
    return (EXIT_FAILURE);
  }

  const unsigned num_maxima = argc>=3 ? atoi(argv[2]) : 1;
  const int mask_size_xy = argc>=4 ? atoi(argv[3]) : 1;
  const int mask_size_z = argc>=5 ? atoi(argv[4]) : mask_size_xy;

  cerr << "Finding " << num_maxima << " maxima, each at least \n\t"
       << mask_size_xy << " pixels from each other in x,y direction, and\n\t"
       << mask_size_z << " pixels from each other in z direction.\n\n";

  shared_ptr< DiscretisedDensity<3,float> >  input_image_sptr = 
  DiscretisedDensity<3,float>::read_from_file(argv[1]);
  DiscretisedDensity<3,float>& input_image = *input_image_sptr;
  
/***********************************************************/
  
  vector<float> x_list, y_list, z_list;

//  float res_x_previous=0., res_y_previous=0., res_z_previous=0.;
  const DiscretisedDensityOnCartesianGrid <3,float>*  input_image_cartesian_ptr =
  dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*  > (input_image_sptr.get());

  const bool is_cartesian_grid = input_image_cartesian_ptr!=0;

  CartesianCoordinate3D<float> grid_spacing;
  if (is_cartesian_grid)
    grid_spacing = input_image_cartesian_ptr->get_grid_spacing();
  
  for (unsigned int maximum_num=0; maximum_num!=num_maxima; ++ maximum_num)
  {
    const float current_maximum = input_image.find_max();
    int max_k=0, max_j=0, max_i=0; // initialise to avoid compiler warnings
#if 1
    const BasicCoordinate<3,int> max_index =	maximum_location(input_image);
            max_k = max_index[1];
			  	  max_j = max_index[2];
		  		  max_i = max_index[3];
				  		
		        cout << maximum_num+1 << ". max: " << setw(6)
		             << current_maximum
			           << " at: "  << setw(3) << max_k
				         << " (Z) ," << setw(3) << max_j
      			     << " (Y) ," << setw(3) << max_i << " (X) \n" ;
      			     							
      	  	if (is_cartesian_grid)
		  			{
			  	    cout << "which is "
		    		    	 << setw(6) << max_k*grid_spacing[1]
							     << ',' << setw(6) << max_j*grid_spacing[2]
							     << ',' << setw(6) << max_i*grid_spacing[3]
								   << "  in mm relative to origin";
            }
				  	cout << " \n " ;

/*
template <int num_dimensions, class elemT>
Array<1,elemT>
new_x_list=extract_line(input_image,    
             max_index, 
             int dimension);       
*/
	    	
#else

    bool found=false;
 
    const int min_k_index = input_image.get_min_index();
	  const int max_k_index = input_image.get_max_index();
	  for ( int k = min_k_index; k<= max_k_index && !found; ++k)
	  {
	    const int min_j_index = input_image[k].get_min_index();
	    const int max_j_index = input_image[k].get_max_index();
	    for ( int j = min_j_index; j<= max_j_index && !found; ++j)
	    {
		   const int min_i_index = input_image[k][j].get_min_index();
       const int max_i_index = input_image[k][j].get_max_index();
			  for ( int i = min_i_index; i<= max_i_index && !found; ++i)
				{
          if (input_image[k][j][i] == current_maximum)
		      {
            max_k = k;
			  	  max_j = j;
		  		  max_i = i;
				  		
		        cout << maximum_num+1 << ". max: " << setw(6)
		             << current_maximum
			           << " at: "  << setw(3) << max_k
				         << " (Z) ," << setw(3) << max_j
      			     << " (Y) ," << setw(3) << max_i << " (X) \n" ;
      			     							
      	  	if (is_cartesian_grid)
		  			{
			  	    cout << "which is "
		    		    	 << setw(6) << max_k*grid_spacing[1]
							     << ',' << setw(6) << max_j*grid_spacing[2]
							     << ',' << setw(6) << max_i*grid_spacing[3]
								   << "  in mm relative to origin";
            }
				  	cout << " \n " ;
					  found = true;		
			    }
			  }
	    }
		}
		  if (!found)
		 	{
	  		 warning("Something strange going on: can't find maximum %g\n", current_maximum);
	    	 return EXIT_FAILURE;
			}
#endif		                            			
	                      
    {
      const int min_i_index = input_image[max_k][max_j].get_min_index();
 			const int max_i_index = input_image[max_k][max_j].get_max_index();
 				
      for (int counter=min_i_index ; counter<=max_i_index ; ++counter)
      x_list.push_back((input_image[max_k][max_j][counter]));
    }   
    {
      const int min_j_index = input_image[max_k].get_min_index();
			const int max_j_index = input_image[max_k].get_max_index();

      for (int counter=min_j_index ; counter<=max_j_index ; ++counter)
      y_list.push_back((input_image[max_k][counter][max_i]));
    }
    {
      const int min_k_index = input_image.get_min_index();
			const int max_k_index = input_image.get_max_index();
         	
      for (int counter=min_k_index ; counter<=max_k_index ; ++counter)
      z_list.push_back((input_image[counter][max_j][max_i]));          
    }               	
    	const float res_x = find_level_width(
      x_list.begin(),
  //   std::max_element(x_list.begin(),x_list.end()),
      x_list.end(),
     	parabolic_3points_fit(x_list.begin(),x_list.end())/2.) ;
     	
      const float res_y = find_level_width(
      y_list.begin(),
  //  std::max_element(y_list.begin(),y_list.end()),
      y_list.end(),
     	parabolic_3points_fit(y_list.begin(),y_list.end())/2.) ;
      
      const float res_z = find_level_width(
      z_list.begin(),
 //   std::max_element(z_list.begin(),z_list.end()),
      z_list.end(),
      parabolic_3points_fit(z_list.begin(),z_list.end())/2.) ;
                           
/*      // POSSIBLE CASES //
        if(res_x==-10.||res_y==-10.||res_z==-10.)
        cout << "I cannot find the resolution because the half maximum is outside the FoV \n\
               Try another image or not more than " << maximum_num << " maxima " ;     
        else if(res_x==0.||res_y==.0||res_z==0.)
        cout << "\n You should probably use smaller mask size! \n Try for z mask size:" 
             <<  2.*res_z_previous
             << "and for xy mask size:" << 2.*max(res_x_previous,res_y_previous)
             << ".\n";    
        else 
        {
          res_x_previous=res_x;
          res_y_previous=res_y;
          res_z_previous=res_z;
        }
*/    
      if (res_x==0. || res_y==0. || res_z==0.)
      cout <<"\n I cannot find the resolution in this dimension \n\
                 because this level of maximum is outside the FoV \n\
                 or because the distance of two maxima are near \n\
                 comparing to the used mask size.\n"  ;
             
     
    	cout << "  \n The resolution in z axis might be "
       	   << res_z*grid_spacing[1]
         	 << ", \n The resolution in y axis might be "
       	 	 << res_y*grid_spacing[2]
         	 << ", \n The resolution in x axis might be "
           << res_x*grid_spacing[3]
           << ", in mm relative to origin. \n \n"; 
  
     x_list.clear();
     y_list.clear();
     z_list.clear();  

     if (maximum_num+1!=num_maxima)
		 {
        const int min_k_index = input_image.get_min_index();
	      const int max_k_index = input_image.get_max_index();
	  	  // now mask it out for next run
	      for ( int k = max(max_k-mask_size_z,min_k_index); k<= min(max_k+mask_size_z,max_k_index); ++k)
	      {
	        const int min_j_index = input_image[k].get_min_index();
	        const int max_j_index = input_image[k].get_max_index();
	        for ( int j = max(max_j-mask_size_xy,min_j_index); j<= min(max_j+mask_size_xy,max_j_index); ++j)
	        {
				    const int min_i_index = input_image[k][j].get_min_index();
				    const int max_i_index = input_image[k][j].get_max_index();
				    for ( int i = max(max_i-mask_size_xy,min_i_index); i<= min(max_i+mask_size_xy,max_i_index); ++i)
		  	    input_image[k][j][i] = 0.;
	        }
	      } 
      }
    }                         
  return EXIT_SUCCESS;
}                                          
                
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType begin_iterator,
                       const RandomAccessIterType current_max_iterator,
                       const RandomAccessIterType end_iterator,
                       const float level)
/* This function takes as input the Begin, the End and the Max_Element Iterators of a sequence of numbers 
   (e.g. vector) and the level of the maximum you want to have (e.g. maximum/2 or maximum/10). 
   It gives as output resolution in pixel size.   
*/
{
  RandomAccessIterType current_iter ;
  const float maximum = *current_max_iterator;
  int max_position=0;
 
  current_iter=begin_iterator; 
  while (current_iter!=end_iterator)
  {
    if (maximum==*current_iter)  max_position=current_iter-begin_iterator+1 ;
    ++current_iter;  
  }
  current_iter = current_max_iterator;
  while(current_iter!= end_iterator && *current_iter > level)   ++current_iter;
  if (current_iter==end_iterator)  return -0.;       //  avoid getting out of the borders               
  if (*current_iter==0.) return -100000. ;                //  in case we are in the masked area

  float right_level_max = (*current_iter - level)/(*current_iter-*(current_iter-1));
  right_level_max = float(current_iter-(begin_iterator+max_position)) - right_level_max ;
  
  current_iter = current_max_iterator;
  while(current_iter!=begin_iterator && *current_iter > level) --current_iter;
  if (current_iter == begin_iterator)  return -0.;   //  avoid getting out of the borders
  if (*current_iter==0.) return -100000. ;                //  in case we are in the masked area
    
  float left_level_max = (*current_iter - level)/(*current_iter-*(current_iter+1));
  left_level_max += float(current_iter-(begin_iterator+max_position));

  return right_level_max - left_level_max;   
} 
                     
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType begin_iterator,
                       const RandomAccessIterType end_iterator,
                       const float level)
/* This function takes as input the Begin, the End and the Max_Element Iterators of a sequence of numbers 
   (e.g. vector) and the level of the maximum you want to have (e.g. maximum/2 or maximum/10). 
   It gives as output resolution in pixel size.   
*/
{                  
  RandomAccessIterType current_iter = std::max_element(begin_iterator,end_iterator);
  const float maximum = *std::max_element(begin_iterator,end_iterator);
  int max_position=0;
 
  current_iter=begin_iterator; 
  while (current_iter!=end_iterator)
  {
    if (maximum==*current_iter)  max_position=current_iter-begin_iterator+1 ;
    ++current_iter;  
  }            
/* 
Finding the right and left points with a linear interpolation of the two points values 
who are next to the level, of each side. 
*/                  
//  finding the right point 
  current_iter = std::max_element(begin_iterator,end_iterator);
  while(current_iter!= end_iterator && *current_iter > level)   ++current_iter;
  if (current_iter==end_iterator)  return -0.;       //  avoid getting out of the borders               
  if (*current_iter==0.) return -10000. ;                //  in case we are in the masked area                                                 
  float right_level_max = (*current_iter - level)/(*current_iter-*(current_iter-1));
  right_level_max = float(current_iter-(begin_iterator+max_position)) - right_level_max ;
//  finding the left point 
  current_iter = std::max_element(begin_iterator,end_iterator);
  while(*current_iter > level && current_iter!=begin_iterator) --current_iter;
  if (current_iter == begin_iterator)   return -0.; //  avoid getting out of the borders
  if (*current_iter==0.) return -100000. ;                //  in case we are in the masked area                  
  float left_level_max = (*current_iter - level)/(*current_iter-*(current_iter+1));
  left_level_max += float(current_iter-(begin_iterator+max_position));

   return right_level_max - left_level_max;  
}      
                            
template <int num_dimensions, class elemT>
BasicCoordinate<num_dimensions,int> 
maximum_location(const Array<num_dimensions,elemT>& input_array)
/* This function finds the maximum of the Input_Array and returns its 
   location as a vector in BasicCoordinate Field
*/
{
  const float current_maximum = input_array.find_max();
  BasicCoordinate<3,int> max_location; // initialise to avoid compiler warnings
  bool found=false;

  const int min_k_index = input_array.get_min_index();
	const int max_k_index = input_array.get_max_index();
	for ( int k = min_k_index; k<= max_k_index && !found; ++k)
	{
	  const int min_j_index = input_array[k].get_min_index();
	  const int max_j_index = input_array[k].get_max_index();
	  for ( int j = min_j_index; j<= max_j_index && !found; ++j)
	  {
		 const int min_i_index = input_array[k][j].get_min_index();
     const int max_i_index = input_array[k][j].get_max_index();
		 for ( int i = min_i_index; i<= max_i_index && !found; ++i)
		 {
       if (input_array[k][j][i] == current_maximum)
		   {
         max_location[1] = k;
		  	 max_location[2] = j;
		  	 max_location[3] = i;
		    }
		  }
		}
	}
  found = true;		
  return(max_location);	
} 

 /*
template <int num_dimensions, class elemT>
Array<1,elemT>
extract_line(const Array<num_dimensions,elemT>& ,    
             const BasicCoordinate<num_dimensions,int>& , 
             int dimension);       

{
    



}
      
*/




 template <class RandomAccessIterType>
float parabolic_3points_fit(const RandomAccessIterType begin_iter,const RandomAccessIterType end_iter)  
/* 
   This function calculates the maximum point (x0,y0) of a parabola that passes through 3 points.
   As input it takes the Begin and End Iterators of a sequence of numbers (e.g. vector). 
   The three points are the maximum (x2,y2) of this sequence and the two neihbour points 
   (x1,y1) and (x3,y3). It returns the maximum point value y0. 
*/           
{                     
  const RandomAccessIterType max_iter = std::max_element(begin_iter,end_iter);
  if (max_iter==end_iter-1 || max_iter==begin_iter) 
  return 10000.;           // Maximum is at the borders 
  float real_max_value;
  {                    
    const float y1 = *(max_iter-1);
    const float y2 = *max_iter; 
    const float y3 = *(max_iter+1);
    const float x1 = -1.;
    const float x2 =  0.; // Giving the axis zero point at x2.
    const float x3 =  1.;
    const float a1 = (x1-x2)*(x1-x3);
    const float a2 = (x2-x1)*(x2-x3);
    const float a3 = (x3-x2)*(x3-x1);
/* 
Now find parameters for parabola that fits these 3 points.
Using Langrange's classical formula, equation will be:
y(x)=((x - x2)*(x - x3)*y1/a1)+ ((x - x1)*(x - x3)*y2/a2) + ((x - x1)*(x - x2)*y3/a3)
y'(x0) = 0 =>  x0 = 0.5*(x1*a1*(y2*a3+y3*a2)+x2*a2*(y1*a3+y3*a1)+x3*a3*(y1*a2+y2*a1))/(y1*a2*a3+y2*a1*a3+y3*a1*a2)    
*/                 
    float x0 = 0.5*(x1*a1*(y2*a3+y3*a2)+x2*a2*(y1*a3+y3*a1)+x3*a3*(y1*a2+y2*a1))/(y1*a2*a3+y2*a1*a3+y3*a1*a2) ; 
    real_max_value = ((x0 - x2)*(x0 - x3)*y1/a1)+ ((x0 - x1)*(x0 - x3)*y2/a2) + ((x0 - x1)*(x0 - x2)*y3/a3) ;
  } 

   return real_max_value ;
}                       
