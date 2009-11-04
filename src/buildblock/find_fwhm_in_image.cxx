//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
  \ingroup buildblock
  \brief Implementations of functions defined in find_fwhm_in_image.h

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/find_fwhm_in_image.h"
#include "stir/index_at_maximum.h"
#include "stir/round.h"
#include <algorithm>  
#include <list>

using namespace std;

START_NAMESPACE_STIR

/*
   2 functions that calculate the maximum point (x0,y0) of a parabola that passes through 3 points.

   As input it takes the Begin and End Iterators of a sequence of numbers (e.g. vector). 
   The three points are the maximum (x2,y2) of this sequence and the two neighbour points 
   (x1,y1) and (x3,y3).  

  The first function returns the maximum point value y0, the secodn returns x0
*/           

template <class RandomAccessIterType>
static float parabolic_3points_fit(const RandomAccessIterType& begin_iter,
                                   const RandomAccessIterType& end_iter);

template <class RandomAccessIterType>
static float parabolic_3points_fit_x0(const RandomAccessIterType& begin_iter,
                                      const RandomAccessIterType& end_iter) ; 
   
template <class elemT>
static float find_NEMA_level(const Array<1,elemT>& column, const float level)
{   
  const float real_maximum_value = 
        parabolic_3points_fit(column.begin(),column.end());
  return find_level_width(column.begin(),column.end(),real_maximum_value/level) ;      
}

template <int num_dimensions, class elemT>
std::list<ResolutionIndex<num_dimensions,float> > 
find_fwhm_in_image(DiscretisedDensity<num_dimensions,elemT> & input_image, 
                   const unsigned int num_maxima, 
                   const float level, 
                   const int dimension,
                   const bool nema)
{                      
  ResolutionIndex<3,float> res_index;
  std::list<ResolutionIndex<3,float> > list_res_index;    
 
  const DiscretisedDensityOnCartesianGrid <3,float>* input_image_cartesian_ptr =
    dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*> (&input_image);
  if (input_image_cartesian_ptr == 0)
    {
      error("find_fwhm_in_image currently only works with DiscretisedDensityOnCartesianGrid images");
    }
  BasicCoordinate<3,int> min_index, max_index;
  if (!input_image.get_regular_range(min_index, max_index))
    error("find_fwhm_in_image works only on regular ranges\n");
  
  CartesianCoordinate3D<float> 
    grid_spacing=input_image_cartesian_ptr->get_grid_spacing();
     
  for (unsigned int maximum_num=0; maximum_num!=num_maxima; ++ maximum_num)
    { 
      float current_maximum ;  
      BasicCoordinate<3,int> max_location ;
      Coordinate3D<bool> do_direction(true,true,true);
      if(dimension!=0)//Divide into [maximum_num] slices and returns a max per slice
        { 
          const float step=float( max_index[dimension]- min_index[dimension])/float(num_maxima-1);
          const int slice= round(min_index[dimension] + maximum_num*step);
          max_location =        maximum_location_per_slice(input_image,slice,dimension);
          current_maximum = input_image[max_location];
          do_direction[dimension]=false;
        }  
      else     // Searches through out the image to find the point sources  
        {  
          max_location =        indices_at_maximum(input_image);  
          current_maximum = input_image[max_location];
        }                        
      res_index.voxel_location = max_location;     
      res_index.voxel_value = current_maximum ;
      for(int i=1;i<=3;++i)
        res_index.resolution[i] = grid_spacing[i]*
          (do_direction[i] ? (nema?find_NEMA_level(extract_line(input_image,max_location,
                                                                i), level) : 
                              find_NEMA_level(interpolated_line(input_image,max_location,
                                                                do_direction,
                                                                i), level)): 0); 
      list_res_index.push_back(res_index);
      if (maximum_num+1!= num_maxima && dimension==0)
        assign_to_subregion(input_image,max_location,round(res_index.resolution/grid_spacing/level*2), input_image.find_min());
    }
  return list_res_index ;
}   
              
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType& begin_iterator,
                       const RandomAccessIterType& current_max_iterator,
                       const RandomAccessIterType& end_iterator,
                       const float level_height)
{ 
  const int max_position = current_max_iterator - begin_iterator + 1;
  RandomAccessIterType current_iter = current_max_iterator;
  while(current_iter!= end_iterator && *current_iter > level_height)   ++current_iter;
  if (current_iter==end_iterator)  
    {
      warning("find_level_width: level extends beyond border."
              "Cannot find the real level-width of this point source!");
      // go 1 back to remain inside the range
      --current_iter;
    }

  // do linear interpolation to find position of level_height  
  float right_level_max = (*current_iter - level_height)/(*current_iter-*(current_iter-1));
  right_level_max = float(current_iter-(begin_iterator+max_position)) - right_level_max ;
  
  current_iter = current_max_iterator;
  while(current_iter!=begin_iterator && *current_iter > level_height) --current_iter;
  if (current_iter == begin_iterator && *current_iter > level_height) 
    {
      warning("find_level_width: level extends beyond border."
              "Cannot find the real level-width of this point source!");
    }
        
  float left_level_max = (*current_iter - level_height)/(*current_iter-*(current_iter+1));
  left_level_max += float(current_iter-(begin_iterator+max_position));

  return right_level_max - left_level_max;   
} 
                     
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType& begin_iterator,
                       const RandomAccessIterType& end_iterator,
                       const float level_height)
{
  return find_level_width(begin_iterator, 
                          std::max_element(begin_iterator,end_iterator),
                          end_iterator,
                          level_height); 
}

template<int num_dimensions,class elemT>                         
BasicCoordinate<num_dimensions,int>
maximum_location_per_slice(const Array<num_dimensions,elemT>& input_array,
                           const int slice, const int dimension)
{
  BasicCoordinate<3,int> min_index, max_index;
  if (!input_array.get_regular_range(min_index, max_index))
    error("maximum_location_per_slice works only on regular ranges\n");
  BasicCoordinate<3,int> min_slice_index=min_index, 
    max_slice_index=max_index;
  min_slice_index[dimension] = slice ;
  max_slice_index[dimension] = slice ;
  const IndexRange<3> slice_range(min_slice_index,max_slice_index);
  Array<3,elemT> slice_array(slice_range); 
  BasicCoordinate<3,int> counter; 
  for (counter[1]=min_slice_index[1]; counter[1]<= max_slice_index[1] ; ++counter[1])
    for (counter[2]=min_slice_index[2]; counter[2]<= max_slice_index[2] ; ++counter[2])    
      for (counter[3]=min_slice_index[3]; counter[3]<= max_slice_index[3] ; ++counter[3])    
        slice_array[counter] = input_array[counter];  
  return indices_at_maximum(slice_array);       
}       
                       
template <int num_dimensions, class elemT>
Array<1,elemT>
extract_line(const Array<num_dimensions,elemT>& input_array,    
             BasicCoordinate<num_dimensions,int> voxel_location, 
             const int dimension)   
{ 
  BasicCoordinate<3,int> min_index,max_index;
  int &counter = voxel_location[dimension];  
  min_index[1] = input_array.get_min_index();
  max_index[1] = input_array.get_max_index();
  min_index[2] = input_array[voxel_location[1]].get_min_index();
  max_index[2] = input_array[voxel_location[1]].get_max_index();
  min_index[3] = input_array[voxel_location[1]][voxel_location[2]].get_min_index();
  max_index[3] = input_array[voxel_location[1]][voxel_location[2]].get_max_index();       
  Array<1,elemT> line(min_index[dimension],max_index[dimension]);    
  for (counter=min_index[dimension]; counter<= max_index[dimension] ; ++counter)
    line[counter]= input_array[voxel_location[1]][voxel_location[2]][voxel_location[3]];                    
  return line ;
}  

template <int num_dimensions, class elemT>
Array<1,elemT>
interpolated_line(const Array<num_dimensions,elemT>& input_array,    
                  const BasicCoordinate<num_dimensions,int>& max_location, 
                  const BasicCoordinate<num_dimensions,bool>& do_direction,
                  const int dimension)  
/*
  This function interpolates a column from the given array, that includes the particular voxel and returns 
  a column in Array<1,elemT> type, at the wanted dimension (z=1, y=2, x=3).
//Assume same index at each direction //
*/        
{
  BasicCoordinate<3,int> min_index, max_index;
  if (!input_array.get_regular_range(min_index, max_index))
    error("interpolated_line works only on regular ranges\n");
  Array<1,elemT> line(min_index[dimension],max_index[dimension]);
  {
    Array<1,elemT> line_z(min_index[1],max_index[1]),
      line_y(min_index[2],max_index[2]),
      line_x(min_index[3],max_index[3]),
      line_000(min_index[dimension],max_index[dimension]),
      line_001(min_index[dimension],max_index[dimension]),
      line_010(min_index[dimension],max_index[dimension]),
      line_100(min_index[dimension],max_index[dimension]),
      line_011(min_index[dimension],max_index[dimension]),
      line_101(min_index[dimension],max_index[dimension]),
      line_110(min_index[dimension],max_index[dimension]),
      line_111(min_index[dimension],max_index[dimension]);
    line_z = extract_line(input_array,max_location,1);    
    line_y = extract_line(input_array,max_location,2);    
    line_x = extract_line(input_array,max_location,3);    
    float z0=do_direction[1] ? parabolic_3points_fit_x0(line_z.begin(),line_z.end()) : 0;       
    float y0=do_direction[2] ? parabolic_3points_fit_x0(line_y.begin(),line_y.end()) : 0;       
    float x0=do_direction[3] ? parabolic_3points_fit_x0(line_x.begin(),line_x.end()) : 0;       

    BasicCoordinate<num_dimensions,int> location_000,location_001,location_010,location_100,
      location_011,location_101,location_110,location_111;
    location_000 = max_location;                
                
    location_001[1] = max_location[1]; 
    location_001[2] = max_location[2];  
    location_001[3] = x0>0 ? max_location[3]+1 : (x0<0 ? max_location[3]-1 : max_location[3]) ;
                
    location_010[1] = max_location[1]; 
    location_010[2] = y0>0 ? max_location[2]+1 : (y0<0 ? max_location[2]-1 : max_location[2]) ;
    location_010[3] = max_location[3];
                

    location_100[1] = z0>0 ? max_location[1]+1 : (z0<0 ? max_location[1]-1 : max_location[1]) ;
    location_100[2] = max_location[2];  
    location_100[3] = max_location[3];
                

    location_011[1] = max_location[1]; 
    location_011[2] = y0>0 ? max_location[2]+1 : (y0<0 ? max_location[2]-1 : max_location[2]) ;
    location_011[3] = x0>0 ? max_location[3]+1 : (x0<0 ? max_location[3]-1 : max_location[3]) ;
                
    location_101[1] = z0>0 ? max_location[1]+1 : (z0<0 ? max_location[1]-1 : max_location[1]) ;
    location_101[2] = max_location[2];  
    location_101[3] = x0>0 ? max_location[3]+1 : (x0<0 ? max_location[3]-1 : max_location[3]) ;
                
    location_110[1] = z0>0 ? max_location[1]+1 : (z0<0 ? max_location[1]-1 : max_location[1]) ;
    location_110[2] = y0>0 ? max_location[2]+1 : (y0<0 ? max_location[2]-1 : max_location[2]) ;                                 
    location_110[3] = max_location[3] ;

    location_111[1] = z0>0 ? max_location[1]+1 : (z0<0 ? max_location[1]-1 : max_location[1]) ;
    location_111[2] = y0>0 ? max_location[2]+1 : (y0<0 ? max_location[2]-1 : max_location[2]) ;
    location_111[3] = x0>0 ? max_location[3]+1 : (x0<0 ? max_location[3]-1 : max_location[3]) ;

    line_000 = extract_line(input_array,location_000,dimension);                        
    line_001 = extract_line(input_array,location_001,dimension);
    line_010 = extract_line(input_array,location_010,dimension);                        
    line_100 = extract_line(input_array,location_100,dimension);
    line_011 = extract_line(input_array,location_011,dimension);
    line_101 = extract_line(input_array,location_101,dimension);

    line_110 = extract_line(input_array,location_110,dimension);
    line_111 = extract_line(input_array,location_111,dimension);
    line =  line_000*(1-abs(z0))*(1-abs(y0))*(1-abs(x0)) ;
    line+=  line_001*(1-abs(z0))*(1-abs(y0))*abs(x0)  ;
    line+=      line_010*(1-abs(z0))*abs(y0)*(1-abs(x0)) ;
    line+=      line_100*abs(z0)*(1-abs(y0))*(1-abs(x0)) ;
    line+=      line_011*(1-abs(z0))*abs(y0)*abs(x0) ;
    line+=      line_101*abs(z0)*(1-abs(y0))*abs(x0) ;
    line+=      line_110*abs(z0)*abs(y0)*(1-abs(x0)) ;
    line+=      line_111*abs(z0)*abs(y0)*abs(x0) ;
  }
  return line ;
}          
template <int num_dimensions, class elemT>   
void 
assign_to_subregion(Array<num_dimensions,elemT>& input_array, 
                    const BasicCoordinate<num_dimensions,int>& mask_location,
                    const BasicCoordinate<num_dimensions,int>& half_mask_size,
                    const elemT& value)
{
  const int min_k_index = input_array.get_min_index();
  const int max_k_index = input_array.get_max_index();
  for ( int k = max(mask_location[1]-half_mask_size[1],min_k_index); k<= min(mask_location[1]+half_mask_size[1],max_k_index); ++k)
    {
      const int min_j_index = input_array[k].get_min_index();
      const int max_j_index = input_array[k].get_max_index();
      for ( int j = max(mask_location[2]-half_mask_size[2],min_j_index); j<= min(mask_location[2]+half_mask_size[2],max_j_index); ++j)
        {
          const int min_i_index = input_array[k][j].get_min_index();
          const int max_i_index = input_array[k][j].get_max_index();
          for ( int i = max(mask_location[3]-half_mask_size[3],min_i_index); i<= min(mask_location[3]+half_mask_size[3],max_i_index); ++i)
            input_array[k][j][i] = value;
        }
    } 
}        
    
template <class RandomAccessIterType>
float parabolic_3points_fit(const RandomAccessIterType& begin_iter,
                                   const RandomAccessIterType& end_iter)    
{ 
  float real_max_value;                    
  const RandomAccessIterType max_iter = std::max_element(begin_iter,end_iter);
  if (max_iter==end_iter-1 || max_iter==begin_iter)   
    return *max_iter;           // In case maximum is at the borders of the image     
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
    const float x0 = 0.5*(x1*a1*(y2*a3+y3*a2)+x2*a2*(y1*a3+y3*a1)+x3*a3*(y1*a2+y2*a1))
      /(y1*a2*a3+y2*a1*a3+y3*a1*a2) ; 
    real_max_value = ((x0 - x2)*(x0 - x3)*y1/a1) + 
      ((x0 - x1)*(x0 - x3)*y2/a2) + 
      ((x0 - x1)*(x0 - x2)*y3/a3) ;

  }   
  return real_max_value ;
}     

template <class RandomAccessIterType>
float parabolic_3points_fit_x0(const RandomAccessIterType& begin_iter,
                                      const RandomAccessIterType& end_iter)  
{ 
  //  float real_max_value;                    
  const RandomAccessIterType max_iter = std::max_element(begin_iter,end_iter);
  if (max_iter==end_iter-1) return 0;           // In case maximum is at the borders of the image     
  if (max_iter==begin_iter) return 0;
  //else
  
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
  float x0 = 0.5*(x1*a1*(y2*a3+y3*a2)+x2*a2*(y1*a3+y3*a1)+x3*a3*(y1*a2+y2*a1))
    /(y1*a2*a3+y2*a1*a3+y3*a1*a2) ; 
  return x0 ;     
}                          


/***************************************************
                 instantiations
***************************************************/
template 
std::list<ResolutionIndex<3,float> > 
find_fwhm_in_image<>(DiscretisedDensity<3,float> & input_image, 
                     const unsigned int num_maxima, 
                     const float level, 
                     const int dimension,
                     const bool nema);

  
END_NAMESPACE_STIR
