//
// $Id$
//
/*!
\file
\ingroup scatter
\brief Implementations of functions defined in scatter.h
Function calculates the integral along LOR in a 
image (attenuation or emission). 
(From scatter point to detector coordinate)

  \author Pablo Aguiar
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	
	  
		$Date$
		$Revision$
		
		  Copyright (C) 2004- $Date$, Hammersmith Imanet
		  See STIR/LICENSE.txt for details
*/


#include "local/stir/Scatter.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <map>
#include <utility>
#include <functional>


START_NAMESPACE_STIR

template <class coordT>
bool 
operator<(const Coordinate3D<coordT>& p1, 
          const Coordinate3D<coordT>& p2) 
{
	return 
		p1[1]<p2[1] ||
		(p1[1]==p2[1]&& 
		(p1[2]<p2[2] || 
		(p1[2]==p2[2] && 
		(p1[3]<p2[3]))));
}

template <class coordT>
bool 
operator==(const Coordinate3D<coordT>& p1, 
		   const Coordinate3D<coordT>& p2) 
{
	return 
		p1[1]==p2[1] && p1[2]==p2[2] && p1[3]==p2[3];
}

#if 0
struct KeyType
{
	unsigned long long key;
	KeyType(unsigned a, unsigned b) :
	  key(a | (b<<(sizeof(unsigned)*8)))
	  {
		  unsigned long bshift = b<<(sizeof(unsigned)*8);
		  assert(sizeof(unsigned)*2==sizeof(unsigned long long));
	      assert((1<<3)==8);
	  }
};
inline bool operator<(const KeyType& k1, const KeyType& k2)
{
	return k1.key<k2.key;
}
#else
struct KeyType : public std::pair<unsigned, unsigned > 
{
	KeyType(unsigned a, unsigned b)
	{
		first=a;
		second=b;
	}
};
#endif

float cached_factors(const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
	  			     const unsigned scatter_point_num, 
					 const unsigned det_num,
					 const image_type input_image_type)
{				

	typedef std::map<KeyType,float> map_type;
	const VoxelsOnCartesianGrid<float>& image =
			dynamic_cast<const VoxelsOnCartesianGrid<float>& >(discretised_image);
	if(input_image_type==act_image_type)
	{
	    static map_type cached_integral_act_scattpoint_det;		
		const KeyType key(scatter_point_num, det_num);

		{
			// test if key is present in map, if so, return its value
			map_type::const_iterator  iter =
				cached_integral_act_scattpoint_det.find(key);
			if (iter != cached_integral_act_scattpoint_det.end())
				return iter->second;
		}				
		cached_integral_act_scattpoint_det[key] = integral_scattpoint_det(
			discretised_image,	  	
			scatt_points_vector[scatter_point_num].coord,
			detection_points_vector[det_num]);			
		return cached_integral_act_scattpoint_det[key];
	}	

	if(input_image_type==att_image_type)
	{	
		static map_type cached_integral_att_scattpoint_det;	
		const KeyType key(scatter_point_num, det_num);
		{
			// test if key is present in map, if so, return its value
			map_type::const_iterator  iter =
				cached_integral_att_scattpoint_det.find(key);
			if (iter != cached_integral_att_scattpoint_det.end())
				return iter->second;
		}						
		cached_integral_att_scattpoint_det[key] = 
			exp(-integral_scattpoint_det(
			discretised_image,	  	
			scatt_points_vector[scatter_point_num].coord,
			detection_points_vector[det_num]));			
		return cached_integral_att_scattpoint_det[key];
	}		
	else 
		return EXIT_FAILURE;
}						   
END_NAMESPACE_STIR

