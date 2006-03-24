/*

  \author Nikolaos Dikaios
  \author Kris Thielemans
  $Date$
  
*/


#include <math.h>
#include "stir/CartesianCoordinate3D.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

inline
float
compute_emis_to_scatter_points_solid_angle_factor_doubles11(const CartesianCoordinate3D<float>& scatter_point_1,
							    const CartesianCoordinate3D<float>& scatter_point_2,
							    const CartesianCoordinate3D<float>& emis_point,
							    const CartesianCoordinate3D<float>& voxel_size_atn)
{
 // return 1 to reproduce the estimate scatter as it used to be 
 

  return 1.F ;


  // const CartesianCoordinate3D<float> voxel_vectors_atn(voxel_size_atn.y()*voxel_size_atn.x(), 
  //					       voxel_size_atn.z()*voxel_size_atn.x(),
  //					       voxel_size_atn.z()*voxel_size_atn.y());  
  
  const CartesianCoordinate3D<float> dist_vector_1 = scatter_point_1 - emis_point ; 
  // const CartesianCoordinate3D<float> abs_dist_vector_1(std::fabs(dist_vector_1[1]),
  //						       std::fabs(dist_vector_1[2]),
  //					       std::fabs(dist_vector_1[3]));
      
  const CartesianCoordinate3D<float> dist_vector_2 = scatter_point_2 - emis_point ; 
// const CartesianCoordinate3D<float> abs_dist_vector_2(std::fabs(dist_vector_2[1]),
//					       std::fabs(dist_vector_2[2]),
//					       std::fabs(dist_vector_2[3]));
 
  const float dist_emis_sp1_squared = norm_squared(dist_vector_1);
      
  const float dist_emis_sp2_squared = norm_squared(dist_vector_2);
      
  const float emis_to_scatter_point_1_solid_angle_factor = 1.F / dist_emis_sp1_squared ;
//  inner_product(voxel_vectors_atn, abs_dist_vector_1)/ 
//    std::pow( dist_emis_sp1_squared, 1.5F);
 
   const float emis_to_scatter_point_2_solid_angle_factor = 1.F / dist_emis_sp2_squared ; 
   //  inner_product(voxel_vectors_atn, abs_dist_vector_2)/ 
   // std::pow( dist_emis_sp2_squared, 1.5F);

   return
     std::min(std::min(emis_to_scatter_point_1_solid_angle_factor,
		       emis_to_scatter_point_2_solid_angle_factor),
	      static_cast<float>(_PI/2.F));
}

inline
float
compute_sc1_to_sc2_solid_angle_factor_doubles20(const CartesianCoordinate3D<float>& scatter_point_1,
						const CartesianCoordinate3D<float>& scatter_point_2,
						const CartesianCoordinate3D<float>& voxel_size_atn)
  
{

 // return 1 to reproduce the estimate scatter as it used to be 
 

  return 1.F ;


  const CartesianCoordinate3D<float> voxel_vectors_atn(voxel_size_atn.y()*voxel_size_atn.x(), 
  				       voxel_size_atn.z()*voxel_size_atn.x(),
  				       voxel_size_atn.z()*voxel_size_atn.y());  
  
  const CartesianCoordinate3D<float> dist_vector = scatter_point_2 - scatter_point_1 ; 
  const CartesianCoordinate3D<float> abs_dist_vector(std::fabs(dist_vector[1]),
						     std::fabs(dist_vector[2]),
						     std::fabs(dist_vector[3]));
  
  const float dist_sp1_sp2_squared = norm_squared(dist_vector);
    
  const float scatter_points_solid_angle_factor = //1.F / dist_sp1_sp2_squared ;
    inner_product(voxel_vectors_atn, abs_dist_vector)/ 
    std::pow( dist_sp1_sp2_squared, 1.5F);

  return scatter_points_solid_angle_factor ;

} 

inline
float
compute_emis_to_det_points_solid_angle_factor(
					      const CartesianCoordinate3D<float>& emis_point,
					      const CartesianCoordinate3D<float>& detector_coord)
{
 // return 1 to reproduce the estimate scatter as it used to be 
 

  return 1.F ;


  
  const CartesianCoordinate3D<float> dist_vector = emis_point - detector_coord ;
 
  const CartesianCoordinate3D<float> abs_dist_vector(std::fabs(dist_vector[1]),
						     std::fabs(dist_vector[2]),
						     std::fabs(dist_vector[3]));
  
  const float dist_emis_det_squared = norm_squared(dist_vector);

  const float emis_det_solid_angle_factor = 1.F/ dist_emis_det_squared ;

  return emis_det_solid_angle_factor ;
}

END_NAMESPACE_STIR

