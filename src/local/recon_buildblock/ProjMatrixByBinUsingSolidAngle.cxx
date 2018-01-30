//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2012, IRSL
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for stir::ProjMatrixByBinUsingSolidAngle

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project

*/



#include "local/stir/recon_buildblock/ProjMatrixByBinUsingSolidAngle.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfo.h"
#include "stir/round.h"
#include <algorithm>
#include <math.h>

START_NAMESPACE_STIR


const char * const 
ProjMatrixByBinUsingSolidAngle::registered_name =
  "Solid Angle";

ProjMatrixByBinUsingSolidAngle::
ProjMatrixByBinUsingSolidAngle()
{
  set_defaults();
}

void 
ProjMatrixByBinUsingSolidAngle::initialise_keymap()
{
  ProjMatrixByBin::initialise_keymap();
  parser.add_start_key("Solid Angle Matrix Parameters");
  parser.add_stop_key("End Solid Angle Matrix Parameters");
}


void
ProjMatrixByBinUsingSolidAngle::set_defaults()
{
  ProjMatrixByBin::set_defaults();
}


void
ProjMatrixByBinUsingSolidAngle::
set_up(		 
       const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
       const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr  
       )
{
  proj_data_info_ptr= proj_data_info_ptr_v; 
  const VoxelsOnCartesianGrid<float> * image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByBinUsingSolidAngle initialised with a wrong type of DiscretisedDensity\n");

 
  voxel_size = image_info_ptr->get_voxel_size();
  origin = image_info_ptr->get_origin();
  image_info_ptr->get_regular_range(min_index, max_index);

  symmetries_ptr
    .reset(new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_ptr,
						       density_info_ptr));
  
};


//////////////////////////////////////
                               

inline int sign(const float x)
{
  return x>0 ? 1 : -1;
}

// integral from centre of trapezoid to s
// height is 1, m2 is half length of plateau, m3 is half total length 
inline float trapezoid_integral(const float s, 
			      const float m2,
			      const float m3)
{
  const float abs_s = fabs(s);
  const float abs_s_min_m3 = abs_s-m3;
  const float maxval = m2+m3;
  if(abs_s_min_m3 >= 0)
    return sign(s)*maxval/2;
  if(abs_s >= m2)
  {
    const float minval = m3-m2;
    return sign(s) *(maxval - square(abs_s_min_m3)/minval)/2;
  }
  return s;
}



template <typename T>
inline T cube(const T x)
{
  return x*x*x;
}

// both of height 1
// 0/0 when m2? == m3?
inline float convolution_2_trapezoids(const float x, 
			      const float m21,
			      const float m31,                              
			      const float m22,
			      const float m32)
{
  return 
(cube(fabs(m21 - m22 - x)) + cube(fabs(m21 + m22 - x)) - 
   cube(fabs(m22 - m31 - x)) - cube(fabs(m22 + m31 - x)) - 
   cube(fabs(m21 - m32 - x)) + cube(fabs(m31 - m32 - x)) - 
   cube(fabs(m21 + m32 - x)) + cube(fabs(m31 + m32 - x)) + 
   cube(fabs(m21 - m22 + x)) + cube(fabs(m21 + m22 + x)) - 
   cube(fabs(m22 - m31 + x)) - cube(fabs(m22 + m31 + x)) - 
   cube(fabs(m21 - m32 + x)) + cube(fabs(m31 - m32 + x)) - 
   cube(fabs(m21 + m32 + x)) + cube(fabs(m31 + m32 + x)))/
  (12*(m21 - m31)*(m22 - m32));
}

/*
inline float VOI_small_voxel(const float s_voxel, 
			      const float half_voxel_size)
{
  const float res = 
    min(.5,s_voxel+half_voxel_size) - max(-.5, s_voxel-half_voxel_size);
  return res<0 ? 0 : res;
}

inline float VOI(const float s_voxel, 
		  const float half_voxel_size,
		  const float halfcosminsin,
		  const float halfcosplussin)
{
  
  return 
    VOI_small_voxel(s_voxel+halfcosminsin, half_voxel_size) +
    VOI_small_voxel(s_voxel-halfcosminsin, half_voxel_size) +
    VOI_small_voxel(s_voxel+halfcosplussin, half_voxel_size) +
    VOI_small_voxel(s_voxel-halfcosplussin, half_voxel_size);
}
*/

inline float VOI(const float s_voxel, 
		  const float half_bin_size,
		  const float halfcosminsin,
		  const float halfcosplussin)
{
  return 
    trapezoid_integral(half_bin_size-s_voxel, halfcosminsin, halfcosplussin) - 
    trapezoid_integral(-half_bin_size- s_voxel, halfcosminsin, halfcosplussin);
}

inline float SA_rotated_voxel(const float s_voxel, 
			      const float half_voxel_size,
			      const float m2,
			      const float m3)
{
  return 
    trapezoid_integral(half_voxel_size+s_voxel, m2, m3) - 
    trapezoid_integral(-half_voxel_size+s_voxel, m2, m3);
}

inline float SAapprox(const float s_voxel, 
		 const float half_voxel_size,
		 const float m2,
		 const float m3,
		 const float halfcosminsin,
		 const float halfcosplussin)
{
  
  return 
    /*
    SA_rotated_voxel(s_voxel+halfcosminsin/2, half_voxel_size, m2, m3) +
    SA_rotated_voxel(s_voxel-halfcosminsin/2, half_voxel_size, m2, m3) +
    SA_rotated_voxel(s_voxel+halfcosplussin/2, half_voxel_size, m2, m3) +
    SA_rotated_voxel(s_voxel-halfcosplussin/2, half_voxel_size, m2, m3);
    */
        SA_rotated_voxel(s_voxel+halfcosminsin/2, .5, m2, m3) +
    SA_rotated_voxel(s_voxel-halfcosminsin/2, .5, m2, m3) +
    SA_rotated_voxel(s_voxel+halfcosplussin/2, .5, m2, m3) +
    SA_rotated_voxel(s_voxel-halfcosplussin/2, .5, m2, m3);

}

inline float SA(const float s_voxel, 
		 const float half_voxel_size,
		 const float m2,
		 const float m3,
		 const float halfcosminsin,
		 const float halfcosplussin)
{
  return 
    halfcosminsin==halfcosplussin ? 
     SA_rotated_voxel(s_voxel, .5, m2, m3)
     :
     convolution_2_trapezoids(s_voxel,m2,m3, halfcosminsin, halfcosplussin);
}


void 
ProjMatrixByBinUsingSolidAngle::
calculate_proj_matrix_elems_for_one_bin(
                                        ProjMatrixElemsForOneBin& lor) const
{
  const Bin bin = lor.get_bin();
  //assert(bin.axial_pos_num() == 0);
  assert(bin.tangential_pos_num() >= 0);
  assert(bin.segment_num()  >= 0  );
  assert(bin.segment_num() <= proj_data_info_ptr->get_max_segment_num());    
  assert(bin.view_num() <= proj_data_info_ptr->get_num_views()/4);    
  assert(bin.view_num()>=0);

  assert(lor.size() == 0);
     
  const float tantheta = proj_data_info_ptr->get_tantheta(bin);
  const float costheta = 1/sqrt(1+square(tantheta));
  const float phi = proj_data_info_ptr->get_phi(bin);
  const float cphi = cos(phi);
  const float sphi = sin(phi);
  const float s_in_mm = proj_data_info_ptr->get_s(bin);
  const float t_in_mm = proj_data_info_ptr->get_t(bin);


  // use FOV which is circular, and is slightly 'inside' the image to avoid 
  // index out of range
  const float fovrad_in_mm   = 
    min((min(max_index.x(), -min_index.x())-1)*voxel_size.x(),
        (min(max_index.y(), -min_index.y())-1)*voxel_size.y()); 
  if (s_in_mm >= fovrad_in_mm) return;


  const float sampling_distance_of_adjacent_LORs_z =
    proj_data_info_ptr->get_sampling_in_t(bin)/costheta;
 

  // find number of LORs we have to take, such that we don't miss voxels
  // we have to subtract a tiny amount from the quotient, to avoid having too many LORs
  // solely due to numerical rounding errors
  const int num_lors_per_axial_pos = 
    static_cast<int>(ceil(sampling_distance_of_adjacent_LORs_z / voxel_size.z() - 1.E-3));

  assert(num_lors_per_axial_pos>0);
  // code below is currently restricted to 2 LORs
  assert(num_lors_per_axial_pos<=2);

  // merging code assumes integer multiple
  assert(fabs(sampling_distance_of_adjacent_LORs_z 
              - num_lors_per_axial_pos*voxel_size.z()) <= 1E-4);


  // find offset in z, taking into account if there are 1 or 2 LORs
  // KT 20/06/2001 take origin.z() into account
  const float offset_in_z = 
    (num_lors_per_axial_pos == 1 || tantheta == 0 ? 
     0.F : -sampling_distance_of_adjacent_LORs_z/4)
    - origin.z();


  /* Intersection points of LOR and image FOV (assuming infinitely long scanner)*/
  /* compute   X1f, Y1f,,Z1f et al in voxelcoordinates. */
  
  const float TMP2f = sqrt(square(fovrad_in_mm) - square(s_in_mm));
  //const float X2f = (s_in_mm * cphi - sphi * TMP2f)/voxel_size.x();
  const float X1f = (s_in_mm * cphi + sphi * TMP2f)/voxel_size.x();
  //const float Y2f = (s_in_mm * sphi + cphi * TMP2f)/voxel_size.y();
  const float Y1f = (s_in_mm * sphi - cphi * TMP2f)/voxel_size.y();

  const float Z1f = 
    (t_in_mm/costheta - TMP2f*tantheta + offset_in_z)/voxel_size.z() 
    + (max_index.z() + min_index.z())/2.F;
  //const float Z2f = 
  // (t_in_mm/costheta + TMP2f*tantheta + offset_in_z)/voxel_size.z()
  //  + (max_index.z() + min_index.z())/2.F;

  const CartesianCoordinate3D<float> start_point(Z1f,Y1f,X1f);  
  //const CartesianCoordinate3D<float> stop_point(Z2f,Y2f,X2f);  

  const float tphi = sphi/cphi;

  // do actual computation for this LOR
    const float halfcosplussin = (1+tphi)/2;
    const float halfcosminsin = (1-tphi)/2;
    const float bin_size = proj_data_info_ptr->get_sampling_in_s(bin)*2;// factor 2 necessary for actual detector size TODO
    //? 1/x *2 ?
    const float half_bin_size = bin_size/2/voxel_size.x()/cphi;
    const float fovrad = fovrad_in_mm/voxel_size.x();
    //const float half_voxel_size = cphi;
    const float half_tube_length =
      sqrt(square(proj_data_info_ptr->get_scanner_ptr()->get_effective_ring_radius()) -
           square(s_in_mm))/voxel_size.x();
    { 
      // Compute first pixel in a beam (not on a ray )
      const float max_s = halfcosplussin*2 + half_bin_size +0.01F;
      const float min_s = -max_s;
      
      // start guaranteed on the right of the tube
      int XonpreviousRow = min(static_cast<int>(ceil(X1f + max_s)), max_index[3]);
      int Y = static_cast<int>(ceil(Y1f));
      float s_voxel_onpreviousRow=XonpreviousRow+Y*tphi-s_in_mm/voxel_size.x()/cphi;
      // next assert is not true because we pushed X inside the FOV again
      // assert(s_voxel_onpreviousRow > max_s);
        
      int X;
      float s_voxel;
      do
      {
        while (s_voxel_onpreviousRow>max_s) 
	{
	  /* horizontal*/
	  --XonpreviousRow;
	  --s_voxel_onpreviousRow;
        }
        X= XonpreviousRow;
        s_voxel = s_voxel_onpreviousRow;
	
        const float depth = fabs(-X* sphi + Y*cphi);
        const float rel_depth = depth/half_tube_length;
        const float m3 = half_bin_size;
        const float m2 = m3 * rel_depth;
        const float height_of_trapezoid = 2*m3/(1+rel_depth)/half_tube_length*(cphi);

        // this should have a check on X>=min_index[3] if s<0 was included
        while (s_voxel>min_s) 
	{
	  const float Pbv = 
            //VOI(s_voxel, half_bin_size, halfcosminsin,halfcosplussin);
            SA(s_voxel, half_bin_size, m2,m3,halfcosminsin,halfcosplussin) *
              height_of_trapezoid;
          //assert(Pbv>0);
          // warning: threshold depends on scale (currently VOI max is 1)
          if (Pbv>.0001)
	  lor.push_back(ProjMatrixElemsForOneBin::value_type(Coordinate3D<int>(0,Y,X),Pbv)); 
          // this could be made with  break
          /* horizontal*/
	  --X;
	  --s_voxel;	 
        }
        /* vertical */
        ++Y;
        s_voxel_onpreviousRow+=tphi;
        
      }
      while (square(X) + square(Y) < square(fovrad));
    }

#if 0
  // now add on other LORs
  if ( num_lors_per_axial_pos>1)
  {      
    
    assert(num_lors_per_axial_pos==2);
    if (tantheta==0 ) 
    { 
      assert(Z1f == -origin.z()/voxel_size.z());
      add_adjacent_z(lor);
    }
    else
    { 
      // lor.merge( lor2 );   
      merge_zplus1(lor);
    }

  } // if( num_lors_per_axial_pos>1)
#endif

}

         

END_NAMESPACE_STIR

