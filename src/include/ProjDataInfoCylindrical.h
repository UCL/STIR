//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class ProjDataInfoCylindrical

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
  */
#ifndef __ProjDataInfoCylindrical_H__
#define __ProjDataInfoCylindrical_H__


#include"ProjDataInfo.h"

START_NAMESPACE_TOMO
/*!
  \ingroup buildblock 
  \brief projection data info for data corresponding to a 
  'cylindrical' sampling.

  These data are organised by ring differences (allowing for
  merging of some ring differences into 1 segment). The class is general
  enough to have both the CTI 'spanned' format, and the GE Advance
  format.
*/
// TODOdoc more
class ProjDataInfoCylindrical: public ProjDataInfo
{

public:
  //! Constructors
  ProjDataInfoCylindrical();
  //! Constructor given all the necessary information
  /*! The min and max ring difference in each segment are passed
  as VectorWithOffsets. All three vectors have to have index ranges
  from min_segment_num to max_segment_num.
  
  \warning Most of this library assumes that segment 0 corresponds
  to an average ring difference of 0.
  */
  ProjDataInfoCylindrical(const shared_ptr<Scanner> scanner_ptr,
    const VectorWithOffset<int>& num_axial_poss_per_segment,
    const VectorWithOffset<int>& min_ring_diff, 
    const VectorWithOffset<int>& max_ring_diff,
    const int num_views,const int num_tangential_poss);

  inline virtual float get_tantheta(const Bin&) const; 
		       
  inline float get_phi(const Bin&) const; 
 
  inline float get_t(const Bin&) const;

  //! Return z-coordinate of the middle of the LOR
  inline float get_m(const Bin&) const;

#ifdef SET  
  inline void set_azimuthal_angle_sampling(const float angle);
#endif
 
  //inline void set_axial_sampling(const float samp, int segment_num);
  //! Get the azimuthal sampling (e.g in plane sampling)
  inline float get_azimuthal_angle_sampling() const;
  //! Get the axial sampling (e.g in z_direction)
  inline float get_axial_sampling(int segment_num) const;
  
  //! Get average ring difference for the given segmnet
  inline float get_average_ring_difference(int segment_num) const;
  //! Get minimum ring difference for the given segment 
  inline int get_min_ring_difference(int segment_num) const;
  //! Get maximun ring difference for the given segmnet 
  inline int get_max_ring_difference(int segment_num) const;

  //! Set minimum ring difference
  inline void set_min_ring_difference(int min_ring_diff_v, int segment_num);
  //! Set maximum ring difference
  inline void set_max_ring_difference(int max_ring_diff_v, int segment_num);


  //! Get detector ring radius
  inline float get_ring_radius() const;
  //! Get detector ring spacing
  inline float get_ring_spacing() const;

  //! Set detector ring spacing
  inline void set_ring_spacing(float ring_spacing_v);



protected:

  float azimuthal_angle_sampling;
  float ring_radius;

private:
  float ring_spacing;
  VectorWithOffset<int> min_ring_diff; 
  VectorWithOffset<int> max_ring_diff;
  //! This member stores the offsets used in get_m()
  VectorWithOffset<float> m_offset;

};


END_NAMESPACE_TOMO

#include "ProjDataInfoCylindrical.inl"

#endif
