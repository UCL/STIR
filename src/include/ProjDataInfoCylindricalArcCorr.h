//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Declaration of class ProjDataInfoCylindricalArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#ifndef __ProjDataInfoCylindricalArcCorr_H__
#define __ProjDataInfoCylindricalArcCorr_H__


#include "ProjDataInfoCylindrical.h"

START_NAMESPACE_TOMO

/*!
  \ingroup buildblock 
  \brief projection data info for arc-corrected data
  */
class ProjDataInfoCylindricalArcCorr : public ProjDataInfoCylindrical
{

public:
  //! Constructors
  inline ProjDataInfoCylindricalArcCorr();
  inline  ProjDataInfoCylindricalArcCorr(const shared_ptr<Scanner> scanner_ptr,float bin_size,
    const  VectorWithOffset<int>& num_axial_pos_per_segment,
    const  VectorWithOffset<int>& min_ring_diff_v, 
    const  VectorWithOffset<int>& max_ring_diff_v,
    const int num_views,const int num_tangential_poss);

  
  inline virtual float get_tantheta(int segment_num,int view_num,int axial_position_num, int transaxial_position_num) const; 
  inline virtual float get_s(int segment_num,int view_num,int axial_position_num, int transaxial_position_num) const;
  //! Set tangential sampling
  inline void set_tangential_sampling(const float bin_size);
  //! Get tangential sampling
  inline float get_tangential_sampling() const;
  inline ProjDataInfo* clone() const;

private:
  
  float bin_size;
  

};

END_NAMESPACE_TOMO

#include "ProjDataInfoCylindricalArcCorr.inl"

#endif
