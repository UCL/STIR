//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Declaration of class ProjDataInfoCylindricalNoArcCorr

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#ifndef __ProjDataInfoCylindricalNoArcCorr_H__
#define __ProjDataInfoCylindricalNoArcCorr_H__


#include "ProjDataInfoCylindrical.h"

START_NAMESPACE_TOMO

/*!
  \ingroup buildblock 
  \brief Projection data info for data which are not arc-corrected

  This means that 'tangential_pos_num' actually indexes an angular coordinate
  with a particular angular sampling (usually given by half the angle between
  detectors). That is
  \code
    get_s(Bin(..., tang_pos_num)) == ring_radius * sin(tang_pos_num*angular_increment)
  \endcode
  This class does \c not take the 'interleaving' into account which is 
  customarily applied to raw PET data.
  */
class ProjDataInfoCylindricalNoArcCorr : public ProjDataInfoCylindrical
{

public:
  //! Default constructor (leaves object in ill-defined state)
  ProjDataInfoCylindricalNoArcCorr();
  //! Constructor completely specifying all parameters
  /*! See class documentation for info on parameters */
  ProjDataInfoCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_ptr,
    const float ring_radius, const float angular_increment,
    const  VectorWithOffset<int>& num_axial_pos_per_segment,
    const  VectorWithOffset<int>& min_ring_diff_v, 
    const  VectorWithOffset<int>& max_ring_diff_v,
    const int num_views,const int num_tangential_poss);

  //! Constructor which gets \a ring_radius and \a angular_increment from the scanner
  /*! \a angular_increment is determined as Pi divided by the number of detectors in a ring */
   ProjDataInfoCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_ptr,
    const  VectorWithOffset<int>& num_axial_pos_per_segment,
    const  VectorWithOffset<int>& min_ring_diff_v, 
    const  VectorWithOffset<int>& max_ring_diff_v,
    const int num_views,const int num_tangential_poss);

  ProjDataInfo* clone() const;
  
  inline virtual float get_s(const Bin&) const;

  virtual string parameter_info() const;

private:
  
  float ring_radius;
  float angular_increment;
  

};

END_NAMESPACE_TOMO

#include "ProjDataInfoCylindricalNoArcCorr.inl"

#endif
