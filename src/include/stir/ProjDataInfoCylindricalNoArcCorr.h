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
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __ProjDataInfoCylindricalNoArcCorr_H__
#define __ProjDataInfoCylindricalNoArcCorr_H__


#include "stir/ProjDataInfoCylindrical.h"

START_NAMESPACE_STIR

class Succeeded;

/*!
  \ingroup buildblock 
  \brief Projection data info for data which are not arc-corrected.

  For this class, 'tangential_pos_num' actually indexes an angular coordinate
  with a particular angular sampling (usually given by half the angle between
  detectors). That is
  \code
    get_s(Bin(..., tang_pos_num)) == ring_radius * sin(tang_pos_num*angular_increment)
  \endcode

  This class also contains some functions specific for (static) full-ring PET 
  scanners. In this case, it is assumed that for 'raw' data (i.e. no mashing)
  sinogram space is 'interleaved': 2 adjacent LOR_angles are
  merged to 1 'view', while the corresponding bins are
  interleaved:

  \verbatim
     before interleaving               after interleaving
     a00     a01     a02 ...      view 0: a00 a10 a01 a11  ...
         a10     a11     ...
     a20     a21     a22 ...      view 1: a20 a30 a21 a31 ...
         a30     a31     ...
	 \endverbatim
  This (standard) interleaving is done because for 'odd' LOR_angles there
  is no LOR which goes through the origin.


  Interchanging the 2 detectors:
  ------------------------------
  When the ring difference = 0 (i.e. a 2D - or direct - sinogram),
  interchanging the 2 detectors does not change the LOR. This is why
  (in 2D) one gets away with a full sinogram size of
  num_views * 2 * num_views, where the size of 'detector-space' is
  twice as large.
  However, in 3D, interchanging the detectors, also interchanges the
  rings, and we have a totally different LOR. One has 2 options:
  - have 1 sinogram with twice as many views, together with the rings
    as 'unordered pair' (i.e. ring_difference is always >0)
  - have 2 sinograms of the same size as in 2D, together with the rings
    as 'ordered pair' (i.e. ring_difference can be positive and negative).
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

  //! Gets s coordinate in mm
  /*! \warning   
    This does \c not take the 'interleaving' into account which is 
    customarily applied to raw PET data.
  */
  inline virtual float get_s(const Bin&) const;

  virtual string parameter_info() const;

  //! This gets view_num and tang_pos_num for a particular detector pair
  /*! This function makes only sense if the scanner is a full-ring scanner 
      with discrete detectors and there is no rotation or wobble.

      \arg view runs currently from 0 to num_views-1

      \arg tang_pos_num is centred around 0, where 0 corresponds
      to opposing detectors. The maximum range of tangential positions for any
      scanner is (-(num_detectors)/2,-(num_detectors)/2+num_detectors) , but 
      this range is never used (as the 'last' tang_pos_num would be a LOR 
      between two adjacent detectors).

      \arg det_num1, \a det_num2 run from 0 to num_detectors-1

      \return \c true if the detector pair is stored in a positive segment,
              \c false otherwise.

      \warning Current implementation only checked for Siemens/CTI scanners.
      It assumes interleaved data.

      \see get_det_num_pair_for_view_tangential_pos_num()
  */			
  inline bool 
     get_view_tangential_pos_num_for_det_num_pair(int& view_num,
						 int& tang_pos_num,
						 const int det_num1,
						 const int det_num2) const;
  //! This routine gets \a det_num1 and \a det_num2
  /*! 
      It sets the detectors in a particular order (i.e. it fixes the 
      orientation of the LOR) corresponding to the detector pair belonging
      to a positive segment.

      \warning This currently only works when no mashing of views is used,
      as otherwise there would be multiple detector pairs corresponding 
      to a single bin.
      
      \see get_view_tangential_pos_num_for_det_num_pair() for info and 
      restrictions.
   */
  inline void
    get_det_num_pair_for_view_tangential_pos_num(
						 int& det_num1,
						 int& det_num2,
						 const int view_num,
						 const int tang_pos_num) const;


  //! This gets view_num and tang_pos_num for a particular detector pair
  /*! 
    \return Succeeded::yes when a corresponding segment is found
    \see get_view_tangential_pos_num_for_det_num_pair() for restrictions
  */		       
  inline Succeeded 
    get_bin_for_det_pair(Bin&,
			 const int det_num1, const int ring_num1,
			 const int det_num2, const int ring_num2) const;


  //! This routine gets the detector pair corresponding to a bin.
  /*! 
    \see get_det_pair_for_view_tangential_pos_num() for
    restrictions. In addition, this routine only works for span=1 data,
    i.e. no axial compression.
   */
  inline void
    get_det_pair_for_bin(
			 int& det_num1, int& ring_num1,
			 int& det_num2, int& ring_num2,
			 const Bin&) const;

private:
  
  float ring_radius;
  float angular_increment;
  

};

END_NAMESPACE_STIR

#include "stir/ProjDataInfoCylindricalNoArcCorr.inl"

#endif
