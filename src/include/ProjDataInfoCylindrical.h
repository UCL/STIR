//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class ProjDataInfoCylindrical

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
  */
#ifndef __ProjDataInfoCylindrical_H__
#define __ProjDataInfoCylindrical_H__


#include "ProjDataInfo.h"

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
  ProjDataInfoCylindrical(const shared_ptr<Scanner>& scanner_ptr,
    const VectorWithOffset<int>& num_axial_poss_per_segment,
    const VectorWithOffset<int>& min_ring_diff, 
    const VectorWithOffset<int>& max_ring_diff,
    const int num_views,const int num_tangential_poss);

  inline virtual float get_tantheta(const Bin&) const; 
		       
  inline float get_phi(const Bin&) const; 
 
  inline float get_t(const Bin&) const;

  //! Return z-coordinate of the middle of the LOR
  /*!
  The 0 of the z-axis is chosen in the middle of the scanner.
  
    \warning Current implementation assumes that the axial positions are always 'centred',
    i.e. get_m(Bin(..., min_axial_pos_num,...)) == - get_m(Bin(..., max_axial_pos_num,...))
  */  
  inline float get_m(const Bin&) const;

 
  //void set_azimuthal_angle_sampling(const float angle);
 
  //void set_axial_sampling(const float samp, int segment_num);
  
  //! Get the azimuthal sampling (in radians)
  inline float get_azimuthal_angle_sampling() const;

  //! Get the axial sampling (e.g in z_direction)
  /*! 
   \warning The implementation of this function currently assumes that the axial
   sampling is equal to the ring spacing for non-spanned data 
   (i.e. no axial compression), while it is half the 
   ring spacing for spanned data.
  */
  inline float get_axial_sampling(int segment_num) const;
  
  //! Get average ring difference for the given segmnet
  inline float get_average_ring_difference(int segment_num) const;
  //! Get minimum ring difference for the given segment 
  inline int get_min_ring_difference(int segment_num) const;
  //! Get maximun ring difference for the given segmnet 
  inline int get_max_ring_difference(int segment_num) const;

  //! Set minimum ring difference
  void set_min_ring_difference(int min_ring_diff_v, int segment_num);
  //! Set maximum ring difference
  void set_max_ring_difference(int max_ring_diff_v, int segment_num);


  //! Get detector ring radius
  inline float get_ring_radius() const;
  //! Get detector ring spacing
  inline float get_ring_spacing() const;

  //! Set detector ring spacing
  void set_ring_spacing(float ring_spacing_v);

  //! Get the mashing factor, i.e. how many 'original' views are combined
  /*! This gets the result by comparing the number of detectors in the scanner_ptr
      with the acutal number of views.
   */
  inline int get_view_mashing_factor() const;

  //! Find which segment a particular ring difference belongs to
  /*!
    If there is no segment containing this ring difference, this will return 
    a segment number larger than the maximum or smaller than the minimum.
    */
  inline int 
    get_segment_num_for_ring_difference(const int ring_diff) const;

  //! Find which segment and axial position a ring pair contributes to
  /*!
    \a ring1, \a ring2 have to between 0 and scanner.get_num_rings()-1.

    For CTI data with span, this essentially implements a 'michelogram'.

    \warning Current implementation assumes that the axial positions start from 0 for
    the first ring-pair in the segment.

    \warning When the segment_num returned would be out of range, an out-of-range 
    index will occur (checked by asserts in debug mode, but not in optimised mode).

    \warning The implementation of this function currently assumes that the axial
    sampling is equal to the ring spacing for non-spanned data 
    (i.e. no axial compression), while it is half the 
    ring spacing for spanned data.
  */
  inline void 
    get_segment_axial_pos_num_for_ring_pair(int& segment_num,
                                            int& ax_pos_num,
                                            const int ring1,
                                            const int ring2) const;

  virtual string parameter_info() const;

protected:

  float azimuthal_angle_sampling;
  float ring_radius;

private:
  float ring_spacing;
  VectorWithOffset<int> min_ring_diff; 
  VectorWithOffset<int> max_ring_diff;

  int view_mashing_factor;
  /*
    Next members have to be mutable as they can be modified by const member 
    functions. We need this because of the presence of set_min_ring_difference()
    which invalidates these precalculated arrays.
    If your compiler does not support mutable (and you don't want to upgrade
    it to something more sensible), your best bet is to remove the 
    set_*ring_difference functions, and move the content of  
    initialise_ring_diff_arrays() to the constructor.
  */
  //! This member will signal if the arrays below contain sensible info or not
  mutable bool ring_diff_arrays_computed;
  //! This member stores the offsets used in get_m()
  mutable VectorWithOffset<float> m_offset;

  //! This member stores the offsets used in get_segment_axial_pos_num_for_ring_pair()
  mutable VectorWithOffset<int> ax_pos_num_offset;
  //! This member stores a table converting ring differences to segment numbers
  mutable VectorWithOffset<int> ring_diff_to_segment_num;

  //! This function sets all of the above
  void initialise_ring_diff_arrays() const;

  inline int get_num_rings_per_axial_pos(const int segment_num) const;

};


END_NAMESPACE_TOMO

#include "ProjDataInfoCylindrical.inl"

#endif
