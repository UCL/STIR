//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2011-10-14, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2016, University of Hull
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
  \ingroup projdata

  \brief Declaration of class stir::ProjDataInfo

  \author Nikos Efthimiou
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

*/
#ifndef __stir_ProjDataInfo_H__
#define __stir_ProjDataInfo_H__

#include "stir/VectorWithOffset.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#include <string>

START_NAMESPACE_STIR

template <typename elemT> class Sinogram;
template <typename elemT> class Viewgram;
template <typename elemT> class SegmentByView;
template <typename elemT> class SegmentBySinogram;
template <typename elemT> class RelatedViewgrams;
class DataSymmetriesForViewSegmentNumbers;
class ViewSegmentNumbers;
class Bin;
template <typename T> class LOR;
template <typename T> class LORInAxialAndNoArcCorrSinogramCoordinates;
class PMessage;

/*!
  \ingroup projdata
  \brief An (abstract base) class that contains information on the 
  projection data.

*/
class ProjDataInfo
{
protected:
  typedef ProjDataInfo root_type;

public:

  /********** static members **************/

  //! Ask for the details and return a ProjDataInfo pointer
  static ProjDataInfo* 
  ask_parameters();

  //! Construct a ProjDataInfo suitable for GE Advance data
  //! \warning N.E: TOF mash factor, means no TOF
  static ProjDataInfo*  
  ProjDataInfoGE(const shared_ptr<Scanner>& scanner_ptr,
                 const int max_delta,
                 const int num_views, const int num_tangential_poss,
                 const bool arc_corrected = true,
                 const int tof_mash_factor = 1);

  //! Construct a ProjDataInfo suitable for CTI data
  /*! \c span is used to denote the amount of axial compression (see CTI doc).
     It has to be an odd number. 
     */
  //! \warning N.E: TOF mash factor, means no TOF
  static ProjDataInfo* 
  ProjDataInfoCTI(const shared_ptr<Scanner>& scanner_ptr,
                  const int span, const int max_delta,
                  const int num_views, const int num_tangential_poss,
                  const bool arc_corrected = true,
                  const int tof_mash_factor = 1);
  
  
  /************ constructors ***********/
  // TODO should probably be protected

  //! Construct an empty object
   ProjDataInfo();
  
  //! Constructor setting all relevant info for a ProjDataInfo
   /*! The num_axial_pos_per_segment argument should be such that
       num_axial_pos_per_segment[segment_num] gives you the appropriate value 
       for a particular segment_num
       */
  ProjDataInfo(const shared_ptr<Scanner>& scanner_ptr,
		      const VectorWithOffset<int>& num_axial_pos_per_segment, 
		      const int num_views, 
		      const int num_tangential_poss);

  //! Overloaded Contructor with TOF initialisation
  ProjDataInfo(const shared_ptr<Scanner>& scanner_ptr,
              const VectorWithOffset<int>& num_axial_pos_per_segment,
              const int num_views,
              const int num_tangential_poss,
              const int tof_mash_factor);


  //! Standard trick for a 'virtual copy-constructor' 
  virtual ProjDataInfo* clone() const = 0;

  //! Like clone() but return a shared_ptr
  inline shared_ptr<ProjDataInfo> create_shared_clone() const;

  //! Destructor
  virtual ~ProjDataInfo() {}

  /**************** member functions *********/

  //  ProjDataInfo& operator=(const ProjDataInfo&);

  //! \name Functions that change the data size
  //@{

  //! Set a new range of segment numbers
  /*! 
    This function is virtual in case a derived class needs to know the 
    segment range changed.

    \warning the new range has to be 'smaller' than the old one. */
  virtual void reduce_segment_range(const int min_segment_num, const int max_segment_num);
  //! Set number of views (min_view_num is set to 0).
  /*! This function is virtual in case a derived class needs to know the 
    number of views changed. */
  virtual void set_num_views(const int num_views);
  //! Set number of tangential positions
  /*! This function is virtual in case a derived class needs to know the 
    number of tangential positions changed. */
  virtual void set_num_tangential_poss(const int num_tang_poss);
  //! Set number of axial positions per segment
  /*! 
    \param num_axial_poss_per_segment is a vector with the new numbers,
    where the index into the vector is the segment_num (i.e. it is not
    related to the storage order of the segments or so).

    This function is virtual in case a derived class needs to know the 
    number of axial positions changed. */
  virtual void set_num_axial_poss_per_segment(const VectorWithOffset<int>& num_axial_poss_per_segment); 

  //! Set minimum axial position number for 1 segment
  /*! This function is virtual in case a derived class needs to know the number changed. */
  virtual void set_min_axial_pos_num(const int min_ax_pos_num, const int segment_num);
  //! Set maximum axial position number for 1 segment
  /*! This function is virtual in case a derived class needs to know the number changed. */
  virtual void set_max_axial_pos_num(const int max_ax_pos_num, const int segment_num);
  
  //! Set minimum tangential position number
  /*! This function is virtual in case a derived class needs to know the number changed. */
  virtual void set_min_tangential_pos_num(const int min_tang_poss);
  //! Set maximum tangential position number
  /*! This function is virtual in case a derived class needs to know the number changed. */
  virtual void set_max_tangential_pos_num(const int max_tang_poss);
  //! The the tof mashing factor. Min and Max timing position will be recalculated.
  virtual void set_tof_mash_factor(const int new_num);
  //@}

  //! \name Functions that return info on the data size
  //@{
  //! Get number of segments
  inline int get_num_segments() const;
  //! Get number of axial positions per segment
  inline int get_num_axial_poss(const int segment_num) const;
  //! Get number of views
  inline int get_num_views() const;
  //! Get number of tangential positions
  inline int get_num_tangential_poss() const;
  //! Get number of tof bins
  inline int get_tof_bin(const double& delta) const;
  //! Get number of TOF bins
  inline int get_num_tof_poss() const;
  //! Get minimum segment number
  inline int get_min_segment_num() const;
  //! Get maximum segment number
  inline int get_max_segment_num() const;
  //! Get minimum axial position per segmnet
  inline int get_min_axial_pos_num(const int segment_num) const;
  //! Get maximum axial position per segment
  inline int get_max_axial_pos_num(const int segment_num) const;
  //! Get minimum view number
  inline int get_min_view_num() const;
  //! Get maximum view number
  inline int get_max_view_num() const;
  //! Get minimum tangential position number
  inline int get_min_tangential_pos_num() const;
  //! Get maximum tangential position number
  inline int get_max_tangential_pos_num() const;
  //! Get number of TOF positions
  inline int get_num_timing_poss() const;
  //! Get TOF mash factor
  inline int get_tof_mash_factor() const;
   //! Get the index of the first timing position
  inline int get_min_timing_pos_num() const;
  //! Get the index of the last timgin position.
  inline int get_max_timing_pos_num() const;
  //! Get the coincide window in pico seconds
  //! \warning Proposed convension: If the scanner is not TOF ready then
  //! the coincidence windowis in the timing bin size.
  inline float get_coincidence_window_in_pico_sec() const;
  //! Get the total width of the coincide window in mm
  inline float get_coincidence_window_width() const;
  //@}

  //| \name Functions that return geometrical info for a Bin
  //@{
  //! Get tangent of the co-polar angle of the normal to the projection plane
  /*! theta=0 for 'direct' planes (i.e. projection planes parallel to the scanner axis) */
  virtual float get_tantheta(const Bin&) const =0;
  
  //! Get cosine of the co-polar angle of the normal to the projection plane
  /*! theta=0 for 'direct' planes (i.e. projection planes parallel to the scanner axis) */
  inline float get_costheta(const Bin&) const;
  
  //! Get azimuthal angle phi of the normal to the projection plane
  /*! phi=0 when the normal vector has no component along the horizontal axis */
  virtual float get_phi(const Bin&) const =0;
  
  //! Get value of the (roughly) axial coordinate in the projection plane (in mm)
  /*! t-axis is defined to be orthogonal to the s-axis (and to the vector
      normal to the projection plane */
  virtual float get_t(const Bin&) const =0;

  //! Return z-coordinate of the middle of the LOR (in mm)
  /*!
    The middle is defined as follows: imagine a cylinder centred around
    the scanner axis. The LOR will intersect the cylinder at 2 opposite
    ends. The middle of the LOR is exactly halfway those 2 points.

    The 0 of the z-axis is chosen in the middle of the scanner.

    Default implementation is equivalent to
    \code
    get_t(bin)/get_costheta(bin)
    \endcode
  */  
  virtual inline float get_m(const Bin&) const;

  //! Get value of the tangential coordinate in the projection plane (in mm)
  /*! s-axis is defined to be orthogonal to the scanner axis (and to the vector
      normal to the projection plane */
  virtual float get_s(const Bin&) const =0;

  //! Get value ot the timing location along the LOR (in mm)
  //! k is a line segment connecting the centers of the two detectors.
  float get_k(const Bin&) const;

  //! Get LOR corresponding to a given bin
  /*!
      \see get_bin()
      \warning This function might get a different type of arguments
      in the next release.
  */
  virtual void
    get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>&,
	    const Bin&) const = 0;
  //@}

  //! \name Functions that return info on the sampling in the different coordinates
  //@{
  //! Get sampling distance in the \c t coordinate
  /*! For some coordinate systems, this might depend on the Bin. The 
      default implementation computes it as 
      \code
      1/2(get_t(..., ax_pos+1,...)-get_t(..., ax_pos-1,...)))
      \endcode
  */
  virtual float get_sampling_in_t(const Bin&) const;

  //! Get sampling distance in the \c m coordinate
  /*! For some coordinate systems, this might depend on the Bin. The 
      default implementation computes it as 
      \code
      1/2(get_m(..., ax_pos+1,...)-get_m(..., ax_pos-1,...)))
      \endcode
  */
  virtual float get_sampling_in_m(const Bin&) const;

  //! Get sampling distance in the \c s coordinate
  /*! For some coordinate systems, this might depend on the Bin. The 
      default implementation computes it as 
      \code
      1/2(get_s(..., tang_pos+1)-get_s(..., tang_pos_pos-1)))
      \endcode
  */
  virtual float get_sampling_in_s(const Bin&) const;

  //! Get sampling distance in the k \c coordinate
  float get_sampling_in_k(const Bin&) const;
  //@}


  //! Find the bin in the projection data that 'contains' an LOR
  /*! Projection data corresponds to lines, so most Lines Of Response 
      (LORs) there is a bin in the projection data. Usually this will be
      the bin which has a central LOR that is 'closest' to the LOR that
      is passed as an argument.

      If there is no such bin (e.g. the LOR does not intersect the
      detectors, Bin::get_bin_value() will be less than 0, otherwise
      it will be 1.

      \warning This function might get a different type of arguments
      in the next release.
      \see get_LOR()
  */
  virtual 
    Bin
    get_bin(const LOR<float>&) const = 0;

  //! \name Equality of ProjDataInfo objects
  //@{
  //! check equality
  bool operator ==(const ProjDataInfo& proj) const; 
  
  bool operator !=(const ProjDataInfo& proj) const; 
  //@}

  //! \name Functions that return sinograms etc (filled with 0)
  //@{

  //! Get empty viewgram
  Viewgram<float> get_empty_viewgram(const int view_num, const int segment_num, 
    const bool make_num_tangential_poss_odd = false) const;
  
  //! Get empty_sinogram
  Sinogram<float> get_empty_sinogram(const int ax_pos_num, const int segment_num,
    const bool make_num_tangential_poss_odd = false) const;

  //! Get empty segment sino
  SegmentByView<float> get_empty_segment_by_view(const int segment_num, 
		  	   const bool make_num_tangential_poss_odd = false) const;
  //! Get empty segment view
  SegmentBySinogram<float> get_empty_segment_by_sinogram(const int segment_num, 
				   const bool make_num_tangential_poss_odd = false) const;


  //! Get empty related viewgrams, where the symmetries_ptr specifies the symmetries to use
  RelatedViewgrams<float> get_empty_related_viewgrams(const ViewSegmentNumbers&,
    const shared_ptr<DataSymmetriesForViewSegmentNumbers>&,
    const bool make_num_tangential_poss_odd = false) const;   
  //@}


  //! Get scanner pointer  
  inline const Scanner* get_scanner_ptr() const;
  
  //! Return a string describing the object
  virtual std::string parameter_info() const;

  //! Struct which holds two floating numbers
  struct Float1Float2 { float low_lim; float high_lim; };

  //! Vector which holds the lower and higher boundary for each timing position, for faster access.
  mutable VectorWithOffset<Float1Float2> timing_bin_boundaries_mm;

  mutable VectorWithOffset<Float1Float2> timing_bin_boundaries_ps;
  
protected:
  virtual bool blindly_equals(const root_type * const) const = 0;

private:
  shared_ptr<Scanner> scanner_ptr;
  int min_view_num;
  int max_view_num;
  int min_tangential_pos_num;
  int max_tangential_pos_num;
  //! Minimum timing pos
  int min_timing_pos_num;
  //! Maximum timing pos
  int max_timing_pos_num;
  //! TOF mash factor.
  int tof_mash_factor;
  //! Finally (with any mashing factor) timing bin increament.
  float timing_increament_in_mm;
  //! Number of tof bins (TOF mash factor applied)
  int num_tof_bins;
  VectorWithOffset<int> min_axial_pos_per_seg; 
  VectorWithOffset<int> max_axial_pos_per_seg;
  
};

END_NAMESPACE_STIR

#include "stir/ProjDataInfo.inl"

#endif //  __ProjDataInfo_H__

