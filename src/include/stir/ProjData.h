/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2015-2017, 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata
  \brief Declaration of class stir::ProjData

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project
*/
#ifndef __stir_ProjData_H__
#define __stir_ProjData_H__

#include "stir/Array.h"
#include "stir/shared_ptr.h"
#include "stir/ProjDataInfo.h"
#include <string>
#include <iostream>
#include "stir/Succeeded.h"
#include "stir/SegmentBySinogram.h"
#include "stir/SegmentByView.h"
//#include <ios>

#include "stir/ExamData.h"

START_NAMESPACE_STIR


template <typename elemT> class RelatedViewgrams;
class DataSymmetriesForViewSegmentNumbers;
template <typename elemT> class SegmentBySinogram;
template <typename elemT> class SegmentByView;
template <typename elemT> class Viewgram;
template <typename elemT> class Sinogram;
class ViewSegmentNumbers;
class Succeeded;
class ProjDataInMemory;
//class ExamInfo;

/*!
  \ingroup projdata
  \brief The (abstract) base class for the projection data.

  Projection data are supposed to be indexed by 4 coordinates
  (corresponding to the most general case of projection data in
  all directions of a 3D volume):
  <ul>
  <li> \c segment_num : indexes polar angle theta, or ring difference
          (segment_num==0 are the projections orthogonal to the scanner axis)
  <li> \c view_num : indexes azimuthal angle phi
  <li> \c axial_pos_num : indexes different positions along the scanner axis
          (corresponding to 'z', or different rings)
  <li> \c tangential_pos_num : indexes different positions in a direction 
        tangential to the scanner cylinder.
        (sometimes called 'bin' or 'element')
  </ul>

  The number of axial positions is allowed to depend on segment_num.

  Different 'subsets' of the 4D data are possible by fixing one or
  more of the 4 coordinates. Currently we support the following cases
  <ul>
  <li> SegmentBySinogram (fixed segment_num)
  <li> SegmentByView (fixed segment_num)
  <li> Viewgram (fixed segment_num and view_num)
  <li> Sinogram (fixed segment_num and axial_pos_num)
  <li> RelatedViewgrams (different Viewgrams related by symmetry)
  </ul>

  This abstract class provides the general interface for accessing the
  projection data. This works with get_ and set_ pairs. (Generally, 
  the 4D dataset might be too big to be kept in memory.) In addition, there
  are get_empty_ functions that just create the corresponding object
  of appropriate sizes etc. but filled with 0.

  One important member of this class is get_proj_data_info_sptr() which
  allows access to a ProjDataInfo object, which describes the dimensions
  of the data, the scanner type, the geometry ...

  \warning The arguments 'make_num_tangential_poss_odd' are temporary
  and will be deleted in the next release.
*/
class ProjData : public ExamData
{
public:

   //! A static member to get the projection data from a file
  static shared_ptr<ProjData> 
    read_from_file(const std::string& filename,
		   const std::ios::openmode open_mode = std::ios::in);

  //! Empty constructor 
  ProjData();
  //! construct by specifying info. Data will be undefined.
  ProjData(const shared_ptr<const ExamInfo>& exam_info_sptr,
           const shared_ptr<const ProjDataInfo>& proj_data_info_ptr);
#if 0
  // it would be nice to have something like this. However, it's implementation
  // normally fails as we'd need to use set_viewgram or so, which is virtual, but
  // this doesn't work inside a constructor
  ProjData(const ProjData&);
#endif

  //! Destructor
  virtual ~ProjData() {}
  //! Get shared pointer to proj data info
  inline shared_ptr<const ProjDataInfo>
    get_proj_data_info_sptr() const;
  //! Get viewgram
  virtual Viewgram<float> 
    get_viewgram(const int view, const int segment_num,const bool make_num_tangential_poss_odd = false) const=0;
  //! Set viewgram
  virtual Succeeded 
    set_viewgram(const Viewgram<float>&) = 0;
  //! Get sinogram
  virtual Sinogram<float> 
    get_sinogram(const int ax_pos_num, const int segment_num,const bool make_num_tangential_poss_odd = false) const=0;
  //! Set sinogram
  virtual Succeeded 
    set_sinogram(const Sinogram<float>&) = 0;

  //! construct projection data that stores a subset of the views
  unique_ptr<ProjDataInMemory>
    get_subset(const std::vector<int>& views) const;

  //! Get empty viewgram
  Viewgram<float> get_empty_viewgram(const int view, const int segment_num, 
    const bool make_num_tangential_poss_odd = false) const;
  
  //! Get empty_sinogram
  Sinogram<float> 
    get_empty_sinogram(const int ax_pos_num, const int segment_num,
    const bool make_num_tangential_poss_odd = false) const;

   //! Get empty segment sino
  SegmentByView<float> 
    get_empty_segment_by_view(const int segment_num, 
		  	   const bool make_num_tangential_poss_odd = false) const;
  //! Get empty segment view
  SegmentBySinogram<float> 
    get_empty_segment_by_sinogram(const int segment_num, 
				   const bool make_num_tangential_poss_odd = false) const;


  //! Get segment by sinogram
  virtual SegmentBySinogram<float>
    get_segment_by_sinogram(const int segment_num) const;
  //! Get segment by view
  virtual SegmentByView<float> 
    get_segment_by_view(const int segment_num) const;
  //! Set segment by sinogram
  virtual Succeeded 
    set_segment(const SegmentBySinogram<float>&);
  //! Set segment by view 
  virtual Succeeded 
    set_segment(const SegmentByView<float>&);

  //! Get related viewgrams
  virtual RelatedViewgrams<float> 
    get_related_viewgrams(const ViewSegmentNumbers&,
    const shared_ptr<DataSymmetriesForViewSegmentNumbers>&,
    const bool make_num_tangential_poss_odd = false) const;
  //! Set related viewgrams
  virtual Succeeded set_related_viewgrams(const RelatedViewgrams<float>& viewgrams);
  

  //! Get empty related viewgrams, where the symmetries_ptr specifies the symmetries to use
  RelatedViewgrams<float> 
    get_empty_related_viewgrams(const ViewSegmentNumbers& view_segmnet_num,
    //const int view_num, const int segment_num, 
    const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_ptr,
    const bool make_num_tangential_poss_odd = false) const;   


  //! set all bins to the same value
  /*! will call error() if setting failed */
  virtual void fill(const float value);

  //! set all bins from another ProjData object
  /*! will call error() if setting failed or if the 'source' proj_data is not compatible.
    The current check requires at least the same segment numbers (but the source can have more),
    all other geometric parameters have to be the same.
 */
  virtual void fill(const ProjData&);

  //! Return a vector with segment numbers in a standard order
  /*! This returns a vector filled as \f$ [0, 1, -1, 2, -2, ...] \f$.
    In the (unlikely!) case that the segment range is not symmetric,
    the sequence just continues with
    <i>valid</i> segment numbers, e.g. \f$ [0, 1, -1, 2, 3 ] \f$.
   */
  static
    std::vector<int>
    standard_segment_sequence(const ProjDataInfo& pdi);

  //! set all bins from an array iterator
  /*!
    \return \a array_iter advanced over the number of bins (as \c std::copy)
  
    Data are filled by `SegmentBySinogram`, with segment order given by
    standard_segment_sequence().

    \warning there is no range-check on \a array_iter
  */
  template < typename iterT>
  iterT fill_from( iterT array_iter)
  {
      // A type check would be useful.
      //      BOOST_STATIC_ASSERT((boost::is_same<typename std::iterator_traits<iterT>::value_type, Type>::value));

      for (int s=0; s<= this->get_max_segment_num(); ++s)
      {
          SegmentBySinogram<float> segment = this->get_empty_segment_by_sinogram(s);
          // cannot use std::copy sadly as needs end-iterator for range
          for (SegmentBySinogram<float>::full_iterator seg_iter = segment.begin_all();
               seg_iter != segment.end_all();
               /*empty*/)
              *seg_iter++ = *array_iter++;
          this->set_segment(segment);

          if (s!=0)
          {
              segment = this->get_empty_segment_by_sinogram(-s);
              for (SegmentBySinogram<float>::full_iterator seg_iter = segment.begin_all();
                   seg_iter != segment.end_all();
                   /*empty*/)
                  *seg_iter++ = *array_iter++;
              this->set_segment(segment);
          }
      }
      return array_iter;
  }

  //! Copy all bins to a range specified by a (forward) iterator
  /*! 
    \return \a array_iter advanced over the number of bins (as \c std::copy)

    Data are filled by `SegmentBySinogram`, with segment order given by
    standard_segment_sequence().

    \warning there is no range-check on \a array_iter
  */
  template < typename iterT>
  iterT copy_to(iterT array_iter) const
  {
      for (int s=0; s<= this->get_max_segment_num(); ++s)
      {
          SegmentBySinogram<float> segment= this->get_segment_by_sinogram(s);
          array_iter = std::copy(segment.begin_all_const(), segment.end_all_const(), array_iter);
          if (s!=0)
          {
              segment=this->get_segment_by_sinogram(-s);
              array_iter = std::copy(segment.begin_all_const(), segment.end_all_const(), array_iter);
          }
      }
      return array_iter;
  }

  //! Get number of segments
  inline int get_num_segments() const;
  //! Get number of axial positions per segment
  inline int get_num_axial_poss(const int segment_num) const;
  //! Get number of views
  inline int get_num_views() const;
  //! Get number of tangential positions
  inline int get_num_tangential_poss() const;
  //! Get number of TOF positions
  inline int get_num_tof_poss() const;
  //! Get minimum segment number
  inline int get_min_segment_num() const;
  //! Get maximum segment number
  inline int get_max_segment_num() const;
  //! Get mininum axial position per segmnet
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
  //! Get the total number of sinograms
  inline int get_num_sinograms() const;
  //! Get the number of non-tof sinograms
  /*! Note that this is the sum of the number of axial poss over all segments.
      \see get_num_sinograms()
  */
  inline int get_num_non_tof_sinograms() const;
  //! Get the total size of the data
  inline std::size_t size_all() const;
  //! forward ProjDataInfo::get_original_view_nums()
  inline std::vector<int> get_original_view_nums() const;

  //! writes data to a file in Interfile format
  Succeeded write_to_file(const std::string& filename) const;

  //! \deprecated a*x+b*y (\see xapyb)
  STIR_DEPRECATED virtual void axpby(const float a, const ProjData& x,
                                     const float b, const ProjData& y);

  //! set values of the array to x*a+y*b, where a and b are scalar, and x and y are ProjData
  virtual void xapyb(const ProjData& x, const float a,
                     const ProjData& y, const float b);

  //! set values of the array to x*a+y*b, where a, b, x and y are ProjData
  virtual void xapyb(const ProjData& x, const ProjData& a,
                     const ProjData& y, const ProjData& b);

  //! set values of the array to self*a+y*b where a and b are scalar, y is ProjData
  virtual void sapyb(const float a, const ProjData& y, const float b);

  //! set values of the array to self*a+y*b where a, b and y are ProjData
  virtual void sapyb(const ProjData& a, const ProjData& y, const ProjData& b);

protected:

   shared_ptr<const ProjDataInfo> proj_data_info_sptr;
};


END_NAMESPACE_STIR

#include "stir/ProjData.inl"
#endif

