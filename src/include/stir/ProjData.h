/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2015, University College London
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
#include "stir/SegmentBySinogram.h"
#include "stir/SegmentByView.h"
#include "stir/Succeeded.h"
//#include <ios>

START_NAMESPACE_STIR


template <typename elemT> class RelatedViewgrams;
class DataSymmetriesForViewSegmentNumbers;
template <typename elemT> class SegmentBySinogram;
template <typename elemT> class SegmentByView;
template <typename elemT> class Viewgram;
template <typename elemT> class Sinogram;
class ViewSegmentNumbers;
class Succeeded;
class ExamInfo;

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

  One important member of this class is get_proj_data_info_ptr() which
  allows access to a ProjDataInfo object, which describes the dimensions
  of the data, the scanner type, the geometry ...

  \warning The arguments 'make_num_tangential_poss_odd' are temporary
  and will be deleted in the next release.
*/
class ProjData
{
public:
  //! A static member to get the projection data from a file
  static shared_ptr<ProjData>
    read_from_file(const std::string& filename,
           const std::ios::openmode open_mode = std::ios::in);

  //! Empty constructor
  ProjData();
  //! construct by specifying info. Data will be undefined.
  ProjData(const shared_ptr<ExamInfo>& exam_info_sptr,
           const shared_ptr<ProjDataInfo>& proj_data_info_ptr);
#if 0
  // it would be nice to have something like this. However, it's implementation
  // normally fails as we'd need to use set_viewgram or so, which is virtual, but
  // this doesn't work inside a constructor
  ProjData(const ProjData&);
#endif

  //! Destructor
  virtual ~ProjData() {}
  //! Get proj data info pointer
  inline const ProjDataInfo*
    get_proj_data_info_ptr() const;
  //! Get shared pointer to proj data info
  /*! \warning Use with care. If you modify the object in a shared ptr, everything using the same
    shared pointer will be affected. */
  inline shared_ptr<ProjDataInfo>
    get_proj_data_info_sptr() const;
  //! Get pointer to exam info
  inline const ExamInfo*
    get_exam_info_ptr() const;
  //! Get shared pointer to exam info
  /*! \warning Use with care. If you modify the object in a shared ptr, everything using the same
    shared pointer will be affected. */
  inline shared_ptr<ExamInfo>
    get_exam_info_sptr() const;
  //! change exam info
  /*! This will allocate a new ExamInfo object and copy the data in there. */
  void
    set_exam_info(ExamInfo const&);
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
  void fill(const float value);

  //! set all bins from another ProjData object
  /*! will call error() if setting failed or if the 'source' proj_data is not compatible.
    The current check requires at least the same segment numbers (but the source can have more),
    all other geometric parameters have to be the same.
 */
  void fill(const ProjData&);

  //! set all bins from an array iterator
  template < typename iterT>
  long int fill_from( iterT array_iter)
  {
      // A type check would be usefull.
//      BOOST_STATIC_ASSERT((boost::is_same<typename std::iterator_traits<iterT>::value_type, Type>::value));

      iterT init_pos = array_iter;
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

      return std::distance(init_pos, array_iter);
  }

  //! Copy all bins to an array, using iterator
  template < typename iterT>
  long int copy_to(iterT array_iter) const
  {
      iterT init_pos = array_iter;
      for (int s=0; s<= this->get_max_segment_num(); ++s)
      {
          SegmentBySinogram<float> segment= this->get_segment_by_sinogram(s);
          std::copy(segment.begin_all_const(), segment.end_all_const(), array_iter);
          std::advance(array_iter, segment.size_all());
          if (s!=0)
          {
              segment=this->get_segment_by_sinogram(-s);
              std::copy(segment.begin_all_const(), segment.end_all_const(), array_iter);
              std::advance(array_iter, segment.size_all());
          }
      }

      return std::distance(init_pos, array_iter);
  }

  //! Get number of segments
  inline int get_num_segments() const;
  //! Get number of axial positions per segment
  inline int get_num_axial_poss(const int segment_num) const;
  //! Get number of views
  inline int get_num_views() const;
  //! Get number of tangential positions
  inline int get_num_tangential_poss() const;
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
  //! Get the number of sinograms
  inline size_t get_num_sinograms() const;
  //! Get the total size of the data
  inline size_t size_all() const;

protected:
   shared_ptr<ExamInfo> exam_info_sptr;
   shared_ptr<ProjDataInfo> proj_data_info_ptr; // TODO fix name to _sptr
};


END_NAMESPACE_STIR

#include "stir/ProjData.inl"
#endif

