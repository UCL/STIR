//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief 

  \brief Declaration of class SegmentByView

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe  
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#ifndef __SegmentByView_H__
#define __SegmentByView_H__


#include "Segment.h"
#include "Array.h"
#include "Viewgram.h"

START_NAMESPACE_TOMO

template <typename elemT> class SegmentBySinogram;
template <typename elemT> class Sinogram;

/*!
  \ingroup buildblock
  \brief A class for storing (3d) projection data with a fixed segment_num.

  Storage order is as follows:
  \code
  segment_by_view[axial_pos_num][view_num][tangential_pos_num]
  \endcode  
*/

template <typename elemT> class SegmentByView : public Segment<elemT>, public Array<3,elemT>
{
public:
  // Constructor that sets the data to a given 3d Array
  inline 
    SegmentByView(const Array<3,elemT>& v,  
	          const shared_ptr<ProjDataInfo>& proj_data_info_ptr, 
		  const int segment_num);

  //! Constructor that sets sizes via the ProjDataInfo object, initialising data to 0
  inline
    SegmentByView(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
                  const int segment_num);

  
  //! Conversion from 1 storage order to the other
  SegmentByView(const SegmentBySinogram<elemT>& );
  
  //TODO ? how to declare a conversion routine that works for any Segment ?
  //! Get storage order
  inline StorageOrder get_storage_order() const;
  //! Get view number 
  inline int get_num_views() const;
  //! Get number of axial positions
  inline int get_num_axial_poss() const;
  //! Get number of tangetial positions
  inline int get_num_tangential_poss()  const;
  //! Get minimum view number
  inline int get_min_view_num() const;
  //! Get maximum view number
  inline int get_max_view_num() const;
  //! Get minimum axial position number
  inline int get_min_axial_pos_num() const;
  //! Get maximum axial positin number
  inline int get_max_axial_pos_num() const;
  //! Get minimum tangential position
  inline int get_min_tangential_pos_num() const;
  //! Get maximum tangetial position number
  inline int get_max_tangential_pos_num() const;
  
  //! Get sinogram
  Sinogram<elemT> get_sinogram(int axial_pos_num) const;
  //! Get viewgram
  inline Viewgram<elemT> get_viewgram(int view_num) const;
  //! Set sinogram
  inline void set_sinogram(const Sinogram<elemT> &s);
  //! Set sinogram
  void set_sinogram(Sinogram<elemT> const &s, int axial_pos_num);
  //! Set viewgram
  inline void set_viewgram(const Viewgram<elemT> &v);

  //! Overloading Array::grow
  void grow(const IndexRange<3>& range);

};

END_NAMESPACE_TOMO

#include "SegmentByView.inl"

#endif
