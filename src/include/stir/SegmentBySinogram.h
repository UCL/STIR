//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Declaration of class SegmentBySinogram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe  
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#ifndef __SegmentBySinogram_H__
#define __SegmentBySinogram_H__

#include "Segment.h"
#include "Array.h"
#include "Sinogram.h"

START_NAMESPACE_TOMO

//forward declaration for use in convertion 
template <typename elemT> class SegmentByView;


/*!
  \ingroup buildblock
  \brief A class for storing (3d) projection data with a fixed segment_num.

  Storage order is as follows:
  \code
  segment_by_sino[view_num][axial_pos_num][tangential_pos_num]
  \endcode  
*/
template <typename elemT>
class SegmentBySinogram : public Segment<elemT>, public Array<3,elemT>
{
  
public:
  // Constructor that sets the data to a given 3d Array
  inline
  SegmentBySinogram(const Array<3,elemT>(v), 
		    const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
		    const int segment_num);
  
  //! Constructor that sets sizes via the ProjDataInfo object, initialising data to 0
  inline 
  SegmentBySinogram(const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
		    const int segment_num);

  
  //! Conversion from 1 storage order to the other
  SegmentBySinogram (const SegmentByView<elemT>& );
  //! Get storage order 
  inline StorageOrder get_storage_order() const;
  //! Get number of axial positions
  inline int get_num_axial_poss() const;
  //! Get number of views
  inline int get_num_views() const;
  //! Get number of tangetial positions
  inline int get_num_tangential_poss() const;
  //! Get minimum axial position number
  inline int get_min_axial_pos_num() const;
  //! Get maximum axial position number
  inline int get_max_axial_pos_num() const;
  //! Get minimum view number
  inline int get_min_view_num() const;
  //! Get maximum view number
  inline int get_max_view_num() const;
  //! Get minimum tangetial position number
  inline int get_min_tangential_pos_num()  const;
  //! Get maximum tangential position number
  inline int get_max_tangential_pos_num()  const;
  //! Get sinogram
  inline Sinogram<elemT> get_sinogram(int axial_pos_num) const;  
  //! Get viewgram
  Viewgram<elemT> get_viewgram(int view_num) const;
  //! Set viewgram
  void set_viewgram(const Viewgram<elemT>&);
  //! Set sinogram
  inline void set_sinogram(Sinogram<elemT> const &s, int axial_pos_num);  
  inline void set_sinogram(const Sinogram<elemT>& s);

  //! Overloading Array::grow
  void grow(const IndexRange<3>& range);
};

END_NAMESPACE_TOMO

#include "SegmentBySinogram.inl"
#endif
