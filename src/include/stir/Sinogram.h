//
// $Id$
//
/*!

  \file
  \ingroup projdata

  \brief Declaration of class Sinogram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __Sinogram_h__
#define __Sinogram_h__


#include "stir/Array.h"
#include "stir/ProjDataInfo.h" 
#include "stir/shared_ptr.h"




START_NAMESPACE_STIR

class PMessage;

/*!
  \ingroup projdata
  \brief A class for 2d projection data.

  This represents a subset of the full projection. segment_num and axial_pos_num 
  are fixed.
  
*/
template <typename elemT>
class Sinogram : public Array<2,elemT>
{
  
public:
  //! Construct sinogram from proj_data_info pointer, ring and segment number.  Data are set to 0.
  inline Sinogram(const shared_ptr<ProjDataInfo>& proj_data_info_ptr, 
                  const int ax_pos_num, const int segment_num); 

  //! Construct sinogram with data set to the array.
  inline Sinogram(const Array<2,elemT>& p,const shared_ptr<ProjDataInfo >& proj_data_info_ptr, 
                  const int ax_pos_num, const int segment_num); 
  

  Sinogram(PMessage& msg);

  //! Get segment number
  inline int get_segment_num() const; 
  //! Get number of axial positions
  inline int get_axial_pos_num() const;
  //! Get minimum view number
  inline int get_min_view_num() const;
  //! Get maximum view number
  inline int get_max_view_num() const;
  //! Get number of views
  inline int get_num_views() const;
  //! Get minimum number of tangetial positions
  inline int get_min_tangential_pos_num() const;
  //! Get maximum number of tangential positions
  inline int get_max_tangential_pos_num() const;
  //! Get number of tangential positions
  inline int get_num_tangential_poss() const;
  
  //! Get an empty sinogram of the same dimensions, segment_num etc.
  inline Sinogram get_empty_copy(void) const;
 
  //! Overloading Array::grow
  void grow(const IndexRange<2>& range);
  //! Overloading Array::resize
  void resize(const IndexRange<2>& range);

  //! Get the projection data info pointer
  /*! \warning Do not use this pointer after the Sinogram object is destructed.
  */
  inline const ProjDataInfo* get_proj_data_info_ptr() const;

  //inline Sinogram operator = (const Sinogram &s) const;
  
private:
  
  shared_ptr<ProjDataInfo> proj_data_info_ptr; 
  int axial_pos_num;
  int segment_num;
    
};

END_NAMESPACE_STIR

#include "stir/Sinogram.inl"

#endif
