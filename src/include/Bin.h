//
// $Id$: $Date$
//

/*!
  \file
  \ingroup buildblock

  \brief Declaration of class Bin
  

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Mustapha Sadki
  \author PARAPET project

  \date $Date$
  \version $Revision$
*/
#ifndef __Bin_H__
#define __Bin_H__


#include "Tomography_common.h"


START_NAMESPACE_TOMO
/*!
 \brief
 A class for storing coordinates and value of a single projection bin.

*/

class Bin
{
public: 
  //! default constructor
  inline Bin();

  //!  A constructor : constructs a bin with value (defaulting to 0)
  inline Bin(int segment_num,int view_num, int axial_pos_num,
    int tangential_pos_num,float bin_value=0);
  
  //!get axial position number
  inline int axial_pos_num()const;
  //! get segmnet number
  inline int segment_num()const; 
  //! get tangential position number
  inline int tangential_pos_num()  const; 
  //! get view number
  inline int view_num() const; 
  
  inline int& axial_pos_num(); 
  inline int& segment_num(); 
  inline int& tangential_pos_num(); 
  inline int& view_num(); 
  
  //! get an empty copy
  inline Bin get_empty_copy() const;
  
  //! get the value after forward projection 
  inline float get_bin_value()const; 
  //! set the value to be back projected 
  inline void set_bin_value( float v );
  
  //! accumulate voxel's contribution during forward projection 
  inline Bin&  operator+=(const float dx);
  
  //! comparison operators
  inline bool operator==(const Bin&) const;
  inline bool operator!=(const Bin&) const;
  
private :
  // shared_ptr<ProjDataInfo> proj_data_info_ptr; 
  
  int  segment;
  int  view; 
  int  axial_pos; 
  int  tangential_pos; 
  float bin_value;
  
  
};



END_NAMESPACE_TOMO


#include "Bin.inl"

#endif //__Bin_H__
