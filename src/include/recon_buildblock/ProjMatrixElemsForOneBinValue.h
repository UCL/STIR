//
// $Id$: $Date$
//
/*!
  \file
  \ingroup recon_buildblock
  
  \brief Declaration of class ProjMatrixElemsForOneBinValue
    
  \author Kris Thielemans
  \author Mustapha Sadki
  \author PARAPET project
      
  \date $Date$        
  \version $Revision$
*/

#ifndef __ProjMatrixElemsForOneBinValue_H__
#define __ProjMatrixElemsForOneBinValue_H__


#include "Tomography_common.h"

START_NAMESPACE_TOMO

template <int num_dimensions, typename coordT> class BasicCoordinate;

/*!
  \ingroup recon_buildblock
  \brief Stores voxel coordinates and the value of the matrix element. 
 
  (Probably) only useful in class ProjMatrixElemsForOneBin.

  \warning It is recommended never to use this class name directly, but
  always use the typedef ProjMatrixElemsForOneBin::value_type.

  \warning  Voxel coordinates are currently stored as shorts for saving memory.

 */
class ProjMatrixElemsForOneBinValue
{ 
public:
  explicit inline
    ProjMatrixElemsForOneBinValue(const BasicCoordinate<3,int>& coords,
                                  const float ivalue=0);

  inline ProjMatrixElemsForOneBinValue();


  //! get the coordinates
  inline BasicCoordinate<3,int> get_coords() const;

  //! In effect the same as get_coords()[1] (but faster)
  inline int coord1() const;
  //! In effect the same as get_coords()[2] (but faster)
  inline int coord2() const;
  //! In effect the same as get_coords()[3] (but faster)
  inline int coord3() const;

  //! Get the value of the matrix element
  inline float get_value() const;

  //! Adds el2.get_value() to the value of the current object
  inline ProjMatrixElemsForOneBinValue& operator+=(const ProjMatrixElemsForOneBinValue& el2);
  //! Multiplies the value of with a float
  inline ProjMatrixElemsForOneBinValue& operator*=(const float d);
  //! Adds a float to the value 
  inline ProjMatrixElemsForOneBinValue& operator+=(const float d);
  //! Divides the value of with a float
  inline ProjMatrixElemsForOneBinValue& operator/=(const float d);

  
  //////// comparison functions

  //! Checks if the coordinates are equal
  /*! This function and the next one below are implemented as static members,
      such that you can pass them (as functoon objects) to std::sort.
   */
  static inline bool coordinates_equal(const ProjMatrixElemsForOneBinValue& el1, const ProjMatrixElemsForOneBinValue& el2);

  //! Checks lexicographical order of the coordinates
  static inline bool coordinates_less(const ProjMatrixElemsForOneBinValue& el1, const ProjMatrixElemsForOneBinValue& el2);
   
  //! Checks coordinates and value are equal
  friend inline bool operator==(const ProjMatrixElemsForOneBinValue& el1, const ProjMatrixElemsForOneBinValue& el2);
  
  //! Checks lexicographical order of the coordinates and the value
  friend inline bool operator<(const ProjMatrixElemsForOneBinValue& el1, const ProjMatrixElemsForOneBinValue& el2);
 
private:
  short c3,c2,c1; 
  float value;
  
};


END_NAMESPACE_TOMO

#include "recon_buildblock/ProjMatrixElemsForOneBinValue.inl"

#endif // __ProjMatrixElemsForOneBinValue_H__
