//
//
/*
    Copyright (C) 2005- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup ImageProcessor  
  \brief Declaration of class stir::TruncateToCylindricalFOVImageProcessor
    
  \author Kris Thielemans
      
*/

#ifndef __stir_TruncateToCylindricalFOVImageProcessor_H__
#define __stir_TruncateToCylindricalFOVImageProcessor_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"


START_NAMESPACE_STIR


/*!
  \ingroup ImageProcessor  
  \brief A class in the DataProcessor hierarchy that sets voxels to 0
  outside a given radius.
  
  As it is derived from RegisteredParsingObject, it implements all the 
  necessary things to parse parameter files etc.

  The discretised densities that will be filtered are supposed to be on a 
  Cartesian grid. 
 */

template <typename elemT>
class TruncateToCylindricalFOVImageProcessor : 
  public 
    RegisteredParsingObject<
        TruncateToCylindricalFOVImageProcessor<elemT>,
        DataProcessor<DiscretisedDensity<3,elemT> >,
        DataProcessor<DiscretisedDensity<3,elemT> >
    >
{
 private:
  typedef
    RegisteredParsingObject<
              TruncateToCylindricalFOVImageProcessor<elemT>,
              DataProcessor<DiscretisedDensity<3,elemT> >,
              DataProcessor<DiscretisedDensity<3,elemT> >
	       >
    base_type;
public:
  static const char * const registered_name; 
  
  //! Default constructor
  TruncateToCylindricalFOVImageProcessor();

  void set_truncate_rim(const int truncate_rim) {
    this->_truncate_rim = truncate_rim;
  }

  int get_truncate_rim() {
    return this->_truncate_rim;
  }

  void set_strictly_less_than_radius(const bool arg) {
	  this->_strictly_less_than_radius = arg;
  }
  bool get_strictly_less_than_radius() const {
	  return this->_strictly_less_than_radius;
  }

private:
  bool _strictly_less_than_radius;
  int _truncate_rim;

  virtual void set_defaults();
  virtual void initialise_keymap();
  
  Succeeded virtual_set_up(const DiscretisedDensity<3,elemT>& image);
  // new
  void  virtual_apply(DiscretisedDensity<3,elemT>& out_density, const DiscretisedDensity<3,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<3,elemT>& density) const ;
  
};


END_NAMESPACE_STIR

#endif


