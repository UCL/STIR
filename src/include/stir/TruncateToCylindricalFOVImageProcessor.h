//
//
/*
    Copyright (C) 2005- 2007, Hammersmith Imanet Ltd
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

  void set_strictly_less_than_radius(const bool arg) {
	  this->_strictly_less_than_radius = arg;
  }
  bool get_strictly_less_than_radius() const {
	  return this->_strictly_less_than_radius;
  }

private:
  bool _strictly_less_than_radius;

  virtual void set_defaults();
  virtual void initialise_keymap();
  
  Succeeded virtual_set_up(const DiscretisedDensity<3,elemT>& image);
  // new
  void  virtual_apply(DiscretisedDensity<3,elemT>& out_density, const DiscretisedDensity<3,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<3,elemT>& density) const ;
  
};


END_NAMESPACE_STIR

#endif


