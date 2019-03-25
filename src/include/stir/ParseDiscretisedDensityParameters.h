/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd 
    Copyright (C) 2019, University College London
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
  \ingroup densitydata 
  
  \brief  Definition of the stir::ParseDiscretisedDensityParameters class
    
  \author Kris Thielemans
  \author Matthew Jacobson
  \author Claire Labbe
  \author PARAPET project
      
*/

#ifndef __ParseDiscretisedDensityParameters_H__
#define __ParseDiscretisedDensityParameters_H__

#include "stir/common.h"

START_NAMESPACE_STIR

class KeyParser;

/*!
 \ingroup densitydata 
  
 \brief Class for adding parameters relevant to DiscretisedDensity to a parser

 This class is not very safe. It is only used by ParseAndCreateFrom specialisations.
*/
class ParseDiscretisedDensityParameters
{
 public:
  void
    set_defaults();
  void
    add_to_keymap(KeyParser& parser);

  //! calls error() if something is wrong
  void
    check_values() const;

 protected:
 
  //! the output image size in x and y direction
  /*! convention: if -1, use a size such that the whole FOV is covered
  */
  int output_image_size_xy;

  //! the output image size in z direction
  /*! convention: if -1, use default as provided by VoxelsOnCartesianGrid constructor
  */
  int output_image_size_z; 

  //! the zoom factor
  double zoom;

  //! offset in the x-direction
  double Xoffset;

  //! offset in the y-direction
  double Yoffset;

  //! offset in the z-direction
  double Zoffset;

};

END_NAMESPACE_STIR

#endif
