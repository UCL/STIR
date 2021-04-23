/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd 
    Copyright (C) 2019, University College London
    This file is part of STIR. 
 
    SPDX-License-Identifier: Apache-2.0  AND License-ref-PARAPET-license
 
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

#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

class KeyParser;

/*!
 \ingroup densitydata 
  
 \brief Class for adding parameters relevant to DiscretisedDensity to a parser

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

   
  //! @name the output image size in x and y direction
  /*! convention: if -1, use a size such that the whole FOV is covered
  */
  //@{
  int get_output_image_size_xy() const;
  void set_output_image_size_xy(int);
  //@}

  //! @name the output image size in z direction
  /*! convention: if -1, use default as provided by VoxelsOnCartesianGrid constructor
  */
  //@{
  int get_output_image_size_z() const;
  void set_output_image_size_z(int);
  //@}

  //! @name the xy-zoom factor
  /*! See the VoxelsOnCartesianGrid constructor */
  //@{
  float get_zoom_xy() const;
  void set_zoom_xy(float);
  //@}

  //! @name the zoom factor in z-direction
  //@{
  float get_zoom_z() const;
  void set_zoom_z(float);
  //@}

  //! @name offset from default position
  //@{
  const CartesianCoordinate3D<float>& get_offset() const;
  void set_offset(const CartesianCoordinate3D<float>&);
  //@}

 private:
 
  //! the output image size in x and y direction
  /*! convention: if -1, use a size such that the whole FOV is covered
  */
  int output_image_size_xy;

  //! the output image size in z direction
  /*! convention: if -1, use default as provided by VoxelsOnCartesianGrid constructor
  */
  int output_image_size_z; 

  //! the zoom factor in xy-direction
  float zoom_xy;

  //! the zoom factor in z-direction
  float zoom_z;

  //! offset
  CartesianCoordinate3D<float> offset;

};

END_NAMESPACE_STIR

#endif
