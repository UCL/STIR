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
  
  \brief  Implementation of the stir::ParseDiscretisedDensityParameters class
    
  \author Kris Thielemans
  \author Matthew Jacobson
  \author Claire Labbe
  \author PARAPET project
      
*/
#include "stir/KeyParser.h"
#include "stir/ParseDiscretisedDensityParameters.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/error.h"

START_NAMESPACE_STIR

void 
ParseDiscretisedDensityParameters::
set_defaults()
{
  //base_type::set_defaults();
  output_image_size_xy=-1;
  output_image_size_z=-1;
  zoom_xy=1.F;
  zoom_z=1.F;
  offset.fill(0.F);
}

void
ParseDiscretisedDensityParameters::
add_to_keymap(KeyParser& parser)
{
  //base_type::initialise_keymap();
  parser.add_key("zoom", &zoom_xy);
  parser.add_key("Z zoom", &zoom_z);
  parser.add_key("XY output image size (in pixels)",&output_image_size_xy);
  parser.add_key("Z output image size (in pixels)",&output_image_size_z);
  //parser.add_key("X offset (in mm)", &offset.x()); // KT 10122001 added spaces
  //parser.add_key("Y offset (in mm)", &offset.y());
  parser.add_key("Z offset (in mm)", &offset.z());
}

#if 0
// disable ask_parameters
void ParseDiscretisedDensityParameters::
ask_parameters()
{

  zoom=  ask_num("Specify a zoom factor as magnification effect ? ",0.1,10.,1.);


  output_image_size_xy =  
    ask_num("Final image size (-1 for default)? ",
	    -1,
	    4*static_cast<int>(proj_data_ptr->get_num_tangential_poss()*zoom),
	    -1);
    
#if 0    
    // This section enables you to position a reconstructed image
    // along x (horizontal), y (vertical) and/or z (transverse) axes
    // The default values is in the center of the FOV,
    // the positve direction is
    // for x-axis, toward the patient's left side (assuming typical spinal, head first position)
    // for y-axis, toward the top of the FOV
    // for z-axis, toward the patient's feet (assuming typical spinal, head first position)
    
    cout << endl << "    Enter offset  Xoff, Yoff (in pixels) :";
    Xoffset = ask_num("   X offset  ",-old_size/2, old_size/2, 0);
    Yoffset = ask_num("   Y offset  ",-old_size/2, old_size/2, 0);
#endif

}
#endif // ask_parameters disabled


void
ParseDiscretisedDensityParameters::
check_values() const
{
  if (zoom_xy <= 0)
  { error("zoom should be positive"); }
  if (zoom_z <= 0)
  { error("z zoom should be positive"); }
  
  if (output_image_size_xy!=-1 && output_image_size_xy<1) // KT 10122001 appended_xy
  { error("output image size xy must be positive (or -1 as default)"); }
  if (output_image_size_z!=-1 && output_image_size_z<1) // KT 10122001 new
  { error("output image size z must be positive (or -1 as default)"); }
}

int
ParseDiscretisedDensityParameters::
get_output_image_size_xy() const
{ return this->output_image_size_xy; }

void
ParseDiscretisedDensityParameters::
set_output_image_size_xy(int v)
{ this->output_image_size_xy = v; }

int
ParseDiscretisedDensityParameters::
get_output_image_size_z() const
{ return this->output_image_size_z; }

void
ParseDiscretisedDensityParameters::
set_output_image_size_z(int v)
{ this->output_image_size_z = v; }

float
ParseDiscretisedDensityParameters::
get_zoom_xy() const
{ return this->zoom_xy; }

void
ParseDiscretisedDensityParameters::
set_zoom_xy(float v)
{ this->zoom_xy = v; }

float
ParseDiscretisedDensityParameters::
get_zoom_z() const
{ return this->zoom_z; }

void
ParseDiscretisedDensityParameters::
set_zoom_z(float v)
{ this->zoom_z = v; }

const CartesianCoordinate3D<float>&
ParseDiscretisedDensityParameters::
get_offset() const
{ return this->offset; }

void
ParseDiscretisedDensityParameters::
set_offset(const CartesianCoordinate3D<float>& v)
{ this->offset = v; }

END_NAMESPACE_STIR
