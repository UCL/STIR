/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2018, 2021-2022 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::DiscretisedDensity hierarchy

  \author Kris Thielemans
  \author Robert Twyman
*/

%shared_ptr(stir::DiscretisedDensity<3,float>);
%shared_ptr(stir::DiscretisedDensityOnCartesianGrid<3,float>);
%shared_ptr(stir::VoxelsOnCartesianGrid<float>);

// ignore this one and add it later (see below)
%ignore stir::DiscretisedDensity::read_from_file(const std::string& filename);
%include "stir/DiscretisedDensity.h"
%include "stir/DiscretisedDensityOnCartesianGrid.h"

%include "stir/VoxelsOnCartesianGrid.h"

%extend stir::VoxelsOnCartesianGrid {
  // add read_from_file to this class, as currently there is no way
  // to convert the swigged DiscretisedDensity to a VoxelsOnCartesianGrid
  static stir::VoxelsOnCartesianGrid<elemT> * read_from_file(const std::string& filename)
    {
      using namespace stir;
      unique_ptr<DiscretisedDensity<3,elemT> > 
	ret(read_from_file<DiscretisedDensity<3,elemT> >(filename));
      return dynamic_cast<VoxelsOnCartesianGrid<elemT> *>(ret.release());
    }

    // add write_to_file method for VoxelsOnCartesianGrid, returns the saved filename
    std::string write_to_file(const std::string& filename)
    {
      return write_to_file(filename, *$self);
    }
 }

%template(Float3DDiscretisedDensity) stir::DiscretisedDensity<3,float>;
%template(Float3DDiscretisedDensityOnCartesianGrid) stir::DiscretisedDensityOnCartesianGrid<3,float>;
//%template() stir::DiscretisedDensity<3,float>;
//%template() stir::DiscretisedDensityOnCartesianGrid<3,float>;
%template(FloatVoxelsOnCartesianGrid) stir::VoxelsOnCartesianGrid<float>;

%include "stir/IO/write_to_file.h"
%template(write_image_to_file) stir::write_to_file<DiscretisedDensity<3, float> >;
