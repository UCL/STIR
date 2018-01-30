#ifndef __stir_IO_ITKInputFileFormat_h__
#define __stir_IO_ITKInputFileFormat_h__
/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2014, University College London
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
  \ingroup IO
  \brief Declaration of class stir::ITKInputFileFormat

  \author Berta Marti Fuster
  \author Kris Thielemans
*/
#include "stir/IO/InputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"

START_NAMESPACE_STIR

//! Class for reading images using ITK.
/*! \ingroup IO

    ITK (http://www.itk.org) has its own registry of file formats, so the current class
    provides an interface to that code. We use ITK for reading, and then translate the ITK
    data and meta-info to STIR. 

    ITK can read many file formats, see http://www.itk.org/Wiki/ITK/File_Formats for some info.

    This STIR class has special handling for DICOM images. For many modalities, DICOM stores
    each slice in a different file. Normally, ITK reads only a single DICOM file, and hence a single slice.
    As this is not useful for STIR, we use \c itk::GDCMSeriesFileNames to find
    other slices belonging to the same series/time frame/gate as the input filename to read_from_file().
    
    \warning This translation currently ignores orientation and direction (e.g. of slice order).
*/
class ITKImageInputFileFormat :
public InputFileFormat<DiscretisedDensity<3,float> >
{

 public:

  //! This function always returns \c false as ITK cannot read from istream
  virtual bool
    can_read(const FileSignature& signature,
	std::istream& input) const;
  //! Use ITK reader to check if it can read the file (by seeing if it throws an exception or not)
  virtual bool 
    can_read(const FileSignature& signature,
	     const std::string& filename) const;
 
  virtual const std::string
    get_name() const
  {  return "ITK"; }

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const;

  //! This function always calls error() as ITK cannot read from istream
  virtual unique_ptr<data_type>
    read_from_file(std::istream& input) const;

  //! This function uses ITK for reading and does the translation to STIR
  virtual unique_ptr<data_type>
    read_from_file(const std::string& filename) const;

};
END_NAMESPACE_STIR

#endif
