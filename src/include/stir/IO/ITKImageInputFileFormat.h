#ifndef __stir_IO_ITKInputFileFormat_h__
#define __stir_IO_ITKInputFileFormat_h__
/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
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

  $Date$
  $Revision$
*/
#include "stir/IO/InputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"

START_NAMESPACE_STIR

//! Class for reading images in ITK file-format.
/*! \ingroup IO
    \preliminary

*/
class ITKImageInputFileFormat :
public InputFileFormat<DiscretisedDensity<3,float> >
{

 public:

	   virtual bool
    can_read(const FileSignature& signature,
	std::istream& input) const;
	   virtual bool 
    can_read(const FileSignature& signature,
	     const std::string& filename) const;
 

 template<typename TImageType> VoxelsOnCartesianGrid<float>* 
	  read_file_itk(std::string filename) const;

  virtual const std::string
    get_name() const
  {  return "ITK"; }

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const;

  virtual std::auto_ptr<data_type>
    read_from_file(std::istream& input) const;

  virtual std::auto_ptr<data_type>
    read_from_file(const std::string& filename) const;

};
END_NAMESPACE_STIR

#endif
