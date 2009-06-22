//
// $Id$
//
/*!

  \file
  \ingroup projdata
  \brief Declaration of class stir::ProjDataInterfile

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_ProjDataInterfile_H__
#define __stir_ProjDataInterfile_H__

#include "stir/ProjDataFromStream.h" 
#include "stir/shared_ptr.h"

#include <iostream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::istream;
using std::ios;
using std::iostream;
using std::streamoff;
using std::string;
#endif

START_NAMESPACE_STIR


/*!
  \ingroup projdata
  \brief A class which reads/writes projection data from/to a (binary) stream, but creates the
  corresponding Interfile header.

  \warning The class can ONLY be used to create a new file. Use ProjData::read_from_file() to
  read a projection data file.
*/
class ProjDataInterfile : public ProjDataFromStream
{
public:
    
  //! constructor taking all necessary parameters
  /*! 
    \param filename The name to use for the files. See below.
    \param segment_sequence_in_stream has to be set according to the order
    in which the segments will occur in the stream. segment_sequence_in_stream[i]
    is the segment number of the i-th segment in the stream.

    \par file names that will be used

    <ul>
    <li> if \a filename has no extension or if \a filename has an extension .hs, 
         the extensions .s and .hs will be used for binary file and header file.
    <li> otherwise, \a filename will be used for the binary data, and its extension
         will be replaced with .hs for the header file.
    </ul>
   
    \warning This call will create a new file for the binary data and the Intefile header.
    Any existing files with the same file anmes will be overwritten without warning.
  */
  ProjDataInterfile (shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
		     const string& filename, const ios::openmode, 
		     const vector<int>& segment_sequence_in_stream,
		     StorageOrder o = Segment_View_AxialPos_TangPos,
		     NumericType data_type = NumericType::FLOAT,
		     ByteOrder byte_order = ByteOrder::native,  
		     float scale_factor = 1 );

  //! as above, but with a default value for segment_sequence_in_stream
  /*! The default value for segment_sequence_in_stream is a vector with
    values min_segment_num, min_segment_num+1, ..., max_segment_num
  */
  ProjDataInterfile (shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
                     const string& filename, 
                     const ios::openmode open_mode = 
#ifndef STIR_NO_NAMESPACES
                     std::ios::out, //necessary for VC bug
#else
                     ios::out,
#endif
		     StorageOrder o = Segment_View_AxialPos_TangPos,
		     NumericType data_type = NumericType::FLOAT,
		     ByteOrder byte_order = ByteOrder::native,  
		     float scale_factor = 1 );
private:
  void create_stream(const string& filename, const ios::openmode open_mode);
};

END_NAMESPACE_STIR


#endif
