/*!

  \file
  \ingroup projdata
  \brief Implementation of class stir::ProjDataInterfile

  \author Kris Thielemans
*/
/*
    Copyright (C) 2002 - 2008-10-01, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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

#include "stir/ProjDataInterfile.h" 
#include "stir/utilities.h"
#include "stir/IO/interfile.h"

#include <iostream>
#include <fstream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::fstream;
using std::vector;
using std::string;
using std::ios;
#endif

START_NAMESPACE_STIR


void
ProjDataInterfile ::
create_stream(const string& filename, const ios::openmode open_mode)
{
#if 1
  string data_name=filename;
  {
    string::size_type pos=find_pos_of_extension(filename);
    if (pos!=string::npos && filename.substr(pos)==".hs")
      replace_extension(data_name, ".s");
    else
      add_extension(data_name, ".s");
  }
  string header_name=filename;
#else
  char * data_name = new char[filename.size() + 5];
  {
    strcpy(data_name, filename.c_str());
    const char * const extension = strchr(find_filename(data_name),'.');
    if (extension!=NULL && strcmp(extension, ".hs")==0)
      replace_extension(data_name, ".s");
    else
      add_extension(data_name, ".s");
  }
  char * header_name = new char[filename.size() + 5];
  strcpy(header_name, data_name);
#endif
  replace_extension(header_name, ".hs");
  write_basic_interfile_PDFS_header(header_name, data_name,
                                    *this);

  sino_stream.reset( 
		    new fstream (data_name.c_str(), open_mode|ios::binary));
  if (!sino_stream->good())
  {
    error("ProjDataInterfile: error opening output file %s\n", data_name.c_str());
  }
#if 0
  delete[] header_name;
  delete[] data_name;
#endif
}

ProjDataInterfile ::
ProjDataInterfile (shared_ptr<ExamInfo> const& exam_info_sptr,
		     shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
		     const string& filename, const ios::openmode open_mode, 
		     const vector<int>& segment_sequence_in_stream,
		     StorageOrder o,
		     NumericType data_type,
		     ByteOrder byte_order,  
                     float scale_factor)
  : ProjDataFromStream(exam_info_sptr, proj_data_info_ptr, shared_ptr<iostream>(), 0, 
                     segment_sequence_in_stream, o, data_type, byte_order, scale_factor)
{
  create_stream(filename, open_mode);
}

ProjDataInterfile ::
ProjDataInterfile (shared_ptr<ExamInfo> const& exam_info_sptr,
		   shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
		   const string& filename, const ios::openmode open_mode, 
		   StorageOrder o,
		   NumericType data_type,
		   ByteOrder byte_order,  
                   float scale_factor )
  : ProjDataFromStream(exam_info_sptr, proj_data_info_ptr, shared_ptr<iostream>(), 0, 
                     o, data_type, byte_order, scale_factor)
{
  create_stream(filename, open_mode);
}

    

END_NAMESPACE_STIR
