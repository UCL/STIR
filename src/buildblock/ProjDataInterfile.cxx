//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementation of class ProjDataInterfile

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
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
#endif

START_NAMESPACE_STIR


void
ProjDataInterfile ::
create_stream(const string& filename, const ios::openmode open_mode)
{
  char * data_name = new char[filename.size() + 5];
  char * header_name = new char[filename.size() + 5];
  {
    strcpy(data_name, filename.c_str());
    const char * const extension = strchr(find_filename(data_name),'.');
    if (extension!=NULL && strcmp(extension, ".hs")==0)
      replace_extension(data_name, ".s");
    else
      add_extension(data_name, ".s");
  }
  strcpy(header_name, data_name);
  replace_extension(header_name, ".hs");
  write_basic_interfile_PDFS_header(header_name, data_name,
                                    *this);

  sino_stream = 
    new fstream (data_name, open_mode|ios::binary);
  if (!sino_stream->good())
  {
    error("ProjDataInterfile: error opening output file %s\n", data_name);      
  }
  delete header_name;
  delete data_name;
}

ProjDataInterfile ::
ProjDataInterfile (shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
		     const string& filename, const ios::openmode open_mode, 
		     const vector<int>& segment_sequence_in_stream,
		     StorageOrder o,
		     NumericType data_type,
		     ByteOrder byte_order,  
                     float scale_factor)
: ProjDataFromStream(proj_data_info_ptr, 0, 0, 
                     segment_sequence_in_stream, o, data_type, byte_order, scale_factor)
{
  create_stream(filename, open_mode);
}

ProjDataInterfile ::
ProjDataInterfile (shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
		   const string& filename, const ios::openmode open_mode, 
		   StorageOrder o,
		   NumericType data_type,
		   ByteOrder byte_order,  
                   float scale_factor )
: ProjDataFromStream(proj_data_info_ptr, 0, 0, 
                     o, data_type, byte_order, scale_factor)
{
  create_stream(filename, open_mode);
}

    

END_NAMESPACE_STIR
