/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamWithRecords
    
  \author Nikos Efthimiou
*/
/*
    Copyright (C) 2003-2011, Hammersmith Imanet Ltd
    Copyright (C) 2012-2013, Kris Thielemans
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


START_NAMESPACE_STIR

//InputStreamWithRecordsFromUPENNtxt::
//InputStreamWithRecordsFromUPENNtxt(std::string filename,
//                       std::streampos start_of_data)
//  : filename(filename),
//    starting_stream_position(start_of_data)
//{
//  std::fstream* s_ptr = new std::fstream;
//  open_read_binary(*s_ptr, filename.c_str());
//  stream_ptr.reset(s_ptr);
//  if (reset() == Succeeded::no)
//    error("InputStreamWithRecords: error in reset() for filename %s\n",
//      filename.c_str());
//}

Succeeded
InputStreamWithRecordsFromUPENNtxt::
create_output_file(std::string ofilename)
{
    error("InputStreamWithRecordsFromUPENNtxt: We do not support this here!");
}


END_NAMESPACE_STIR
