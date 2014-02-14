//
//

/*! 
\file
\ingroup ECAT_utilities
\brief Utility to make Interfile headers for ECAT7 data
\author Kris Thielemans



  Usage:  $File$ ecat7_filename.

  This will attempt to write interfile headers 'pointing into' the ECAT7 file. 
  
  A question will be
  asked if all data sets should be processed, or only a single one.

  The header filenames will be of the form
    ecat7_filename_extension_f1g1d0b0.h?. For example, for ecat7_filename test.S, and
    frame=2, gate=3, data=4, bed=5, the header name will be test_S_f2g3d4b5.hs

  \see write_basic_interfile_header_for_ecat7()
\warning This only works with some CTI file_types. In particular, it does NOT
work with the ECAT6-like files_types, as then there are subheaders 'in' the 
datasets.

\warning Implementation uses the Louvain la Neuve Ecat library. So, it will
only work on systems where this library works properly.

*/
/*
    Copyright (C) 2000- 2010, Hammersmith Imanet Ltd
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

#ifdef HAVE_LLN_MATRIX


#include "stir/ProjDataInfo.h"
#include "stir/ProjDataFromStream.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/IO/stir_ecat7.h"
#include <iostream>
#include <fstream>
#include <string>

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <stdarg.h>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::ios;
using std::iostream;
using std::fstream;
using std::cerr;
using std::endl;
#endif


USING_NAMESPACE_STIR
USING_NAMESPACE_ECAT
USING_NAMESPACE_ECAT7

int	
main( int argc, char **argv)
{
  MatrixFile *mptr;
  
  if (argc<2)
  {
    cerr << "usage    : "<< argv[0] << " filename\n";
    exit(EXIT_FAILURE);
  }

  mptr = matrix_open( argv[1], MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (!mptr) {
    matrix_perror(argv[1]);
    exit(EXIT_FAILURE);
  }
  
  const int num_frames = std::max(static_cast<int>( mptr->mhptr->num_frames),1);
  // funnily enough, num_bed_pos seems to be offset with 1
  // (That's to say, in a singled bed study, num_bed_pos==0) 
  // TODO maybe not true for multi-bed studies
  const int num_bed_poss = static_cast<int>( mptr->mhptr->num_bed_pos) + 1;
  const int num_gates = std::max(static_cast<int>( mptr->mhptr->num_gates),1);

  fclose(mptr->fptr);
  delete mptr;

  string interfile_header_filename;
  if (ask("Attempt all data-sets (Y) or single data-set (N)", true))
  {
    const int data_num=ask_num("Data number ? ",0,8, 0);

    for (int frame_num=1; frame_num<=num_frames;++frame_num)
      for (int bed_num=0; bed_num<num_bed_poss;++bed_num)
        for (int gate_num=1; gate_num<=num_gates;++gate_num)
          write_basic_interfile_header_for_ECAT7( interfile_header_filename,
                                                  argv[1], 
						  frame_num, gate_num, data_num, bed_num);
  }
  else
  {
    const int frame_num=ask_num("Frame number ? ",1,num_frames, 1);
    const int bed_num=ask_num("Bed number ? ",0,num_bed_poss-1, 0);
    const int gate_num=ask_num("Gate number ? ",1,num_gates, 1);
    const int data_num=ask_num("Data number ? ",0,8, 0);
    
    write_basic_interfile_header_for_ECAT7( interfile_header_filename,
                                            argv[1], frame_num, gate_num, data_num,
                                            bed_num);
  }
  return EXIT_SUCCESS;
}

#endif

