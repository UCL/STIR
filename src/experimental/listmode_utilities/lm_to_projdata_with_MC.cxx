//
//
/*!
  \file 
  \ingroup listmode

  \brief main for LmToProjDataWithMC
 
  \author Kris Thielemans
  \author Sanida Mustafovic
  
*/
/*
    Copyright (C) 2003- 2003, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#include "stir_experimental/listmode/LmToProjDataWithMC.h"

#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

/*
 Here's a sample .par file
\verbatim
LmToProjDataWithMC Parameters := 

input file := listmode data
output filename := precorrected_MC_data.s
  
Bin Normalisation type:= From ECAT7
  Bin Normalisation From ECAT7:=
  normalisation_ECAT7_filename:= norm_filename.n
End Bin Normalisation From ECAT7:=

  store_prompts:=
  delayed_increment:=
  do_time_frame := 
  start_time:=
  end_time:=
  num_segments_in_memory:


 
*/



USING_NAMESPACE_STIR



USING_NAMESPACE_STIR

int main(int argc, char * argv[])
{
  
  if (argc!=1 && argc!=2) {
    cerr << "Usage: " << argv[0] << " [par_file]\n";
    exit(EXIT_FAILURE);
  }
  LmToProjDataWithMC application(argc==2 ? argv[1] : 0);
  application.process_data();

  return EXIT_SUCCESS;
}
