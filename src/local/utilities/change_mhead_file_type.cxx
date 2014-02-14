//
//

/*! 
\file
\ingroup utilities
\brief A small utility that changes the file type in an ECAT7 main header.
\author Kris Thielemans
*/
/*
    Copyright (C) 2002- 2002, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/Succeeded.h"
#include "matrix.h"

#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

USING_NAMESPACE_STIR




int main(int argc, char *argv[])
{
  
  if(argc!=3)
  {
    cerr<< "\nChange file type in  ECAT7 CTI main header.\n"
        << "Usage: \n"
	<< "\t" <<argv[0] <<" ECAT7_name new_file_type_number\n";
    return EXIT_FAILURE;
  }
  char cti_name[1000];
  const char * const program_name = argv[0];
  strcpy(cti_name,argv[1]);	
  const int file_type= atoi(argv[2]);

			    
  MatrixFile* mptr= matrix_open(cti_name, MAT_OPEN_EXISTING, MAT_UNKNOWN_FTYPE);
  if (mptr == 0)
    {
      warning("%s: error opening file %s\n", program_name, cti_name);
      return EXIT_FAILURE;
    }

  mptr->mhptr->file_type = static_cast<short>(file_type);
  Succeeded success =
    mat_write_main_header(mptr->fptr, mptr->mhptr)==0?Succeeded::yes : Succeeded::no;
  matrix_close(mptr);
  

  return success==Succeeded::yes ?EXIT_SUCCESS:EXIT_FAILURE;
}


