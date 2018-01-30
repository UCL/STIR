//
//
/*
    Copyright (C) 2004- 2007, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \brief Utility program that prints out values from an RDF file.

  \author Kris Thielemans
*/


#include "local/stir/IO/GE/SinglesRatesFromRDF.h"
#include "stir/stream.h"
#include "stir/IndexRange2D.h"
#include <iostream>
#include <string>

USING_NAMESPACE_STIR




int 
main (int argc, char **argv)
{

  // Check arguments. 
  // Singles filename 
  if (argc != 2) {
    cerr << "Program to print out values from a singles file.\n\n";
    cerr << "Usage: " << argv[0] << " rdf_filename \n\n";
    exit(EXIT_FAILURE);
  }

  const std::string rdf_filename = argv[1];
  // Singles file object.
  GE_IO::SinglesRatesFromRDF singles_from_rdf;

  // Read the singles file.
  if (singles_from_rdf.read_from_file(rdf_filename)==0)
    error("Error while reading singles file");


  // Get total number of frames
  //int num_frames = singles_from_rdf.get_num_frames();
  
  // Get scanner details and, from these, the number of singles units.
  const Scanner& scanner = *singles_from_rdf.get_scanner_ptr();
  Array<2,float> singles(IndexRange2D(scanner.get_num_axial_singles_units(),
				      scanner.get_num_transaxial_singles_units()));
  for (int ax=0; ax<scanner.get_num_axial_singles_units(); ++ax)
    {
      for (int transax=0; transax<scanner.get_num_transaxial_singles_units(); ++transax)
	{
	  const int singles_bin_index = scanner.get_singles_bin_index(ax, transax);
	  singles[ax][transax] = singles_from_rdf.get_singles_rate(singles_bin_index, 1);
	}
    }

  std::cout<< singles;

  
  return EXIT_SUCCESS;
}
