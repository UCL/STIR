//
//
/*!

  \file
  \ingroup test

  \brief Test program for MultipleProjDataTests

  Give a txt file with the names of the projection data within. e.g.,:

  Multi :=
  	total number of data sets := 2
  	data set[1] := sinogram_1.hs
  	data set[2] := sinogram_2.hs
  end :=


  \author Richard Brown

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
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

#include "stir/num_threads.h"
#include "stir/RunTests.h"
#include "stir/MultipleProjData.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::setw;
using std::endl;
using std::min;
using std::max;
using std::size_t;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for MultipleProjData
*/

class MultipleProjDataTests: public RunTests
{
public:
  void run_tests();
  void set_multi_file(std::string multi_file) { _multi_file = multi_file; }
private:
  std::string _multi_file;
};

void
MultipleProjDataTests::run_tests()

{
  std::cout << "-------- Testing MultipleProjData --------\n";
  {
    // Test on the empty constructor
    std::cout << "\n\nTesting empty constructor.\n";
    MultipleProjData test1;
    std::cout << "OK!\n";

    // Test with parser
    std::cout << "\n\nTesting MultipleProjData with parser.\n";
    shared_ptr<MultipleProjData> test2;
    test2 = MultipleProjData::read_from_file(_multi_file);
    std::cout << "OK!\n";
  }
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 2) {
    std::cerr << "\n\tUsage: " << argv[0] << " multi_filename\n";
    return EXIT_FAILURE;
  }

  set_default_num_threads();

  {
    MultipleProjDataTests tests;
    tests.set_multi_file(argv[1]);
    tests.run_tests();
    if (!tests.is_everything_ok())
      return tests.main_return_value();
  }
}
