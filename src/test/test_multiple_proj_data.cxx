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
#include "stir/ProjDataInMemory.h"
#include <fstream>

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
  void set_multi_file(const std::string &multi_file) { _multi_file = multi_file; }
private:
  std::string _multi_file;
};

void
MultipleProjDataTests::run_tests()
{
    std::cout << "-------- Testing MultipleProjData --------\n";

    // Create single proj data
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
    shared_ptr<ProjDataInfo> proj_data_info_sptr(ProjDataInfo::ProjDataInfoCTI(scanner_sptr, 1, 10, 96, 128, true));
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);

    // Create and write proj data 1
    shared_ptr<ProjDataInMemory> proj_data_1_sptr(new ProjDataInMemory(exam_info_sptr, proj_data_info_sptr));
    float fill_value = 5.F;
    proj_data_1_sptr->fill(fill_value);
    proj_data_1_sptr->write_to_file("test_proj_data1");

    // Create and write proj data 2
    shared_ptr<ProjDataInMemory> proj_data_2_sptr(new ProjDataInMemory(exam_info_sptr, proj_data_info_sptr));
    proj_data_2_sptr->fill(fill_value*2.F);
    proj_data_2_sptr->write_to_file("test_proj_data2");

    // Create a multi header file
    std::ofstream myfile("test_multi_file.txt");
    if (myfile.is_open()) {
        myfile << "Multi :=\n";
        myfile << "\ttotal number of data sets := 2\n";
        myfile << "\tdata set[1] := test_proj_data1.hs\n";
        myfile << "\tdata set[2] := test_proj_data2.hs\n";
        myfile << "end :=\n";
        myfile.close();
    }
    else {
        everything_ok = false;
        return;
    }

    // Read back in
    shared_ptr<MultipleProjData> read_in_multi_proj_data;
    read_in_multi_proj_data = MultipleProjData::read_from_file("test_multi_file.txt");

    // Compare results
    check_if_equal(read_in_multi_proj_data->get_proj_data(1).get_viewgram(0,0).find_max(),
                   proj_data_1_sptr->get_viewgram(0,0).find_max(),
                   "test between maxes of first sinogram");

    check_if_equal(read_in_multi_proj_data->get_proj_data(2).get_viewgram(0,0).find_min(),
                   proj_data_2_sptr->get_viewgram(1,1).find_min(),
                   "test between mins of second sinogram");
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{

  set_default_num_threads();

  {
    MultipleProjDataTests tests;
    tests.run_tests();

    return tests.main_return_value();
  }
}
