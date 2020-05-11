//
//
/*
    Copyright (C) 2020, University College London
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
/*!
  \file
  \ingroup test

  \brief Test program for stir::KeyParser

  \author Kris Thielemans

*/

#include "stir/KeyParser.h"

#include <sstream>
#include "stir/RunTests.h"

START_NAMESPACE_STIR

template <typename elemT>
class TestKP : public KeyParser
{
public:
  TestKP()
    :
    scalar_v(0),
    vector_v(2,0)
  {
    add_start_key("start");
    add_stop_key("stop");
    add_key("scalar", &scalar_v);
    add_vectorised_key("vector", &vector_v);
  }
  elemT scalar_v;
  std::vector<elemT> vector_v;
};

/*!
  \ingroup test
  \brief Test class for KeyParser

*/
class KeyParserTests : public RunTests
{
public:
  template <typename elemT> void run_tests_one_type();
  void run_tests();
};

void
KeyParserTests::run_tests()
{
  std::cerr << "Tests for KeyParser\n";
  std::cerr << "... int parsing\n";
  run_tests_one_type<int>();
  std::cerr << "... unsigned int parsing\n";
  run_tests_one_type<unsigned int>();
  std::cerr << "... float parsing\n";
  run_tests_one_type<float>();
  std::cerr << "... double parsing\n";
  run_tests_one_type<double>();
}

template <typename elemT>
void
KeyParserTests::run_tests_one_type()
{

  TestKP<elemT>  parser;
  // basic test if parsing ok
  {
    std::stringstream str;
    str << "start:=\n"
        << "scalar:=2\n"
        << "vector[1] := 3\n"
        << "stop     :=\n";
    parser.parse(str);
    check_if_equal(parser.scalar_v, static_cast<elemT>(2), "parsing int");
    check_if_equal(parser.vector_v[0], static_cast<elemT>(3), "parsing int vector");
  }
  // test 1 if parsing catches errors
  {
    std::stringstream str;
    str << "start:=\n"
        << "scalar[1]:=2\n"
        << "vector[1] := 3\n"
        << "stop     :=\n";
    try
      {
        std::cerr <<  "Next test should write an error (but not crash!)"  << std::endl;
        parser.parse(str);
        check(false, "parsing non-vectorised key with vector should have failed");
      }
    catch (...)
      {
        // ok
      }
  }
  // test 2 if parsing catches errors
  {
    std::stringstream str;
    str << "start:=\n"
        << "scalar:=2\n"
        << "vector := 3\n"
        << "stop     :=\n";
    try
      {
        std::cerr <<  "Next test should write an error (but not crash!)"  << std::endl;
        parser.parse(str);
        check(false, "parsing vectorised key with non-vector should have failed");
      }
    catch (...)
      {
        // ok
      }
  }

}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main()
{
  KeyParserTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
