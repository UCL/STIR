//
//
/*
    Copyright (C) 2020, 2024, University College London
    This file is part of STIR.
 
    SPDX-License-Identifier: Apache-2.0

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
#include <vector>
#include <algorithm>
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
    add_alias_key("scalar", "new alias", false);
    add_alias_key("scalar", "deprecated alias", true);
    add_alias_key("vector", "new vector alias", false);
    add_alias_key("vector", "deprecated vector alias", true);
    add_vectorised_key("vector", &vector_v);
  }
  elemT scalar_v;
  std::vector<elemT> vector_v;
  bool operator==(TestKP& other) const
  {
    return scalar_v == other.scalar_v &&
      std::equal(vector_v.begin(),vector_v.end(), other.vector_v.begin());
  }
};

/*!
  \ingroup test
  \brief Test class for KeyParser

*/
class KeyParserTests : public RunTests
{
public:
  template <typename elemT> void run_tests_one_type();
  void run_tests() override;
};

void
KeyParserTests::run_tests()
{
  std::cerr << "Tests for KeyParser\n";
  std::cerr << "\n..... int parsing\n";
  run_tests_one_type<int>();
  std::cerr << "\n..... unsigned int parsing\n";
  run_tests_one_type<unsigned int>();
  std::cerr << "\n..... float parsing\n";
  run_tests_one_type<float>();
  std::cerr << "\n..... double parsing\n";
  run_tests_one_type<double>();
}

template <typename elemT>
void
KeyParserTests::run_tests_one_type()
{
  // basic test if parsing ok
  {
    TestKP<elemT>  parser;
    TestKP<elemT>  parser2;
    std::stringstream str;
    str << "start:=\n"
        << "scalar:=2\n"
        << "vector[1] := 3\n"
        << "stop     :=\n";
    parser.parse(str);
    check_if_equal(parser.scalar_v, static_cast<elemT>(2), "parsing scalar");
    check_if_equal(parser.vector_v[0], static_cast<elemT>(3), "parsing vector");
    parser2.parse(parser.parameter_info());
    check(parser == parser2, "check parsing of parameter_info()");
  }
  // test alias
  {
    TestKP<elemT>  parser;
    TestKP<elemT>  parser2;
    std::stringstream str;
    str << "start:=\n"
        << "new alias:=2\n"
        << "new vector alias[1] := 3\n"
        << "stop     :=\n";
    parser.parse(str);
    check_if_equal(parser.scalar_v, static_cast<elemT>(2), "parsing scalar with alias");
    check_if_equal(parser.vector_v[0], static_cast<elemT>(3), "parsing vector with alias");
    check(parser.parameter_info().find("alias") == std::string::npos, "check alias is not in parameter_info()");
    // check if parsing back parameter_info() gives same results
    parser2.parse(parser.parameter_info());
    check(parser == parser2, "check parsing of parameter_info() with alias");

    // same with deprecated alias
    std::cerr << "\nNext test should write warnings about deprecated alias\n";
    str << "start:=\n"
        << "deprecated alias:=3\n"
        << "deprecated vector alias[1] := 4\n"
        << "stop     :=\n";
    parser.parse(str);
    check_if_equal(parser.scalar_v, static_cast<elemT>(3), "parsing scalar with deprecated alias");
    check_if_equal(parser.vector_v[0], static_cast<elemT>(4), "parsing vector with deprecated alias");
    check(parser.parameter_info().find("alias") == std::string::npos, "check alias is not in parameter_info()");
    parser2.parse(parser.parameter_info());
    check(parser == parser2, "check parsing of parameter_info() with alias");
  }
  // test 1 if parsing catches errors
  {
    TestKP<elemT>  parser;
    std::stringstream str;
    str << "start:=\n"
        << "scalar[1]:=2\n"
        << "vector[1] := 3\n"
        << "stop     :=\n";
    try
      {
        std::cerr <<  "\nNext test should write an error (but not crash!)"  << std::endl;
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
    TestKP<elemT>  parser;
    std::stringstream str;
    str << "start:=\n"
        << "scalar:=2\n"
        << "vector := 3\n"
        << "stop     :=\n";
    try
      {
        std::cerr <<  "\nNext test should write an error (but not crash!)"  << std::endl;
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
