/*!

  \file
  \ingroup test

  \brief Test program for stir::ByteOrder and ByteOrderDefine.h

  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"
#include "stir/RunTests.h"

#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR


/*!
  \brief Test class for ByteOrder and the preprocessor defines from
  ByteOrderDefine.h
  \ingroup test
*/
class ByteOrderTests : public RunTests
{
public:
  void run_tests();
};

void
ByteOrderTests::run_tests()
{
  cerr << "Tests for ByteOrder\n"
       << "Everythings is fine if the program runs without any output." << endl;
  // if any of these is wrong, check ByteOrderDefine.h

#if STIRIsNativeByteOrderBigEndian
  check(ByteOrder::get_native_order() == ByteOrder::big_endian,
	"STIRIsNativeByteOrderBigEndian preprocessor define is determined incorrectly.");
#else
  check(ByteOrder::get_native_order() == ByteOrder::little_endian,
	"STIRIsNativeByteOrderBigEndian preprocessor define is determined incorrectly.");
#endif

#if STIRIsNativeByteOrderLittleEndian
  check(ByteOrder::get_native_order() == ByteOrder::little_endian,
	"STIRIsNativeByteOrderBigEndian preprocessor define is determined incorrectly.");
#else
  check(ByteOrder::get_native_order() == ByteOrder::big_endian,
	"STIRIsNativeByteOrderBigEndian preprocessor define is determined incorrectly.");
#endif

}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  ByteOrderTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
