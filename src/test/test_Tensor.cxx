/*!
  \file
  \ingroup test
  \ingroup TensorWrapper

  \brief tests for the stir::TensorWrapper class

  \author Nikos Efthimiou
*/

#ifndef NDEBUG
// set to high level of debugging
#  ifdef _DEBUG
#    undef _DEBUG
#  endif
#  define _DEBUG 2
#endif

#include "stir/Array.h"
#include "stir/TensorWrapper.h"
#include "stir/make_array.h"
#include "stir/Coordinate2D.h"
#include "stir/Coordinate3D.h"
#include "stir/Coordinate4D.h"
#include "stir/convert_array.h"
#include "stir/Succeeded.h"
#include "stir/IO/write_data.h"
#include "stir/IO/read_data.h"

#include "stir/RunTests.h"

#include "stir/ArrayFunction.h"
#include "stir/array_index_functions.h"
#include "stir/copy_fill.h"
#include <functional>
#include <algorithm>

// for open_read/write_binary
#include "stir/utilities.h"
#include "stir/info.h"
#include "stir/error.h"

#include "stir/HighResWallClockTimer.h"

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <boost/format.hpp>
using std::ofstream;
using std::ifstream;
using std::plus;
using std::cerr;
using std::endl;

START_NAMESPACE_STIR

namespace detail
{

// static TensorWrapper<2, float>
// test_make_array()
// {
//   return make_array(make_1d_array(1.F, 0.F, 0.F), make_1d_array(0.F, 1.F, 1.F), make_1d_array(0.F, -2.F, 2.F));
// }
} // namespace detail

/*!
  \brief Tests Array functionality
  \ingroup test
  \warning Running this will create and delete 2 files with names
  output.flt and output.other. Existing files with these names will be overwritten.

*/
class TensorTests : public RunTests
{
private:
  // // this function tests the next() function and compare it to using full_iterators
  // // sadly needs to be declared in the class for VC 6.0
  // template <int num_dimensions, class elemT>
  // void run_tests_on_next(const Array<num_dimensions, elemT>& test)
  // {
  //   // exit if empty array (as do..while() loop would fail)
  //   if (test.size() == 0)
  //     return;

  //   BasicCoordinate<num_dimensions, elemT> index = get_min_indices(test);
  //   typename Array<num_dimensions, elemT>::const_full_iterator iter = test.begin_all();
  //   do
  //     {
  //       check(*iter == test[index], "test on next(): element out of sequence?");
  //       ++iter;
  //   } while (next(index, test) && (iter != test.end_all()));
  //   check(iter == test.end_all(), "test on next() : did we cover all elements?");
  // }

  // // functions that runs IO tests for an array of arbitrary dimension
  // // sadly needs to be declared in the class for VC 6.0
  // template <int num_dimensions, typename elemT>
  // void run_IO_tests(const Array<num_dimensions, elemT>& t1)
  // {
  //   std::fstream os;
  //   std::fstream is;
  //   run_IO_tests_with_file_args(os, is, t1);
  //   FILE* ofptr;
  //   FILE* ifptr;
  //   run_IO_tests_with_file_args(ofptr, is, t1);
  //   run_IO_tests_with_file_args(ofptr, ifptr, t1);
  // }
  // template <int num_dimensions, typename elemT, class OFSTREAM, class IFSTREAM>
  // void run_IO_tests_with_file_args(OFSTREAM& os, IFSTREAM& is, const Array<num_dimensions, elemT>& t1)
  // {
  //   {
  //     open_write_binary(os, "output.flt");
  //     check(write_data(os, t1) == Succeeded::yes, "write_data could not write array");
  //     close_file(os);
  //   }
  //   Array<num_dimensions, elemT> t2(t1.get_index_range());
  //   {
  //     open_read_binary(is, "output.flt");
  //     check(read_data(is, t2) == Succeeded::yes, "read_data could not read from output.flt");
  //     close_file(is);
  //   }
  //   check_if_equal(t1, t2, "test out/in");
  //   remove("output.flt");

  //   {
  //     open_write_binary(os, "output.flt");
  //     const Array<num_dimensions, elemT> copy = t1;
  //     check(write_data(os, t1, ByteOrder::swapped) == Succeeded::yes, "write_data could not write array with swapped byte order");
  //     check_if_equal(t1, copy, "test out with byte-swapping didn't change the array");
  //     close_file(os);
  //   }
  //   {
  //     open_read_binary(is, "output.flt");
  //     check(read_data(is, t2, ByteOrder::swapped) == Succeeded::yes, "read_data could not read from output.flt");
  //     close_file(is);
  //   }
  //   check_if_equal(t1, t2, "test out/in (swapped byte order)");
  //   remove("output.flt");

  //   cerr << "\tTests writing as shorts\n";
  //   run_IO_tests_mixed(os, is, t1, NumericInfo<short>());
  //   cerr << "\tTests writing as floats\n";
  //   run_IO_tests_mixed(os, is, t1, NumericInfo<float>());
  //   cerr << "\tTests writing as signed chars\n";
  //   run_IO_tests_mixed(os, is, t1, NumericInfo<signed char>());

  //   /* check on failed IO.
  //      Note: needs to be after the others, as we would have to call os.clear()
  //      for ostream to be able to write again, but that's not defined for FILE*.
  //   */
  //   {
  //     const Array<num_dimensions, elemT> copy = t1;
  //     cerr << "\n\tYou should now see a warning that writing failed. That's by intention.\n";
  //     check(write_data(os, t1, ByteOrder::swapped) != Succeeded::yes, "write_data with swapped byte order should have failed");
  //     check_if_equal(t1, copy, "test out with byte-swapping didn't change the array even with failed IO");
  //   }
  // }

  // //! function that runs IO tests with mixed types for array of arbitrary dimension
  // // sadly needs to be implemented in the class for VC 6.0
  // template <int num_dimensions, typename elemT, class OFSTREAM, class IFSTREAM, class output_type>
  // void run_IO_tests_mixed(OFSTREAM& os,
  //                         IFSTREAM& is,
  //                         const Array<num_dimensions, elemT>& orig,
  //                         NumericInfo<output_type> output_type_info)
  // {
  //   {
  //     open_write_binary(os, "output.orig");
  //     elemT scale(1);
  //     check(write_data(os, orig, NumericInfo<elemT>(), scale) == Succeeded::yes,
  //           "write_data could not write array in original data type");
  //     close_file(os);
  //     check_if_equal(scale, static_cast<elemT>(1), "test out/in: data written in original data type: scale factor should be 1");
  //   }
  //   elemT scale(1);
  //   bool write_data_ok;
  //   {
  //     ofstream os;
  //     open_write_binary(os, "output.other");
  //     write_data_ok = check(write_data(os, orig, output_type_info, scale) == Succeeded::yes,
  //                           "write_data could not write array as other_type");
  //     close_file(os);
  //   }

  //   if (write_data_ok)
  //     {
  //       // only do reading test if data was written
  //       Array<num_dimensions, output_type> data_read_back(orig.get_index_range());
  //       {
  //         open_read_binary(is, "output.other");
  //         check(read_data(is, data_read_back) == Succeeded::yes, "read_data could not read from output.other");
  //         close_file(is);
  //         remove("output.other");
  //       }

  //       // compare with convert()
  //       {
  //         float newscale = static_cast<float>(scale);
  //         Array<num_dimensions, output_type> origconverted = convert_array(newscale, orig, NumericInfo<output_type>());
  //         check_if_equal(newscale, scale, "test read_data <-> convert : scale factor ");
  //         check_if_equal(origconverted, data_read_back, "test read_data <-> convert : data");
  //       }

  //       // compare orig/scale with data_read_back
  //       {
  //         const Array<num_dimensions, elemT> orig_scaled(orig / scale);
  //         this->check_array_equality_with_rounding(
  //             orig_scaled, data_read_back, "test out/in: data written as other_type, read as other_type");
  //       }

  //       // compare data written as original, but read as other_type
  //       {
  //         Array<num_dimensions, output_type> data_read_back2(orig.get_index_range());

  //         ifstream is;
  //         open_read_binary(is, "output.orig");

  //         elemT in_scale = 0;
  //         check(read_data(is, data_read_back2, NumericInfo<elemT>(), in_scale) == Succeeded::yes,
  //               "read_data could not read from output.orig");
  //         // compare orig/in_scale with data_read_back2
  //         const Array<num_dimensions, elemT> orig_scaled(orig / in_scale);
  //         this->check_array_equality_with_rounding(
  //             orig_scaled, data_read_back2, "test out/in: data written as original_type, read as other_type");
  //       }
  //     } // end of if(write_data_ok)
  //   remove("output.orig");
  // }

  // //! a special version of check_if_equal just for this class
  // /*! we check up to .5 if output_type is integer, and up to tolerance otherwise
  //  */
  // template <int num_dimensions, typename elemT, class output_type>
  // bool check_array_equality_with_rounding(const Array<num_dimensions, elemT>& orig,
  //                                         const Array<num_dimensions, output_type>& data_read_back,
  //                                         const char* const message)
  // {
  //   NumericInfo<output_type> output_type_info;
  //   bool test_failed = false;
  //   typename Array<num_dimensions, elemT>::const_full_iterator diff_iter = orig.begin_all();
  //   typename Array<num_dimensions, output_type>::const_full_iterator data_read_back_iter = data_read_back.begin_all_const();
  //   while (diff_iter != orig.end_all())
  //     {
  //       if (output_type_info.integer_type())
  //         {
  //           std::stringstream full_message;
  //           // construct useful error message even though we use a boolean check
  //           full_message << boost::format("unequal values are %2% and %3%. %1%: difference larger than .5") % message
  //                               % static_cast<elemT>(*data_read_back_iter) % *diff_iter;
  //           // difference should be maximum .5 (but we test with slightly larger tolerance to accomodate numerical precision)
  //           test_failed = check(fabs(*diff_iter - *data_read_back_iter) <= .502, full_message.str().c_str());
  //         }
  //       else
  //         {
  //           std::string full_message = message;
  //           full_message += ": difference larger than tolerance";
  //           test_failed = check_if_equal(static_cast<elemT>(*data_read_back_iter), *diff_iter, full_message.c_str());
  //         }
  //       if (test_failed)
  //         break;
  //       diff_iter++;
  //       data_read_back_iter++;
  //     }
  //   return test_failed;
  // }

public:
  void run_tests() override;
};

// // helper function to create a shared_ptr that doesn't delete the data (as it's still owned by the vector)
// template <typename T>
// shared_ptr<T[]>
// vec_to_shared(std::vector<T>& v)
// {
//   shared_ptr<T[]> sptr(v.data(), [](auto) {});
//   return sptr;
// }

void
TensorTests::run_tests()
{
  cerr << "Testing Tensor classes\n";
  {
    cerr << "Testing 1D stuff" << endl;
    {
      TensorWrapper<1, int> testint(IndexRange<1>(5));
      testint.at(1) = 2;
      check_if_equal(testint.size(), size_t(5), "test size()");
      check_if_equal(testint.size_all(), size_t(5), "test size_all()");
      TensorWrapper<1, float> test(IndexRange<1>(10));
      check_if_zero(test, "Array1D not initialised to 0");
      test.at(1) = 10.5f;
      test.set_offset(-1);
      check_if_equal(test.size(), size_t(10), "test size() with non-zero offset");
      check_if_equal(test.size_all(), size_t(10), "test size_all() with non-zero offset");
      check_if_equal(test.at(0), 10.5F, "test indexing of Array1D");
      test += 1;
      check_if_equal(test.at(0), 11.5F, "test operator+=(float)");
      check_if_equal(test.sum(), 20.5F, "test operator+=(float) and sum()");
      check_if_zero(test - test, "test operator-(Tensor1D)");
      BasicCoordinate<1, int> c;
      c[1] = 0;
      check_if_equal(test.at(c), 11.5F, "test at(BasicCoordinate)");
      test.at(c) = 12.5;
      check_if_equal(test.at(c), 12.5F, "test at(BasicCoordinate)");
      {
        //! NE: Here, test_Array calls for partial specialisations, or support for
        //! factories. I will skip that in favour of IndexRange<>
        // TensorWrapper<1, float> ref(-1, 2);
        const IndexRange<1> range(-1,1);
        TensorWrapper<1, float> ref(range);
        ref.at(-1) = 1.F;
        ref.at(0) = 3.F;
        ref.at(1) = 3.14F;
        TensorWrapper<1, float> test = ref;

        test += 1.f;
        for (int i = ref.get_min_index(); i <= ref.get_max_index(); ++i){
          check_if_equal(test.at(i), ref.at(i) + 1, "test operator+=(float)");
          }
        test = ref;
        test -= 4;
        for (int i = ref.get_min_index(); i <= ref.get_max_index(); ++i)
          check_if_equal(test.at(i), ref.at(i) - 4, "test operator-=(float)");
        test = ref;
        test *= 3;
        for (int i = ref.get_min_index(); i <= ref.get_max_index(); ++i)
          check_if_equal(test.at(i), ref.at(i) * 3, "test operator*=(float)");
        test = ref;
        test /= 3;
        for (int i = ref.get_min_index(); i <= ref.get_max_index(); ++i)
          check_if_equal(test.at(i), ref.at(i) / 3, "test operator/=(float)");
      }
        TensorWrapper<1, float> test2;
        test2 = test * 2;
        check_if_equal(2 * test.at(0), test2.at(0), "test operator*(float)");
      {

      }
#if 1
      {
        // tests on log/exp
        const IndexRange<1> range(-3,9);
        TensorWrapper<1, float> test(range);
        test.fill(1.F);
        in_place_log(test);
        {
          TensorWrapper<1, float> testeq(range);
          check_if_equal(test, testeq, "test in_place_log of TensorWrapper<1,float>");
        }
        {
          for (int i = test.get_min_index(); i <= test.get_max_index(); i++)
            test.at(i) = 3.5F * i + 100;
        }
        TensorWrapper<1, float> test_copy = test;
        in_place_log(test);
        in_place_exp(test);
        check_if_equal(test, test_copy, "test log/exp of Array1D");
      }
#endif
    }

    {
      cerr << "Testing 2D stuff" << endl;
      {
        const IndexRange<2> range(Coordinate2D<int>(0, 0), Coordinate2D<int>(9, 9));
        TensorWrapper<2, float> test2(range);
        check_if_equal(test2.size(), size_t(10), "test size()");
        check_if_equal(test2.size_all(), size_t(100), "test size_all()");
        check_if_zero(test2, "test TensorWrapper<2, float> not initialised to 0");
        test2.at(3,4) = 23.3f;
      }
      {
        IndexRange<2> range(Coordinate2D<int>(0, 0), Coordinate2D<int>(3, 3));
        TensorWrapper<2, float> testfp(range);
        TensorWrapper<2, float> t2fp(range);
        testfp.at(3,2) = 3.3F;
        t2fp.at(3,2) = 2.2F;
        TensorWrapper<2, float> t2 = t2fp + testfp;
        check_if_equal(t2.at(3,2), 5.5F, "test operator +(Array2D)");
        t2fp += testfp;
        check_if_equal(t2fp.at(3,2), 5.5F, "test operator +=(Array2D)");
        check_if_equal(t2, t2fp, "test comparing Array2D+= and +");
      }
    }

  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  TensorTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
