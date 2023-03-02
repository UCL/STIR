#ifndef NDEBUG
// set to high level of debugging
#  ifdef _DEBUG
#    undef _DEBUG
#  endif
#  define _DEBUG 2
#endif

#include "stir/ProjDataInfo.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Succeeded.h"
#include "stir/IO/write_data.h"
#include "stir/IO/read_data.h"
#include "stir/IO/write_to_file.h"
#include "stir/numerics/BSplines.h"
#include "stir/interpolate_projdata.h"
#include "stir/inverse_SSRB.h"

#include "stir/RunTests.h"

START_NAMESPACE_STIR

class InterpolationTests : public RunTests
{
public:
  void run_tests();

private:
  void scatter_interpolation_test();
};

void
InterpolationTests::scatter_interpolation_test()
{
  auto upsampled_proj_data = ProjDataInMemory::read_from_file("data_to_fit.hs"); // "scaled_1.hs");
  auto downsampled_proj_data = ProjDataInMemory::read_from_file("unscaled_1.hs");
  ProjDataInMemory interpolated_direct_scatter(upsampled_proj_data->get_exam_info_sptr(), upsampled_proj_data->get_proj_data_info_sptr()->create_shared_clone());
  interpolate_projdata(interpolated_direct_scatter, *downsampled_proj_data, BSpline::quadratic, false, false);
  ProjDataInMemory interpolated_scatter(upsampled_proj_data->get_exam_info_sptr(), upsampled_proj_data->get_proj_data_info_sptr()->create_shared_clone());
  // interpolated_scatter.fill_from(interpolated_direct_scatter.begin());
  inverse_SSRB(interpolated_scatter, interpolated_direct_scatter);
  interpolated_scatter.write_to_file("scaled_from_unscaled_1.hs");
}

void
InterpolationTests::run_tests()
{
  scatter_interpolation_test();
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  Verbosity::set(1);
  InterpolationTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
