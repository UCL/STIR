
/*
    Copyright (C) 2025, UMCG
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup test

  \brief Test program for stir::PETSIRD hierarchy

  \author Nikos Efthimiou

*/
#include "stir/detail/PETSIRDInfo_helpers.h"
#include "stir/RunTests.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

class PETSIRDTests : public RunTests
{
public:
  void run_tests() override;

private:
  void test_find_unique_values_1D();
  void test_find_unique_values_2D();
  void test_get_largest_vector();
  void test_get_spacing_uniform();
  void test_get_AxisFromSkewMatrix();
  void test_infer_group_sizes_dim2_dim3();
};

void
PETSIRDTests::run_tests()
{
  test_find_unique_values_1D();
  test_find_unique_values_2D();
  test_get_largest_vector();
  test_get_spacing_uniform();
  test_get_AxisFromSkewMatrix();
  test_infer_group_sizes_dim2_dim3();
}

void
PETSIRDTests::test_find_unique_values_1D()
{
  std::vector<float> input = { 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 1.0f, 5.0f };
  std::set<float> expected = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
  std::set<float> result;

  vector_utils::find_unique_values_1D(result, input);
  for (const auto& val : expected)
    {
      this->check(result.find(val) != result.end(), fmt::format("Value {} should be in the unique set", val));
    }
}

void
PETSIRDTests::test_find_unique_values_2D()
{
  std::vector<std::vector<float>> input = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 2.0f, 6.0f }, { 1.0f, 8.0f, 9.0f } };
  std::set<float> expected = { 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 9.0f };
  std::set<float> result;

  vector_utils::find_unique_values_2D(result, input);
  for (const auto& val : expected)
    {
      this->check(result.find(val) != result.end(), fmt::format("Value {} should be in the unique set", val));
    }
}

void
PETSIRDTests::test_get_largest_vector()
{
  std::set<float> x = { 1.0f, 2.0f };
  std::set<float> y = { 1.0f, 2.0f, 3.0f };
  std::set<float> z = { 1.0f };

  const std::set<float>& largest = vector_utils::get_largest_vector(x, y, z);
  this->check_if_equal(largest.size(), y.size(), "Y should be the largest vector");
}

void
PETSIRDTests::test_get_spacing_uniform()
{
  std::set<float> values = { 0.0f, 2.0f, 4.0f, 6.0f, 8.0f };
  std::vector<float> spacing;
  std::set<float> spacings;
  bool is_uniform = vector_utils::get_spacing_uniform(spacing, values);

  this->check(is_uniform, "Spacing should be uniform");
  vector_utils::find_unique_values_1D(spacings, spacing);

  this->check_if_equal(static_cast<uint>(spacings.size()), 1u, "There should be one unique spacing value");
  this->check_if_equal(*spacings.begin(), 2.0f, "Spacing value should be 2.0f");
}

void
PETSIRDTests::test_get_AxisFromSkewMatrix()
{
  float angle_rad = static_cast<float>(M_PI) / 6.0f; // 30 degrees
  {
    // Create a rotation matrix around z axis
    matrix::Mat3 R = { { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };
    R[0][0] = std::cos(angle_rad);
    R[0][1] = -std::sin(angle_rad);
    R[1][0] = std::sin(angle_rad);
    R[1][1] = std::cos(angle_rad);

    std::array<std::array<float, 3>, 3> skew = matrix::subtract(R, matrix::transpose(R));
    auto axis = matrix::getAxisFromSkew(skew);
    this->check_if_equal(axis[0], 0.0f, "X component of rotation axis should be 0");
    this->check_if_equal(axis[1], 0.0f, "Y component of rotation axis should be 0");
    this->check_if_equal(axis[2], 1.0f, "Z component of rotation axis should be 1");
  }
  {
    // Create a rotation matrix around x axis
    matrix::Mat3 R = { { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };
    R[1][1] = std::cos(angle_rad);
    R[1][2] = -std::sin(angle_rad);
    R[2][1] = std::sin(angle_rad);
    R[2][2] = std::cos(angle_rad);

    std::array<std::array<float, 3>, 3> skew = matrix::subtract(R, matrix::transpose(R));
    auto axis = matrix::getAxisFromSkew(skew);
    this->check_if_equal(axis[0], 1.0f, "X component of rotation axis should be 1");
    this->check_if_equal(axis[1], 0.0f, "Y component of rotation axis should be 0");
    this->check_if_equal(axis[2], 0.0f, "Z component of rotation axis should be 0");
  }
}

void
PETSIRDTests::test_infer_group_sizes_dim2_dim3()
{
  std::vector<stir::CartesianCoordinate3D<float>> pts;
  // Create a grid of points with groupSize_dim2 = 3 and groupSize_dim3 = 4
  for (int z = 0; z < 4; ++z)
    {
      for (int y = 0; y < 3; ++y)
        {
          // Test with float numbers and something in the x coordintate so that we don't have all zeros
          pts.emplace_back(static_cast<float>(z + 0.4), static_cast<float>(z + 0.35), 0.1f + z / 2.f);
        }
    }

  std::size_t groupSize_dim2 = 0;
  std::size_t groupSize_dim3 = 0;
  bool success = inferGroupSizes_dim2_dim3(pts, groupSize_dim2, groupSize_dim3);

  this->check(success, "inferGroupSizes_dim2_dim3 should succeed");
  this->check_if_equal(groupSize_dim2, static_cast<std::size_t>(3), "groupSize_dim2 should be 3");
  this->check_if_equal(groupSize_dim3, static_cast<std::size_t>(4), "groupSize_dim3 should be 4");
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  PETSIRDTests test;
  test.run_tests();
  return test.main_return_value();
}
