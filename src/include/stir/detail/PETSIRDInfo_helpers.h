
#pragma once
#include "stir/CartesianCoordinate3D.h"
#include <array>
#include <set>
#include <fmt/format.h>
#include "stir/info.h"
#include <vector>

/*!
  \namespace matrix
  \brief Lightweight 3x3 matrix and 3D vector helpers used during PETSIRD geometry analysis.

  Provides utilities to:
  - transpose a 3x3 matrix,
  - subtract two 3x3 matrices,
  - extract a rotation axis vector from the skew-symmetric part of a matrix.

  \details
  - Mat3: std::array<std::array<float,3>,3> for compact fixed-size storage.
  - Vec3: std::array<float,3> for simple 3D vectors.
  - getAxisFromSkew():
    Given S = R - R^T (skew-symmetric part of a rotation matrix R),
    returns the axis proportional to:
    (S_z,y - S_y,z)/2, (S_x,z - S_z,x)/2, (S_y,x - S_x,y)/2.
    For a pure rotation, S encodes the axis direction.
  - These helpers assume small numerical noise; thresholds are handled by callers.
  - No external dependencies; intended for quick geometric inference (e.g., rotation axis detection).
*/

namespace matrix
{

using Mat3 = std::array<std::array<float, 3>, 3>;
using Vec3 = std::array<float, 3>;

/*!
  \brief Transpose a 3x3 matrix.
  \param mat Input matrix.
  \return Transposed matrix.
*/
inline Mat3
transpose(const Mat3& mat)
{
  std::array<std::array<float, 3>, 3> result{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      result[j][i] = mat[i][j];
  return result;
}

/*!
  \brief Subtract two 3x3 matrices (A - B).
  \param A Left-hand matrix.
  \param B Right-hand matrix.
  \return Result of A - B.
*/
inline Mat3
subtract(const Mat3& A, const Mat3& B)
{
  std::array<std::array<float, 3>, 3> result{};
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      result[i][j] = A[i][j] - B[i][j];
  return result;
}

/*!
  \brief Extract rotation axis from the skew-symmetric matrix S = R - R^T.
  \param S Skew-symmetric matrix.
  \return Axis vector proportional to the rotation axis.
*/
inline Vec3
getAxisFromSkew(const Mat3& S)
{
  return {
    0.5f * (S[2][1] - S[1][2]), // x
    0.5f * (S[0][2] - S[2][0]), // y
    0.5f * (S[1][0] - S[0][1])  // z
  };
}

} // namespace matrix

/*!
  \namespace vector_utils
  \brief Helpers for spacing analysis and axis inference from coordinate sets.

  \details
  - \ref vector_utils::get_spacing_uniform computes successive spacings between
    sorted unique values and tests if they are uniform within a tolerance.
  - \ref vector_utils::getLargestVector returns the largest of three coordinate
    sets and logs which axis is inferred as “axial”.
*/
namespace vector_utils
{
/*!
  \brief Compute spacings between sorted unique values and test uniformity.
  \param spacing Output vector. Appends |x[i] - x[i-1]| for i=1..N-1, where x is the sorted version of \a unsorted_block_poss.
  \param unsorted_block_poss Set of unique positions (e.g., angles or translations).
  \param epsilon Tolerance for uniformity check (default 1e-4).
  \return True if all spacings differ from the first spacing by <= \a epsilon, false otherwise.

  \details
  - If \a spacing ends up empty (e.g., input size < 2), returns true (trivially uniform).
*/
inline bool
get_spacing_uniform(std::vector<float>& spacing, const std::set<float>& unsorted_block_poss, double epsilon = 1e-4)
{
  std::vector<float> sorted_z(unsorted_block_poss.begin(), unsorted_block_poss.end());
  for (size_t i = 1; i < sorted_z.size(); ++i)
    {
      spacing.push_back(std::abs(sorted_z[i] - sorted_z[i - 1]));
    }

  return std::all_of(spacing.begin(), spacing.end(), [&](float s) { return std::abs(s - spacing.front()) <= epsilon; });
}

/*!
  \brief Return the largest of three sets and report inferred axial direction.
  \param x Values along X.
  \param y Values along Y.
  \param z Values along Z.
  \return Const reference to the largest set among \a x, \a y, \a z.

  \details
  Logs the index of the inferred axial direction: 0 (x), 1 (y), or 2 (z).
*/
const std::set<float>&
get_largest_vector(const std::set<float>& x, const std::set<float>& y, const std::set<float>& z)
{
  const std::set<float>* largest = &x;
  int axis = 0;
  if (y.size() > largest->size())
    {
      largest = &y;
      axis = 1;
    }
  else if (z.size() > largest->size())
    {
      largest = &z;
      axis = 2;
    }

  //   stir::info(fmt::format("I believe the axial direction is the {}.", axis));
  return *largest;
}

/*!
  \brief Collect unique values from a 1D vector.
  \param values Output set for unique values.
  \param input Input vector.
*/
void
find_unique_values_1D(std::set<float>& values, const std::vector<float>& input)
{
  for (float val : input)
    {
      // std::cout << val << std::endl;
      values.insert(val);
    }
}

/*!
  \brief Collect unique values from a 2D vector (matrix).
  \param values Output set for unique values.
  \param input Input 2D vector [rows][cols].
*/
void
find_unique_values_2D(std::set<float>& values, const std::vector<std::vector<float>>& input)
{
  for (size_t row = 0; row < input.size(); ++row)
    for (size_t col = 0; col < input[row].size(); ++col)
      values.insert(input[row][col]);
}

} // namespace vector_utils

bool
almostEqual(double a, double b, double tol = 1e-6)
{
  return std::fabs(a - b) <= tol;
}

// Detect groupSize along dim2 (y) and loops along dim3 (z)
// Returns true on success, false if pattern doesn't match the assumed structure.
bool
inferGroupSizes_dim2_dim3(const std::vector<stir::CartesianCoordinate3D<float>>& pts,
                          std::size_t& groupSize_dim2,
                          std::size_t& groupSize_dim3,
                          float tol = 1e-5f)
{
  const std::size_t n = pts.size();
  groupSize_dim2 = groupSize_dim3 = 0;

  if (n == 0)
    return false;

  if (n == 1)
    {
      groupSize_dim2 = 1;
      groupSize_dim3 = 1;
      return true;
    }

  // STIR CartesianCoordinate3D<T> is (z, y, x)
  const float x0 = pts[0].x();
  const float z0 = pts[0].z();

  // 1) Find how many initial points keep x and z the same
  std::size_t runLen = 1;
  while (runLen < n && almostEqual(pts[runLen].x(), x0, tol) && almostEqual(pts[runLen].z(), z0, tol))
    {
      ++runLen;
    }

  groupSize_dim2 = runLen;

  // Must tile the full array
  if (groupSize_dim2 == 0 || n % groupSize_dim2 != 0)
    return false;

  groupSize_dim3 = n / groupSize_dim2;

  // 2) Check each block of size groupSize_dim2 has constant x,z
  for (std::size_t b = 0; b < groupSize_dim3; ++b)
    {
      std::size_t start = b * groupSize_dim2;
      float xb = pts[start].x();
      float zb = pts[start].z();

      for (std::size_t i = 1; i < groupSize_dim2; ++i)
        {
          const auto& p = pts[start + i];
          if (!almostEqual(p.x(), xb, tol) || !almostEqual(p.z(), zb, tol))
            {
              return false; // pattern breaks inside a block
            }
        }
    }

  // 3) Optionally check that z actually changes between blocks
  for (std::size_t b = 1; b < groupSize_dim3; ++b)
    {
      float z_prev = pts[(b - 1) * groupSize_dim2].z();
      float z_curr = pts[b * groupSize_dim2].z();
      if (almostEqual(z_prev, z_curr, tol))
        {
          return false; // outer loop didn't move in z
        }
    }

  return true;
}