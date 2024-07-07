/*
 *  Copyright (C) 2016, UCL
    Copyright (C) 2002 - 2011-02-23, Hammersmith Imanet Ltd
    Copyright (C) 2019-2020, 2023, UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata
  \brief Declaration of class stir::ProjDataInMemory

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#ifndef __stir_ProjDataInMemory_H__
#define __stir_ProjDataInMemory_H__

#include "stir/ProjData.h"
#include "stir/Array.h"
#include <string>

START_NAMESPACE_STIR

class Succeeded;

/*!
  \ingroup projdata
  \brief A class which reads/writes projection data from/to memory.

  Mainly useful for temporary storage of projection data.

*/
class ProjDataInMemory : public ProjData
{
#ifdef SWIG
  // SWIG needs this typedef to be public
 public:
#endif
  typedef ProjDataInMemory self_type;
  typedef ProjData base_type;

public:
  //! constructor with only info, but no data
  /*!
    \param proj_data_info_ptr object specifying all sizes etc.
      The ProjDataInfo object pointed to will not be modified.
    \param initialise_with_0 specifies if the data should be set to 0.
        If \c false, the data is undefined until you set it yourself.
  */
  ProjDataInMemory(shared_ptr<const ExamInfo> const& exam_info_sptr,
                   shared_ptr<const ProjDataInfo> const& proj_data_info_ptr,
                   const bool initialise_with_0 = true);

  //! constructor that copies data from another ProjData
  ProjDataInMemory(const ProjData& proj_data);

  //! Copy constructor
  ProjDataInMemory(const ProjDataInMemory& proj_data);

  Viewgram<float> get_viewgram(const int view_num,
                               const int segment_num,
                               const bool make_num_tangential_poss_odd = false,
                               const int timing_pos = 0) const override;
  Succeeded set_viewgram(const Viewgram<float>& v) override;

  Sinogram<float> get_sinogram(const int ax_pos_num,
                               const int segment_num,
                               const bool make_num_tangential_poss_odd = false,
                               const int timing_pos = 0) const override;

  Succeeded set_sinogram(const Sinogram<float>& s) override;

  //! Get all sinograms for the given segment
  SegmentBySinogram<float> get_segment_by_sinogram(const int segment_num, const int timing_pos = 0) const override;
  //! Get all viewgrams for the given segment
  SegmentByView<float> get_segment_by_view(const int segment_num, const int timing_pos = 0) const override;

  //! Set all sinograms for the given segment
  Succeeded set_segment(const SegmentBySinogram<float>&) override;
  //! Set all viewgrams for the given segment
  Succeeded set_segment(const SegmentByView<float>&) override;

  //! set all bins to the same value
  /*! will call error() if setting failed */
  void fill(const float value) override;

  //! set all bins from another ProjData object
  /*! will call error() if setting failed or if the 'source' proj_data is not compatible.
    The current check requires at least the same segment numbers (but the source can have more),
    all other geometric parameters have to be the same.
 */
  void fill(const ProjData&) override;

  //! destructor deallocates all memory the object owns
  ~ProjDataInMemory() override;

  //! Returns a  value of a bin
  float get_bin_value(Bin& bin);

  void set_bin_value(const Bin& bin);

  //! @name arithmetic operations
  ///@{
  //! return sum of all elements
  float sum() const override;

  //! return maximum value of all elements
  float find_max() const override;

  //! return minimum value of all elements
  float find_min() const override;

  //! return L2-norm (sqrt of sum of squares)
  double norm() const override;

  //! return L2-norm squared (sum of squares)
  double norm_squared() const override;

  //! elem by elem addition
  self_type operator+(const self_type& iv) const;

  //! elem by elem subtraction
  self_type operator-(const self_type& iv) const;

  //! elem by elem multiplication
  self_type operator*(const self_type& iv) const;

  //! elem by elem division
  self_type operator/(const self_type& iv) const;

  //! addition with a 'float'
  self_type operator+(const float a) const;

  //! subtraction with a 'float'
  self_type operator-(const float a) const;

  //! multiplication with a 'float'
  self_type operator*(const float a) const;

  //! division with a 'float'
  self_type operator/(const float a) const;

  // corresponding assignment operators

  //! adding elements of \c v to the current data
  self_type& operator+=(const base_type& v) override;

  //! subtracting elements of \c v from the current data
  self_type& operator-=(const base_type& v) override;

  //! multiplying elements of the current data with elements of \c v
  self_type& operator*=(const base_type& v) override;

  //! dividing all elements of the current data by elements of \c v
  self_type& operator/=(const base_type& v) override;

  //! adding an \c float to the elements of the current data
  self_type& operator+=(const float v) override;

  //! subtracting an \c float from the elements of the current data
  self_type& operator-=(const float v) override;

  //! multiplying the elements of the current data with an \c float
  self_type& operator*=(const float v) override;

  //! dividing the elements of the current data by an \c float
  self_type& operator/=(const float v) override;

  //! \deprecated a*x+b*y (use xapyb)
  STIR_DEPRECATED void axpby(const float a, const ProjData& x, const float b, const ProjData& y) override;

  //! set values of the array to x*a+y*b, where a and b are scalar, and x and y are ProjData.
  /// This implementation requires that x and y are ProjDataInMemory
  /// (else falls back on general method)
  void xapyb(const ProjData& x, const float a, const ProjData& y, const float b) override;

  //! set values of the array to x*a+y*b, where a, b, x and y are ProjData.
  /// This implementation requires that a, b, x and y are ProjDataInMemory
  /// (else falls back on general method)
  void xapyb(const ProjData& x, const ProjData& a, const ProjData& y, const ProjData& b) override;

  //! set values of the array to self*a+y*b where a and b are scalar, y is ProjData
  /// This implementation requires that a, b and y are ProjDataInMemory
  /// (else falls back on general method)
  void sapyb(const float a, const ProjData& y, const float b) override;

  //! set values of the array to self*a+y*b where a, b and y are ProjData
  /// This implementation requires that a, b and y are ProjDataInMemory
  /// (else falls back on general method)
  void sapyb(const ProjData& a, const ProjData& y, const ProjData& b) override;
  ///@}

  /** @name iterator typedefs
   *  iterator typedefs
   */
  ///@{
  typedef Array<1, float>::iterator iterator;
  typedef Array<1, float>::const_iterator const_iterator;
  typedef Array<1, float>::full_iterator full_iterator;
  typedef Array<1, float>::const_full_iterator const_full_iterator;
  ///@}

  //! start value for iterating through all elements in the array, see iterator
  iterator begin()
  {
    return buffer.begin();
  }
  //! start value for iterating through all elements in the (const) array, see iterator
  const_iterator begin() const
  {
    return buffer.begin();
  }
  //! end value for iterating through all elements in the array, see iterator
  iterator end()
  {
    return buffer.end();
  }
  //! end value for iterating through all elements in the (const) array, see iterator
  const_iterator end() const
  {
    return buffer.end();
  }
  //! start value for iterating through all elements in the array, see iterator
  iterator begin_all()
  {
    return buffer.begin_all();
  }
  //! start value for iterating through all elements in the (const) array, see iterator
  const_iterator begin_all() const
  {
    return buffer.begin_all();
  }
  //! end value for iterating through all elements in the array, see iterator
  iterator end_all()
  {
    return buffer.end_all();
  }
  //! end value for iterating through all elements in the (const) array, see iterator
  const_iterator end_all() const
  {
    return buffer.end_all();
  }

  //! \name access to the data via a pointer
  //@{
  //! member function for access to the data via a float*
  float* get_data_ptr()
  {
    return buffer.get_data_ptr();
  }

  //! member function for access to the data via a const float*
  const float* get_const_data_ptr() const
  {
    return buffer.get_const_data_ptr();
  }

  //! signal end of access to float*
  void release_data_ptr()
  {
    buffer.release_data_ptr();
  }

  //! signal end of access to const float*
  void release_const_data_ptr() const
  {
    buffer.release_const_data_ptr();
  }
  //@}

private:
  Array<1, float> buffer;

  //! allocates buffer for storing the data. Has to be called by constructors
  void create_buffer(const bool initialise_with_0 = false);
  //! offset of the whole 3d sinogram in the stream
  std::streamoff offset;
  //! offset of a complete non-tof sinogram
  std::streamoff offset_3d_data;

  //! the order in which the segments occur in the stream
  std::vector<int> segment_sequence;
  //! the order in which the timing bins occur in the stream
  std::vector<int> timing_poss_sequence;

  //! Calculate the offset for a specific bin
  /*! Throws if out-of-range or other error */
  std::streamoff get_index(const Bin&) const;
};

inline double
norm(const ProjDataInMemory& p)
{
  return p.norm();
}

inline double
norm_squared(const ProjDataInMemory& p)
{
  return p.norm_squared();
}

END_NAMESPACE_STIR

#endif
