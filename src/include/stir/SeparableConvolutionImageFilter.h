//
//
/*!

  \file
  \ingroup ImageProcessor
  \brief Declaration of class stir::SeparableConvolutionImageFilter

  \author Kris Thielemans

*/
/*
    Copyright (C) 2002- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SeparableConvolutionImageFilter_H__
#define __stir_SeparableConvolutionImageFilter_H__

#include "stir/SeparableArrayFunctionObject.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VectorWithOffset.h"
#include <vector>

START_NAMESPACE_STIR

// TODO!! remove define
// currently fixed at 3 because I didn't really have a good idea for the parsing
// keywords in n dimensions.
//
#define num_dimensions 3

/*!
  \ingroup ImageProcessor
  \brief A class derived from DataProcessor for performing separable
    non-periodic convolutions.

    This filter applies a 1D convolution in all directions (z,y,x)
    with potentially a different filter kernel for every direction.

    When parsing, the filter coefficients are read as a list of numbers for each
    direction. The following conventions is used:
    <ol>
    <li>A list of 0 length (which is the default) corresponds to no filtering.
    <li>When the list contains an even number of data, a 0 is appended (at the end).
    <li>After this, the central element of the list corresponds to the 0-th element
    in the kernel, see below.
    </ol>
    Convolution is non-periodic. In each direction, the following is applied:

    \f[ out_i = \sum_j kernelforthisdirection_j in_{i-j} \f]

    Note that for most kernels, the above convention means that the zero-
    index of the kernel corresponds to the peak in the kernel.

    Elements of the input array that are outside its
    index range are considered to be 0.

    \warning There is NO check if the kernel coefficients add up to 1. This is
    because not all filters need this (e.g. edge enhancing filters).

    \par Example input for a low-pass filter in x,y, no filtering in z
    \verbatim
    Separable Convolution Parameters :=
    x-dir filter coefficients := {0.25, .5, .25}
    y-dir filter coefficients := {0.25, .5, .25}
    ;z-dir filter coefficients :=
    END Separable Convolution Parameters :=
    \endverbatim

    The filter is implemented using the class ArrayFilter1DUsingConvolution.
*/
template <typename elemT>
class SeparableConvolutionImageFilter : public RegisteredParsingObject<SeparableConvolutionImageFilter<elemT>,
                                                                       DataProcessor<DiscretisedDensity<3, elemT>>,
                                                                       DataProcessor<DiscretisedDensity<3, elemT>>>
{
private:
  typedef RegisteredParsingObject<SeparableConvolutionImageFilter<elemT>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>>
      base_type;

public:
  //! Name for parsing registry
  static const char* const registered_name;

  //! Default constructor
  SeparableConvolutionImageFilter();

  //! Constructor taking filter coefficients explicitly.
  /*! These filter coefficients are passed to the
      ArrayFilter1DUsingConvolution constructor.

      \a filter_coefficients has to have length 3. (Start index is irrelevant). Its
      first element will be applied to the 'first dimension', i.e. the first index.
  */
  SeparableConvolutionImageFilter(const VectorWithOffset<VectorWithOffset<elemT>>& filter_coefficients);

  //! Overloaded get and set methods the filter coefficients for axis or set of filter coefficients
  //@{
  VectorWithOffset<VectorWithOffset<elemT>> get_filter_coefficients();
  void set_filter_coefficients(const VectorWithOffset<VectorWithOffset<elemT>>& v);
  // These methods are made available as VectorWithOffset<VectorWithOffset<elemT>> is not accessable via swig
  VectorWithOffset<elemT> get_filter_coefficients(const int axis);
  void set_filter_coefficients(const int axis, const VectorWithOffset<elemT>& v);
  //@}

private:
  // silly business because KeyParser supports only LIST_OF_DOUBLES
  // TODO remove
  std::vector<std::vector<double>> filter_coefficients_for_parsing;

  VectorWithOffset<VectorWithOffset<elemT>> filter_coefficients;

  SeparableArrayFunctionObject<num_dimensions, elemT> filter;

  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

  Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions, elemT>& image) override;
  void virtual_apply(DiscretisedDensity<num_dimensions, elemT>& out_density,
                     const DiscretisedDensity<num_dimensions, elemT>& in_density) const override;
  void virtual_apply(DiscretisedDensity<num_dimensions, elemT>& density) const override;
};

#undef num_dimensions

END_NAMESPACE_STIR

#endif
