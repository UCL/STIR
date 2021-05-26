//
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class SeparableGaussianImageFilter

  \author Sanida Mustafovic
  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SeparableGaussianImageFilter_H__
#define __stir_SeparableGaussianImageFilter_H__

#include "stir_experimental/SeparableGaussianArrayFilter.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

// TODO!! remove define

#define num_dimensions 3

template <typename elemT>
class SeparableGaussianImageFilter : public RegisteredParsingObject<SeparableGaussianImageFilter<elemT>,
                                                                    DataProcessor<DiscretisedDensity<num_dimensions, elemT>>,
                                                                    DataProcessor<DiscretisedDensity<num_dimensions, elemT>>> {
private:
  typedef RegisteredParsingObject<SeparableGaussianImageFilter<elemT>, DataProcessor<DiscretisedDensity<num_dimensions, elemT>>,
                                  DataProcessor<DiscretisedDensity<num_dimensions, elemT>>>
      base_type;

public:
  static const char* const registered_name;

  //! Default constructor
  SeparableGaussianImageFilter();

  float get_standard_deviation();

private:
  float standard_deviation;
  int number_of_coefficients;

  SeparableGaussianArrayFilter<num_dimensions, elemT> gaussian_filter;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions, elemT>& image);
  // new
  void virtual_apply(DiscretisedDensity<num_dimensions, elemT>& out_density,
                     const DiscretisedDensity<num_dimensions, elemT>& in_density) const;
  void virtual_apply(DiscretisedDensity<num_dimensions, elemT>& density) const;
};

#undef num_dimensions

END_NAMESPACE_STIR

#endif
