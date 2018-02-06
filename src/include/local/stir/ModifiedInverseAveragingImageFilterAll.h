//
//
/*!

  \file
  \ingroup buildblock  
  \brief This is a messy, first, attempt to design spatially varying filter 
   Given the kernel which in this case is a lospass filter with a DC gain 1 the filter 
   is design such that the output kernel varies depending on the k0 and k1 ( for more details on these
   factors look at Fessler)
    
  \author Sanida Mustafovic
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ModifiedInverseAveragingImageFilterAll_H__
#define __stir_ModifiedInverseAveragingImageFilterAll_H__



#include "local/stir/ModifiedInverseAverigingArrayFilter.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DataProcessor.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/VectorWithOffset.h"
#include "stir/ProjData.h"
#include "local/stir/ArrayFilter2DUsingConvolution.h"
#include "local/stir/ArrayFilter3DUsingConvolution.h"

#include "stir/shared_ptr.h"




START_NAMESPACE_STIR

// TODO!! remove define

#define num_dimensions 3


template <typename elemT>
class ModifiedInverseAveragingImageFilterAll: 
public 
    RegisteredParsingObject<
        ModifiedInverseAveragingImageFilterAll<elemT>,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >
    >
{
 private:
  typedef
    RegisteredParsingObject<
        ModifiedInverseAveragingImageFilterAll<elemT>,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >
    >
    base_type;
public:  
  static const char * const registered_name;   

  //! Default constructor
   ModifiedInverseAveragingImageFilterAll();
 
   ModifiedInverseAveragingImageFilterAll(string proj_data_filename,
				  string attenuation_proj_data_filename,
				  const VectorWithOffset<elemT>& filter_coefficients,
				  shared_ptr<ProjData> proj_data_ptr,
				  shared_ptr<ProjData> attenuation_proj_data_ptr,
				  DiscretisedDensity<3,float>* initial_image,
				  DiscretisedDensity<3,float>* sensitivity_image,
				  DiscretisedDensity<3,float>* precomputed_coefficients_image,
				  DiscretisedDensity<3,float>* normalised_bck_image,
				  int mask_size, 
				  int num_dim);
			     
  
  				  
private: 

  int mask_size;
  vector<double> filter_coefficients_for_parsing;
  VectorWithOffset<float> filter_coefficients;
  
  shared_ptr<ProjData> proj_data_ptr;
  string proj_data_filename;

  shared_ptr<ProjData> attenuation_proj_data_ptr;
  string attenuation_proj_data_filename;

  DiscretisedDensity<3,float>* initial_image;
  string initial_image_filename;
  
  DiscretisedDensity<3,float>* sensitivity_image;
  string sensitivity_image_filename;

  DiscretisedDensity<3,float>* precomputed_coefficients_image ;
  string precomputed_coefficients_filename;

  DiscretisedDensity<3,float>* normalised_bck_image ;
  string normalised_bck_filename;
  int num_dim;

   Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions,elemT>& density);
   // new
   void virtual_apply(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
   void virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const;
   void precalculate_filter_coefficients (VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFilter3DUsingConvolution<float> >  > > >& all_filter_coefficients,
					  DiscretisedDensity<3,elemT>* in_density) const;

   void precalculate_filter_coefficients_2D (VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFilter2DUsingConvolution<float> >  > > >& all_filter_coefficients,
					  DiscretisedDensity<3,elemT>* in_density) const;

   void precalculate_filter_coefficients_separable(VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ModifiedInverseAverigingArrayFilter <3, float> >  > > >& all_filter_coefficients,
					  DiscretisedDensity<3,elemT>* in_density) const;
	

  virtual void set_defaults();
  virtual void initialise_keymap();

  virtual bool post_processing();

   mutable
   ModifiedInverseAverigingArrayFilter<num_dimensions,elemT> inverse_filter;
};


#undef num_dimensions

END_NAMESPACE_STIR

#endif


