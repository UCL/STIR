//
// $Id: 
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
      
  \date $Date: 
  \version $Revision: 
*/

#ifndef __Tomo_ModifiedInverseAverigingImageFilter_H__
#define __Tomo_ModifiedInverseAverigingImageFilter_H__


#include "local/tomo/ModifiedInverseAverigingArrayFilter.h"
#include "DiscretisedDensity.h"
#include "tomo/ImageProcessor.h"
#include "tomo/RegisteredParsingObject.h"
#include "VectorWithOffset.h"
#include "ProjData.h"




START_NAMESPACE_TOMO

// TODO!! remove define

#define num_dimensions 3


template <typename elemT>
class ModifiedInverseAverigingImageFilter: 
public 
    RegisteredParsingObject<
        ModifiedInverseAverigingImageFilter<elemT>,
        ImageProcessor<num_dimensions,elemT>
    >
{
public:  
  static const char * const registered_name; 
  

  //! Default constructor
   ModifiedInverseAverigingImageFilter();
 
   ModifiedInverseAverigingImageFilter(string proj_data_filename,
//				       string filename_kapa0,
//				       string filename_kapa1,
				 const VectorWithOffset<elemT>& filter_coefficients,
     shared_ptr<ProjData> proj_data_ptr);
//			    DiscretisedDensity<num_dimensions,elemT>* kapa0,
  //   DiscretisedDensity<num_dimensions,elemT>* kapa1);
 
  				  
private: 
  vector<double> filter_coefficients_for_parsing;
  VectorWithOffset<float> filter_coefficients;
  
  shared_ptr<ProjData> proj_data_ptr;
  
   // filenames for kapa0 and kapa1 that need to be parsed
   //string filename_kapa0;
   //string filename_kapa1;

   string proj_data_filename;

   //DiscretisedDensity<num_dimensions,elemT>* kapa0;
   //DiscretisedDensity<num_dimensions,elemT>* kapa1;  

   Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions,elemT>& density);
   // new
   void virtual_apply(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
   void virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const;

  virtual void set_defaults();
  virtual void initialise_keymap();

  virtual bool post_processing();

   mutable
   ModifiedInverseAverigingArrayFilter<num_dimensions,elemT> inverse_filter;
};


#undef num_dimensions

END_NAMESPACE_TOMO

#endif
