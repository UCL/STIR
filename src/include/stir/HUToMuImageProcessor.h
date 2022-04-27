/*!

  \file
  \ingroup ImageProcessor 
  \brief Declaration of class stir::HUToMuImageProcessor
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2020, UCL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_HUToMuImageProcessor_H__
#define __stir_HUToMuImageProcessor_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

/*!
  \ingroup ImageProcessor
  \brief A class in the DataProcessor hierarchy that convert from Hounsfield Units to mu-values

  This convert HU to mu-values using a piece-wise linear curve.
  Currently, it supports 2-segment piecewise linear transformations only.

  \warning This does not implement post-filtering to PET resolution, nor resampling to the PET voxel-size.

  \par Parsing from parameter file
The parameters specify a file in JSON format, and the parameters selecting the relevant record.
\verbatim
HUToMu Parameters:=
slope filename := json_filename
; next defaults to GENERIC
manufacturer_name := IN_CAPITALS
; CT tube voltage (defaults to 120)
kilovoltage_peak :=
; gamma energy (defaults to 511 for PET)
target_photon_energy :=
End HUToMu Parameters:=
\endverbatim

  \par Format of the slope filename

  This file is in JSON format. An example is distributed with STIR.
  The manufacturer name has to be in capitals. kvp and kev are matched after rounding.
\verbatim
{"scale": {
  "MANUFACTURER": {
    "type": "bilinear",
    "transform": [
      {
        "kvp": 120,
        "kev": 75,
        "a1": 0.16,
        "b1": 1.66e-4,
        "a2": 0.16,
        "b2": 1.48e-4,
        "break": 0
      },
      # more entries like the above
    ]
  }
}
}
\endverbatim
This implements the following transformation for every voxel in the image:
\f[
  \mu = a + b * \mathrm{HU}
\f]
with \f$a=a1, b=b1\f$ if \f$\mathrm{HI} < \mathrm{break}\f$, and $a2,b2$ otherwise.

When adding your own entries, you want avoid a discontinuity at the break point.  
*/
template <typename TargetT>
class HUToMuImageProcessor : 
  public 
    RegisteredParsingObject<
        HUToMuImageProcessor<TargetT>,
        DataProcessor<TargetT >,
        DataProcessor<TargetT >
    >
{
 private:
  typedef
    RegisteredParsingObject<
        HUToMuImageProcessor<TargetT>,
        DataProcessor<TargetT >,
        DataProcessor<TargetT >
    >
    base_type;
public:
  static constexpr const char * const registered_name = "HUToMu"; 
  
  //! Default constructor
  HUToMuImageProcessor();

  //! set the JSON filename with the slopes
  void set_slope_filename(const std::string& filename);
  //! set the manufacturer name used to select from the JSON entries
  void set_manufacturer_name(const std::string& name);
  //! set the CT kVp used to select from the JSON entries
  void set_kilovoltage_peak(const float kVp);
  //! set the gamma photon energy (in keV) used to select from the JSON entries
  void set_target_photon_energy(const float gamma_energy);

  //! same as apply
  void apply_scaling_to_HU(TargetT& output_image,
                           const TargetT& input_image) const;

#ifndef HAVE_JSON  // if we don't have JSON, we need another way to set the slope
  void set_slope(float a1, float a2, float b1, float b2, float breakPoint);
#endif
  
protected:

  // parsing functions
  //! sets default values
  /*! Sets \c manufacturer_name to "GENERIC", \c kilovoltage_peak to 120.F, \c target_photon_energy to 511.F
   */
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! just checks if all variables are set
  /*! \todo could get manufacturer name, kVp from the image later on, when these become available */
  Succeeded virtual_set_up(const TargetT& image);

  void  virtual_apply(TargetT& out_density, const TargetT& in_density) const;
  void  virtual_apply(TargetT& density) const ;

private:
  std::string filename;
  std::string manufacturer_name;
  float kilovoltage_peak;
  float target_photon_energy;

  // parameters for piecewise linear curve
  float a1;
  float b1;

  float a2;
  float b2;
  float breakPoint;
  
#ifdef HAVE_JSON
  void get_record_from_json();
#endif
};

END_NAMESPACE_STIR

#endif


