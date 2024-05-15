//
// $Id: SRT2DSPECTReconstruction.h $
//
#ifndef __stir_analytic_SRT2DSPECT_SRT2DSPECTReconstruction_H__
#define __stir_analytic_SRT2DSPECT_SRT2DSPECTReconstruction_H__

//author Dimitra Kyriakopoulou
 
#include "stir/analytic/SRT2DSPECT/SRT2DSPECTReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h" 
#include "stir/ArcCorrection.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Array.h"
#include <vector>   
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include <math.h>
#include "stir/Bin.h" 
#include "stir/round.h"  
#include "stir/display.h"
#include <algorithm>
#include "stir/IO/interfile.h"
#include "stir/info.h"
#include <boost/format.hpp>


#include "stir/recon_buildblock/AnalyticReconstruction.h"
#include "stir/RegisteredParsingObject.h"
#include <string>
#include "stir/shared_ptr.h"
#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;
class ProjData;

class SRT2DSPECTReconstruction :
        public
            RegisteredParsingObject<
                SRT2DSPECTReconstruction,
                    Reconstruction < DiscretisedDensity < 3,float> >,
                    AnalyticReconstruction
                 >
{
  //typedef AnalyticReconstruction base_type;
    typedef
    RegisteredParsingObject<
        SRT2DSPECTReconstruction,
            Reconstruction < DiscretisedDensity < 3,float> >,
            AnalyticReconstruction
         > base_type;
#ifdef SWIG
  // work-around swig problem. It gets confused when using a private (or protected)
  // typedef in a definition of a public typedef/member
 public:
#else  
 private: 
#endif  
    typedef DiscretisedDensity < 3,float> TargetT;
public:
    //! Name which will be used when parsing a ProjectorByBinPair object
    static const char * const registered_name;

  //! Default constructor (calls set_defaults())
  SRT2DSPECTReconstruction (); 
  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit  
    SRT2DSPECTReconstruction(const std::string& parameter_filename);

    SRT2DSPECTReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, const int num_segments_to_combine=-1, const int filter_wiener=0, const int filter_median=0, const int filter_gamma=0);


  virtual std::string method_info() const;

  virtual void ask_parameters(); 
 
  virtual Succeeded set_up(shared_ptr <TargetT > const& target_data_sptr);

 protected: // make parameters protected such that doc shows always up in doxygen
  // parameters used for parsing

  //! number of segments to combine (with SSRB) before starting 2D reconstruction
  /*! if -1, a value is chosen depending on the axial compression.
      If there is no axial compression, num_segments_to_combine is
      effectively set to 3, otherwise it is set to 1.
      \see SSRB
  */
  int num_segments_to_combine;
  //! potentially display data
  /*! allowed values: \c display_level=0 (no display), 1 (only final image), 
      2 (filtered-viewgrams). Defaults to 0.
   */
  string attenuation_filename; 
  int display_level;
  int filter_wiener; 
  int filter_median; 
  int filter_gamma;
	float thres_restr_bound;
	std::vector<double> thres_restr_bound_vector;  
            shared_ptr<ProjData> atten_data_ptr;
private:
  Succeeded actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & target_image_ptr);

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing(); 

float hilbert_node(float x, const std::vector<float>& f, const std::vector<float>& ddf, const std::vector<float>& p, int sp, float fn);
float hilbert(float x, const std::vector<float>& f, const std::vector<float>& ddf, const std::vector<float>& p, int sp, std::vector<float>& lg);
void hilbert_der_double(float x, const std::vector<float>& f, const std::vector<float>& ddf, const std::vector<float>& f1, const std::vector<float>& ddf1, const std::vector<float>& p, int sp, float* dhp, float* dh1p, const std::vector<float>& lg);
float splint(const std::vector<float>& xa, const std::vector<float>& ya, const std::vector<float>& y2a, int n, float x);
void spline(const std::vector<float>& x, const std::vector<float>& y, int n, std::vector<float>& y2);
float integ(float dist, int max, float ff[]);

void wiener(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa); 
void median(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa); 
void gamma(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa);
};


END_NAMESPACE_STIR

#endif
