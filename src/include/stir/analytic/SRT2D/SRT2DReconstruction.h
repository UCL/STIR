//
// $Id: SRT2DReconstruction.h $
//
#ifndef __stir_analytic_SRT2D_SRT2DReconstruction_H__
#define __stir_analytic_SRT2D_SRT2DReconstruction_H__

 
#include "stir/recon_buildblock/AnalyticReconstruction.h"
#include <string>
#include <vector>
#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;
class ProjData;

class SRT2DReconstruction : public AnalyticReconstruction
{
  typedef AnalyticReconstruction base_type;
public:

  SRT2DReconstruction (); 

  explicit 
    SRT2DReconstruction(const string& parameter_filename);

 SRT2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, const float thres_restr_bound_v=-pow(10,6));
  
  virtual string method_info() const;

  virtual void ask_parameters();

  int num_segments_to_combine;
  
 protected: 

string attenuation_filename; 
float thres_restr_bound;
vector<double> thres_restr_bound_vector; 
shared_ptr<ProjData> atten_data_ptr;

 private:
  Succeeded actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & target_image_ptr);

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

 inline void spline(float x[],float y[],int n, float y2[]);
	float integ(float dist, int max, float ff[]);
float splint(float xa[], float ya[], float y2a[], int n, float x);
float hilbert_node(float x, float f[], float ddf[], float p[], int sp, float fn); 
float hilbert_derivative(float x, float f[], float ddf[], float p[], int sp);
void hilbert_der_double(float x, float f[], float ddf[], float f1[], float ddf1[], float p[], int sp, float *dhp, float *dh1p, float lg[]);
float hilbert(float x, float f[], float ddf[], float p[], int sp, float lg[]);

};


END_NAMESPACE_STIR

#endif

