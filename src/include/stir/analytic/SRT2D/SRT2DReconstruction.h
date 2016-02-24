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


float thres_restr_bound;
vector<double> thres_restr_bound_vector; 

 private:
  Succeeded actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & target_image_ptr);

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

 inline void spline(float x[],float y[],int n, float y2[]);


};


END_NAMESPACE_STIR

#endif

