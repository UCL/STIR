#include "stir/scatter/ScatterEstimation.h"

START_NAMESPACE_STIR

void
ScatterEstimation::
set_input_proj_data_sptr(const shared_ptr<ProjData> arg)
{
  this->_already_setup = false;
    this->input_projdata_sptr = arg;
}

void
ScatterEstimation::
set_reconstruction_method_sptr(const shared_ptr<Reconstruction < DiscretisedDensity < 3, float > > > arg)
{
  this->_already_setup = false;
    this->reconstruction_template_sptr = arg;
}

void
ScatterEstimation::
set_attenuation_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> > arg)
{
  this->_already_setup = false;
    this->atten_image_sptr = arg;
}

void
ScatterEstimation::
set_background_proj_data_sptr(const shared_ptr<ProjData> arg)
{
  this->_already_setup = false;
    this->back_projdata_sptr = arg;
}

void
ScatterEstimation::
set_initial_activity_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> > arg)
{
  this->_already_setup = false;
    this->current_activity_image_sptr.reset(arg->clone());
}

void
ScatterEstimation::
set_mask_image_sptr(const shared_ptr<const DiscretisedDensity<3, float> > arg)
{
  this->_already_setup = false;
    this->mask_image_sptr = arg;
}

void
ScatterEstimation::
set_mask_proj_data_sptr(const shared_ptr<ProjData> arg)
{
  this->_already_setup = false;
    this->mask_projdata_sptr = arg;
}

void
ScatterEstimation::
set_scatter_simulation_method_sptr(const shared_ptr<ScatterSimulation > arg)
{
  this->_already_setup = false;
    this->scatter_simulation_sptr = arg;
}

void
ScatterEstimation::
set_num_iterations(int arg)
{
  this->num_scatter_iterations = arg;
}

END_NAMESPACE_STIR
