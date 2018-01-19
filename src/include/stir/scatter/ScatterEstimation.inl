START_NAMESPACE_STIR

void
ScatterEstimation::
set_zoom_xy(float arg)
{
    this->zoom_xy = arg;
}

void
ScatterEstimation::
set_zoom_z(float arg)
{
    this->zoom_z = arg;
}


float
ScatterEstimation::
get_zoom_xy()
{
    return zoom_xy;
}

float
ScatterEstimation::
get_zoom_z()
{
    return zoom_z;
}

void
ScatterEstimation::
set_input_proj_data_sptr(const shared_ptr<ProjData> arg)
{
    this->input_projdata_sptr = arg;
}

void
ScatterEstimation::
set_reconstruction_method_sptr(const shared_ptr<Reconstruction < DiscretisedDensity < 3, float > > > arg)
{
    this->reconstruction_template_sptr = arg;
}

void
ScatterEstimation::
set_attenuation_image_sptr(const shared_ptr<DiscretisedDensity<3,float> > arg)
{
    this->atten_image_sptr = arg;
}

void
ScatterEstimation::
set_attenuation_correction_proj_data_sptr(const shared_ptr<ProjData> arg)
{
    //this->atten_projdata_sptr = arg;
}

void
ScatterEstimation::
set_normalisation_proj_data_sptr(const shared_ptr<ProjData> arg)
{
    //    this->norm_projdata_sptr = arg;
}

void
ScatterEstimation::
set_background_proj_data_sptr(const shared_ptr<ProjData> arg)
{
    //    this->back_projdata_sptr = arg;
}

void
ScatterEstimation::
set_initial_activity_image_sptr(const shared_ptr<DiscretisedDensity<3,float> > arg)
{
    this->current_activity_image_sptr = arg;
}

void
ScatterEstimation::
set_mask_image_sptr(const shared_ptr<DiscretisedDensity<3, float> > arg)
{
    this->mask_image_sptr = arg;
}

void
ScatterEstimation::
set_mask_proj_data_sptr(const shared_ptr<ProjData> arg)
{
    this->mask_projdata_sptr = arg;
}

void
ScatterEstimation::
set_scatter_simulation_method_sptr(const shared_ptr<ScatterSimulation > arg)
{
    this->scatter_simulation_sptr = arg;
}

END_NAMESPACE_STIR
