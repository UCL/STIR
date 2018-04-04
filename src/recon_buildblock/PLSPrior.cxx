//
//
/*
    Copyright (C) 2018 University of Leeds and University College of London

    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup priors
  \brief  implementation of the stir::PLSPrior class

  \author Daniel Deidda
  \author Yu-Jung Tsai

*/

#include "stir/recon_buildblock/PLSPrior.h"
#include "stir/Succeeded.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/IO///write_to_file.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include <algorithm>
using std::min;
using std::max;

/* Pretty horrible code because we don't have an iterator of neigbhourhoods yet
 */

START_NAMESPACE_STIR

template <typename elemT>
void
PLSPrior<elemT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("PLS Prior Parameters");
  this->parser.add_key("only 2D", &only_2D);
  this->parser.add_key("eta", &eta);
  this->parser.add_key("alpha", &alpha);
  this->parser.add_key("kappa filename", &kappa_filename);
  this->parser.add_key("anatomical_filename", &anatomical_filename);
  this->parser.add_key("gradient filename prefix", &gradient_filename_prefix);
  this->parser.add_stop_key("END PLS Prior Parameters");
}


template <typename elemT>
Succeeded
PLSPrior<elemT>::set_up (shared_ptr<DiscretisedDensity<3,elemT> > const& target_sptr)
{
    base_type::set_up(target_sptr);

    if (is_null_ptr( this->anatomical_sptr))
    {
       error("PLS prior needs an anatomical image");
      return Succeeded::no;
    }

    shared_ptr<DiscretisedDensity<3,elemT> > anatomical_im_grad_z_sptr;
    if (!only_2D)
    anatomical_im_grad_z_sptr.reset(this->anatomical_sptr->get_empty_copy ());

    shared_ptr<DiscretisedDensity<3,elemT> > anatomical_im_grad_y_sptr(this->anatomical_sptr->get_empty_copy ());
    shared_ptr<DiscretisedDensity<3,elemT> > anatomical_im_grad_x_sptr(this->anatomical_sptr->get_empty_copy ());
    shared_ptr<DiscretisedDensity<3,elemT> > norm_sptr(this->anatomical_sptr->get_empty_copy ());

    if (!only_2D)
    compute_image_gradient_element ((*anatomical_im_grad_z_sptr),0,*this->anatomical_sptr);

    compute_image_gradient_element (*anatomical_im_grad_y_sptr,1,*this->anatomical_sptr);
    compute_image_gradient_element (*anatomical_im_grad_x_sptr,2,*this->anatomical_sptr);

    if (!only_2D)
    this->set_anatomical_grad_sptr (anatomical_im_grad_z_sptr,0);

    this->set_anatomical_grad_sptr (anatomical_im_grad_y_sptr,1);
    this->set_anatomical_grad_sptr (anatomical_im_grad_x_sptr,2);

    compute_normalisation_anatomical_gradient (*norm_sptr,*anatomical_im_grad_z_sptr,*anatomical_im_grad_y_sptr,*anatomical_im_grad_x_sptr );

    this->set_anatomical_grad_norm_sptr (shared_ptr<DiscretisedDensity<3,elemT> >(norm_sptr));

return Succeeded::yes;
}

template <typename elemT>
bool
PLSPrior<elemT>::post_processing()
{
  if (base_type::post_processing()==true)
    return true;

  if (kappa_filename.size() != 0)
    this->set_kappa_filename(kappa_filename);

  if (anatomical_filename.size() != 0)
      this->set_anatomical_filename(anatomical_filename);

  return false;

}

template <typename elemT>
void PLSPrior<elemT>::check(DiscretisedDensity<3,elemT> const& current_image_estimate) const
{
  // Do base-class check
  base_type::check(current_image_estimate);

  // Check anatomical and current image have same characteristics
  if (!this->anatomical_sptr->has_same_characteristics(current_image_estimate))
    error("The anatomical image must have the same charateristics as the PET image");
}

template <typename elemT>
void
PLSPrior<elemT>::set_defaults()
{
  base_type::set_defaults();
  this->only_2D = false;
  this->alpha=1;
  this->eta=1;
  this->kappa_ptr.reset();
}

template <>
const char * const
PLSPrior<float>::registered_name =
  "PLS";

template <typename elemT>
PLSPrior<elemT>::PLSPrior()
{
  set_defaults();
}


template <typename elemT>
PLSPrior<elemT>::PLSPrior(const bool only_2D_v, float penalisation_factor_v)
  :  only_2D(only_2D_v)
{
  set_defaults();
  this->penalisation_factor = penalisation_factor_v;
}

template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >
PLSPrior<elemT>::
get_anatomical_grad_sptr(int direction) const{

    if(direction==2){
        return this->anatomical_grad_x_sptr;}
    if(direction==1){
        return this->anatomical_grad_y_sptr;
    }
    if(direction==0){
        return this->anatomical_grad_z_sptr;
    }
    error(boost::format("PLSPrior::get_anatomical_grad_sptr called with out-of-range argument: %1%") % direction);
    // will never get here, but this avoids a compiler warning
    shared_ptr<DiscretisedDensity<3,elemT> > dummy;
    return dummy;
}

template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >
PLSPrior<elemT>::
get_norm_sptr () const{
return this->norm_sptr;
}

template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >
PLSPrior<elemT>::
get_anatomical_image_sptr() const{
return this->anatomical_sptr;
}

template <typename elemT>
double
PLSPrior<elemT>::
get_eta() const{
return this->eta;
}

template <typename elemT>
double
PLSPrior<elemT>::
get_alpha() const{
return this->alpha;
}

template <typename elemT>
void
PLSPrior<elemT>::
set_anatomical_image_sptr (const shared_ptr<DiscretisedDensity<3,elemT> >& arg)
{ this->anatomical_sptr = arg; }

template <typename elemT>
void
PLSPrior<elemT>::
set_eta (const double& arg)
{ this->eta = arg; }

template <typename elemT>
void
PLSPrior<elemT>::
set_alpha (const double& arg)
{ this->alpha = arg; }

template <typename elemT>
void
PLSPrior<elemT>::set_anatomical_grad_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >& arg, int direction){

    if(direction==2){
        this->anatomical_grad_x_sptr=arg;}
    if(direction==1){
        this->anatomical_grad_y_sptr=arg;
    }
    if(direction==0){
        this->anatomical_grad_z_sptr=arg;
    }
}

template <typename elemT>
void
PLSPrior<elemT>::set_anatomical_grad_norm_sptr (const shared_ptr<DiscretisedDensity<3,elemT> >& arg){


        this->norm_sptr=arg;
}

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >
PLSPrior<elemT>::
get_kappa_sptr() const
{ return this->kappa_ptr; }

//! set kappa image
template <typename elemT>
void
PLSPrior<elemT>::
set_kappa_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >& k)
{ this->kappa_ptr = k; }

//! Set kappa filename
template <typename elemT> void PLSPrior<elemT>::set_kappa_filename(const std::string& filename)
{
  kappa_filename  = filename;
  this->kappa_ptr = read_from_file<DiscretisedDensity<3,elemT> >(kappa_filename);
  info(boost::format("Reading kappa data '%1%'") % kappa_filename  );
  kappa_filename = ""; // Clear at end
}

//! Set anatomical filename
template <typename elemT> void PLSPrior<elemT>::set_anatomical_filename(const std::string& filename)
{
  anatomical_filename  = filename;
  this->anatomical_sptr = read_from_file<DiscretisedDensity<3,elemT> >(anatomical_filename);
  info(boost::format("Reading anatomical data '%1%'") % anatomical_filename  );
  anatomical_filename = ""; // Clear at end
}

template <typename elemT>
void PLSPrior<elemT>::compute_image_gradient_element(DiscretisedDensity<3,elemT> & image_gradient_elem, int direction, const DiscretisedDensity<3,elemT> & image ){
//std::cout<<"dentro ="<<direction<<std::endl;

    const int min_z = image.get_min_index();
    const int max_z = image.get_max_index();


        for (int z=min_z; z<=max_z; z++)
          {

            const int min_y = image[z].get_min_index();
            const int max_y = image[z].get_max_index();



              for (int y=min_y;y<= max_y;y++)
                {

                  const int min_x = image[z][y].get_min_index();
                  const int max_x = image[z][y].get_max_index();



                    for (int x=min_x;x<= max_x;x++)
                    {

                        if(direction==0){
                            if(z+1>max_z)
                                continue;
                       image_gradient_elem[z][y][x]=image[z+1][y][x]- image[z][y][x];

                        }
                        if(direction==1){
                            if(y+1>max_y)
                                continue;
                       image_gradient_elem[z][y][x]=image[z][y+1][x]- image[z][y][x];
//                       std::cout<<"grady ="<<image[z][y+1][x]- image[z][y][x]<<std::endl;
                        }
                        if(direction==2){
                            if(x+1>max_x )
                                continue;
                       image_gradient_elem[z][y][x]=image[z][y][x+1]- image[z][y][x];
                        }

                    }
                 }
          }

}

template <typename elemT>
void
PLSPrior<elemT>::compute_normalisation_anatomical_gradient(DiscretisedDensity<3,elemT> &norm_im_grad,
                                          const DiscretisedDensity<3,elemT> &image_grad_z,
                                          const DiscretisedDensity<3,elemT> &image_grad_y,
                                          const DiscretisedDensity<3,elemT> &image_grad_x){

    const int min_z = image_grad_x.get_min_index();
    const int max_z = image_grad_x.get_max_index();


        for (int z=min_z; z<=max_z; z++)
          {

            const int min_y = image_grad_x[z].get_min_index();
            const int max_y = image_grad_x[z].get_max_index();



              for (int y=min_y;y<= max_y;y++)
                {

                  const int min_x = image_grad_x[z][y].get_min_index();
                  const int max_x = image_grad_x[z][y].get_max_index();



                    for (int x=min_x;x<= max_x;x++)
                    {
                        if(only_2D){
                     norm_im_grad[z][y][x]   = sqrt (square(image_grad_y[z][y][x]) + square(image_grad_x[z][y][x]) + square(this->eta));}
                        else{
                            norm_im_grad[z][y][x]   = sqrt (square(image_grad_z[z][y][x]) + square(image_grad_y[z][y][x]) +
                                                            square(image_grad_x[z][y][x]) + square(this->eta));

                        }
                    }

                 }
          }
}

template <typename elemT>
void
PLSPrior<elemT>::
compute_inner_product_and_penalty(DiscretisedDensity<3,elemT> &inner_product,
                                  DiscretisedDensity<3,elemT> &penalty,
                                  DiscretisedDensity<3,elemT> &pet_im_grad_z,
                                  DiscretisedDensity<3,elemT> &pet_im_grad_y,
                                  DiscretisedDensity<3,elemT> &pet_im_grad_x,
                      const DiscretisedDensity<3,elemT> &pet_image){



    if(!only_2D)
    compute_image_gradient_element (pet_im_grad_z,0,pet_image);

    compute_image_gradient_element (pet_im_grad_y,1,pet_image);
    compute_image_gradient_element (pet_im_grad_x,2,pet_image);


    const int min_z = pet_image.get_min_index();
    const int max_z = pet_image.get_max_index();


        for (int z=min_z; z<=max_z; z++)
          {

            const int min_y = pet_image[z].get_min_index();
            const int max_y = pet_image[z].get_max_index();



              for (int y=min_y;y<= max_y;y++)
                {

                  const int min_x = pet_image[z][y].get_min_index();
                  const int max_x = pet_image[z][y].get_max_index();



                    for (int x=min_x;x<= max_x;x++)
                    {
                        if(only_2D){
                            inner_product[z][y][x]   = ((pet_im_grad_y[z][y][x]*(*anatomical_grad_y_sptr)[z][y][x]/(*get_norm_sptr())[z][y][x]) +
                                                        (pet_im_grad_x[z][y][x]*(*anatomical_grad_x_sptr)[z][y][x]/(*get_norm_sptr())[z][y][x]));

                            penalty[z][y][x]= sqrt (square(this->alpha) + square(pet_im_grad_y[z][y][x]) +
                                                                              square(pet_im_grad_x[z][y][x]) -
                                                                              square(inner_product[z][y][x]));
                        }
                        else{
                     inner_product[z][y][x]   = (pet_im_grad_z[z][y][x]*(*anatomical_grad_z_sptr)[z][y][x] +
                                                 pet_im_grad_y[z][y][x]*(*anatomical_grad_y_sptr)[z][y][x] +
                                                 pet_im_grad_x[z][y][x]*(*anatomical_grad_x_sptr)[z][y][x])/(*get_norm_sptr())[z][y][x];

                     penalty[z][y][x]= sqrt (square(this->alpha) + square(pet_im_grad_z[z][y][x]) +
                                                                       square(pet_im_grad_y[z][y][x]) +
                                                                       square(pet_im_grad_x[z][y][x]) -
                                                                       square(inner_product[z][y][x]));
                        }
                    }
                 }
          }

}

template <typename elemT>
double
PLSPrior<elemT>::
compute_value(const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  if (this->penalisation_factor==0)
  {
    return 0.;
  }

  this->check(current_image_estimate);

  shared_ptr<DiscretisedDensity<3,elemT> > pet_im_grad_z_sptr;
  if(!only_2D)
  pet_im_grad_z_sptr.reset(this->anatomical_sptr->get_empty_copy ());

  shared_ptr<DiscretisedDensity<3,elemT> > pet_im_grad_y_sptr(this->anatomical_sptr->get_empty_copy ());
  shared_ptr<DiscretisedDensity<3,elemT> > pet_im_grad_x_sptr(this->anatomical_sptr->get_empty_copy ());

  shared_ptr<DiscretisedDensity<3,elemT> > inner_product_sptr(this->anatomical_sptr.get ()->get_empty_copy ());
  shared_ptr<DiscretisedDensity<3,elemT> > penalty_sptr(this->anatomical_sptr.get ()->get_empty_copy ());

  compute_inner_product_and_penalty (*inner_product_sptr,
                                     *penalty_sptr,
                                     *pet_im_grad_z_sptr,
                                     *pet_im_grad_y_sptr,
                                     *pet_im_grad_x_sptr,
                                     current_image_estimate);

  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("PLSPrior: kappa image has not the same index range as the reconstructed image\n");


  double result = 0.;
  const int min_z = current_image_estimate.get_min_index();
  const int max_z = current_image_estimate.get_max_index();
  for (int z=min_z; z<=max_z; z++)
    {

      const int min_y = current_image_estimate[z].get_min_index();
      const int max_y = current_image_estimate[z].get_max_index();

        for (int y=min_y;y<= max_y;y++)
          {

            const int min_x = current_image_estimate[z][y].get_min_index();
            const int max_x = current_image_estimate[z][y].get_max_index();

            for (int x=min_x;x<= max_x;x++)
              {

                /* formula:
                  sum_x,y,z

                   (penalty[z][y][x]) * (*kappa_ptr)[z][y][x];
                */

                        elemT current = (*penalty_sptr)[z][y][x];

                        if (do_kappa)
                          current *=
                            (*kappa_ptr)[z][y][x] ;

                        result += static_cast<double>(current);

              }
          }
    }
  return result * this->penalisation_factor;
}

template <typename elemT>
void
PLSPrior<elemT>::
compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient,
                 const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  this->check(current_image_estimate);

  if (this->penalisation_factor==0)
  {
    prior_gradient.fill(0);
    return;
  }

  shared_ptr<DiscretisedDensity<3,elemT> > pet_im_grad_z_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > gradientz_sptr;

  if(!only_2D){
  pet_im_grad_z_sptr.reset(this->anatomical_sptr->get_empty_copy ());
  gradientz_sptr.reset(this->anatomical_sptr->get_empty_copy ());
  }

  shared_ptr<DiscretisedDensity<3,elemT> > pet_im_grad_y_sptr(this->anatomical_sptr->get_empty_copy ());
  shared_ptr<DiscretisedDensity<3,elemT> > pet_im_grad_x_sptr(this->anatomical_sptr->get_empty_copy ());

  shared_ptr<DiscretisedDensity<3,elemT> > inner_product_sptr(this->anatomical_sptr->get_empty_copy ());
  shared_ptr<DiscretisedDensity<3,elemT> > penalty_sptr(this->anatomical_sptr->get_empty_copy ());

  shared_ptr<DiscretisedDensity<3,elemT> > gradienty_sptr(this->anatomical_sptr->get_empty_copy ());
  shared_ptr<DiscretisedDensity<3,elemT> > gradientx_sptr(this->anatomical_sptr->get_empty_copy ());

  compute_inner_product_and_penalty (*inner_product_sptr,
                                     *penalty_sptr,
                                     *pet_im_grad_z_sptr,
                                     *pet_im_grad_y_sptr,
                                     *pet_im_grad_x_sptr,
                                     current_image_estimate);


  const bool do_kappa = !is_null_ptr(kappa_ptr);
  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("PLSPrior: kappa image has not the same index range as the reconstructed image\n");
 shared_ptr<DiscretisedDensity<3,elemT> > gradient_sptr(this->anatomical_sptr->get_empty_copy ());

  const int min_z = current_image_estimate.get_min_index();
  const int max_z = current_image_estimate.get_max_index();

  for (int z=min_z; z<=max_z; z++)
    {

      const int min_y = current_image_estimate[z].get_min_index();
      const int max_y = current_image_estimate[z].get_max_index();

      for (int y=min_y;y<= max_y;y++)
        {


          const int min_x = current_image_estimate[z][y].get_min_index();
          const int max_x = current_image_estimate[z][y].get_max_index();

          for (int x=min_x;x<= max_x;x++)
            {

              if(x+1>max_x || y+1>max_y ||(z+1>max_z && !only_2D))
                  continue;

                /* formula:
                  sum_x,y,z
                   div * (pet_im_grad[z][y][x]-inner_product[z][y][x]*anatomical_im_grad[z][y][x]/(*get_norm_sptr ())[z][y][x])*
                   (*kappa_ptr)[z][y][x] /penalty[z][y][x];
                */

              if(only_2D){
                  (*gradientx_sptr)[z][y][x+1] =
                     (((*pet_im_grad_x_sptr)[z][y][x+1]-(*anatomical_grad_x_sptr)[z][y][x+1]*(*inner_product_sptr)[z][y][x+1]/
                          (*get_norm_sptr ())[z][y][x+1])/(*penalty_sptr)[z][y][x+1] -
                     (((*pet_im_grad_x_sptr)[z][y][x]-(*anatomical_grad_x_sptr)[z][y][x]*(*inner_product_sptr)[z][y][x]/(*get_norm_sptr ())[z][y][x])/
                      (*penalty_sptr)[z][y][x]) );

                  (*gradienty_sptr)[z][y+1][x] =
                      (((*pet_im_grad_y_sptr)[z][y+1][x]-(*anatomical_grad_y_sptr)[z][y+1][x]*(*inner_product_sptr)[z][y+1][x]/
                          (*get_norm_sptr ())[z][y+1][x])/(*penalty_sptr)[z][y+1][x] -
                      (((*pet_im_grad_y_sptr)[z][y][x]-(*anatomical_grad_y_sptr)[z][y][x]*(*inner_product_sptr)[z][y][x]/(*get_norm_sptr ())[z][y][x])/
                       (*penalty_sptr)[z][y][x]) );
              }
              else{

                  (*gradientx_sptr)[z][y][x+1] =
                          (((*pet_im_grad_x_sptr)[z][y][x+1]-(*anatomical_grad_x_sptr)[z][y][x+1]*(*inner_product_sptr)[z][y][x+1]/(*get_norm_sptr ())[z][y][x+1])/
                          (*penalty_sptr)[z][y][x+1] -
                          ((*pet_im_grad_x_sptr)[z][y][x]-(*anatomical_grad_x_sptr)[z][y][x]*(*inner_product_sptr)[z][y][x]/(*get_norm_sptr ())[z][y][x])/
                          (*penalty_sptr)[z][y][x]);

                  (*gradienty_sptr)[z][y+1][x] =
                          (((*pet_im_grad_y_sptr)[z][y+1][x]-(*anatomical_grad_y_sptr)[z][y+1][x]*(*inner_product_sptr)[z][y+1][x]/
                          (*get_norm_sptr ())[z][y+1][x])/(*penalty_sptr)[z][y+1][x] -
                          (((*pet_im_grad_y_sptr)[z][y][x]-(*anatomical_grad_y_sptr)[z][y][x]*(*inner_product_sptr)[z][y][x]/
                            (*get_norm_sptr ())[z][y][x])/(*penalty_sptr)[z][y][x]) );

                  (*gradientz_sptr)[z+1][y][x] =
                      (((*pet_im_grad_z_sptr)[z+1][y][x]-(*anatomical_grad_z_sptr)[z+1][y][x]*(*inner_product_sptr)[z+1][y][x]/
                          (*get_norm_sptr ())[z+1][y][x])/(*penalty_sptr)[z+1][y][x] -
                          (((*pet_im_grad_z_sptr)[z][y][x]-(*anatomical_grad_z_sptr)[z][y][x]*(*inner_product_sptr)[z][y][x]/
                            (*get_norm_sptr ())[z][y][x])/(*penalty_sptr)[z][y][x]) );
              }
              }}}

  for (int z=min_z; z<=max_z; z++)
    {

      const int min_y = current_image_estimate[z].get_min_index();
      const int max_y = current_image_estimate[z].get_max_index();

      for (int y=min_y;y<= max_y;y++)
        {


          const int min_x = current_image_estimate[z][y].get_min_index();
          const int max_x = current_image_estimate[z][y].get_max_index();

          for (int x=min_x;x<= max_x;x++)
            {
              if(only_2D){

              (*gradient_sptr)[z][y][x] = -((*gradienty_sptr)[z][y][x] + (*gradientx_sptr)[z][y][x]);

              }
              else{

              (*gradient_sptr)[z][y][x] = -((*gradientz_sptr)[z][y][x] + (*gradienty_sptr)[z][y][x] + (*gradientx_sptr)[z][y][x]);

              }

              if (do_kappa)
                  (*gradient_sptr)[z][y][x] *=
                  (*kappa_ptr)[z][y][x] ;



                        prior_gradient[z][y][x]= (*gradient_sptr)[z][y][x] * this->penalisation_factor;
            }}}

  info(boost::format("Prior gradient max %1%, min %2%\n") % prior_gradient.find_max() % prior_gradient.find_min());

  static int count = 0;
  ++count;
  if (gradient_filename_prefix.size()>0)
    {
      char *filename = new char[gradient_filename_prefix.size()+100];
      sprintf(filename, "%s%d.v", gradient_filename_prefix.c_str(), count);
      write_to_file(filename, prior_gradient);
      delete[] filename;
    }
}


#  ifdef _MSC_VER
// prevent warning message on reinstantiation,
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


template class PLSPrior<float>;

END_NAMESPACE_STIR

