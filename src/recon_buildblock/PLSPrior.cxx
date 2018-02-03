//
//
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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
  \author Tsai Yu Jung

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
  this->parser.add_key("scale_par", &scale_par);
  this->parser.add_key("smooth_par", &smooth_par);
  this->parser.add_key("kappa filename", &kappa_filename);
  this->parser.add_key("anatomical_filename", &anatomical_filename);
  this->parser.add_key("weights", &weights);
  this->parser.add_key("gradient filename prefix", &gradient_filename_prefix);
  this->parser.add_stop_key("END PLS Prior Parameters");
}


template <typename elemT>
bool
PLSPrior<elemT>::post_processing()
{
  if (base_type::post_processing()==true)
    return true;
  if (kappa_filename.size() != 0)
    this->kappa_ptr = read_from_file<DiscretisedDensity<3,elemT> >(kappa_filename);

  if (anatomical_filename.size() != 0){
    this->anatomical_sptr = read_from_file<DiscretisedDensity<3,elemT> >(anatomical_filename);}
  else{
      this->anatomical_sptr->fill (0);
  }


  if (this->anatomical_filename != "0"){
      info(boost::format("Reading MR data '%1%'")
           % anatomical_filename  );}

  bool warn_about_even_size = false;

  DiscretisedDensity<3,elemT> &anat_im_grad_z=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &anat_im_grad_y=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &anat_im_grad_x=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &norm=*this->anatomical_sptr.get ()->get_empty_copy ();

  compute_image_gradient_element (anat_im_grad_z,"z",*this->anatomical_sptr.get ());
//  std::cout<<"calc gradz ="<<std::endl;
  compute_image_gradient_element (anat_im_grad_y,"y",*this->anatomical_sptr.get ());
  compute_image_gradient_element (anat_im_grad_x,"x",*this->anatomical_sptr.get ());

  this->set_anat_grad_sptr (shared_ptr<DiscretisedDensity<3,elemT> >(anat_im_grad_z.clone()),"z");
  this->set_anat_grad_sptr (shared_ptr<DiscretisedDensity<3,elemT> >(anat_im_grad_y.clone()),"y");
  this->set_anat_grad_sptr (shared_ptr<DiscretisedDensity<3,elemT> >(anat_im_grad_x.clone()),"x");
  //write_to_file("agrady", anat_im_grad_y);
  //write_to_file("agradx", anat_im_grad_x);
  compute_normalis_image_gradient (norm,anat_im_grad_x,anat_im_grad_y,anat_im_grad_z );
 //write_to_file("norm", norm);
  this->set_anat_grad_norm_sptr (shared_ptr<DiscretisedDensity<3,elemT> >(norm.clone()));


  if (this->weights.size() ==0)
    {
      // will call compute_weights() to fill it in
    }
  else
    {
      if (!this->weights.is_regular())
        {
          warning("Sorry. PLSPrior currently only supports regular arrays for the weights");
          return true;
        }

      const unsigned int size_z = this->weights.size();
      if (size_z%2==0)
        warn_about_even_size = true;
      const int min_index_z = -static_cast<int>(size_z/2);
      this->weights.set_min_index(min_index_z);

      for (int z = min_index_z; z<= this->weights.get_max_index(); ++z)
        {
          const unsigned int size_y = this->weights[z].size();
          if (size_y%2==0)
            warn_about_even_size = true;
          const int min_index_y = -static_cast<int>(size_y/2);
          this->weights[z].set_min_index(min_index_y);
          for (int y = min_index_y; y<= this->weights[z].get_max_index(); ++y)
            {
              const unsigned int size_x = this->weights[z][y].size();
              if (size_x%2==0)
                warn_about_even_size = true;
              const int min_index_x = -static_cast<int>(size_x/2);
              this->weights[z][y].set_min_index(min_index_x);
            }
        }
    }

  if (warn_about_even_size)
    warning("Parsing PLSPrior: even number of weights occured in either x,y or z dimension.\n"
            "I'll (effectively) make this odd by appending a 0 at the end.");
  return false;

}

template <typename elemT>
void
PLSPrior<elemT>::set_defaults()
{
  base_type::set_defaults();
  this->only_2D = false;
  this->smooth_par=1;
  this->scale_par=1;
  this->kappa_ptr.reset();
  this->weights.recycle();
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
  this->penalisation_factor = penalisation_factor_v;
}


  //! get penalty weights for the neigbourhood
template <typename elemT>
Array<3,float>
PLSPrior<elemT>::
get_weights() const
{ return this->weights; }

template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >
PLSPrior<elemT>::
get_anat_grad_sptr(std::string direction) const{

    if(direction=="x"){
        return this->anat_grad_x_sptr;}
    if(direction=="y"){
        return this->anat_grad_y_sptr;
    }
    if(direction=="z"){
        return this->anat_grad_z_sptr;
    }

}

template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >
PLSPrior<elemT>::
get_norm_sptr () const{
return this->norm_sptr;
}
//! set penalty weights for the neigbourhood
template <typename elemT>
void
PLSPrior<elemT>::
set_weights(const Array<3,float>& w)
{ this->weights = w; }

template <typename elemT>
void
PLSPrior<elemT>::set_anat_grad_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >& arg, std::string direction){

    if(direction=="x"){
        this->anat_grad_x_sptr=arg;}
    if(direction=="y"){
        this->anat_grad_y_sptr=arg;
    }
    if(direction=="z"){
        this->anat_grad_z_sptr=arg;
    }
}

template <typename elemT>
void
PLSPrior<elemT>::set_anat_grad_norm_sptr (const shared_ptr<DiscretisedDensity<3,elemT> >& arg){


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


// TODO move to set_up
// initialise to 1/Euclidean distance
static void
compute_weights(Array<3,float>& weights, const CartesianCoordinate3D<float>& grid_spacing, const bool only_2D)
{
  int min_dz, max_dz;
  if (only_2D)
    {
      min_dz = max_dz = 0;
    }
  else
    {
      min_dz = -1;
      max_dz = 1;
    }
  weights = Array<3,float>(IndexRange3D(min_dz,max_dz,-1,1,-1,1));
  for (int z=min_dz;z<=max_dz;++z)
    for (int y=-1;y<=1;++y)
      for (int x=-1;x<=1;++x)
        {
          if (z==0 && y==0 && x==0)
            weights[0][0][0] = 0;
          else
            {
              weights[z][y][x] =
                grid_spacing.x()/
                sqrt(square(x*grid_spacing.x())+
                     square(y*grid_spacing.y())+
                     square(z*grid_spacing.z()));
            }
        }
}

template <typename elemT>
void PLSPrior<elemT>::compute_image_gradient_element(DiscretisedDensity<3,elemT> & image_gradient_elem,std::string direction, const DiscretisedDensity<3,elemT> & image ){
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

                        if(direction=="z"){
                            if(z+1>max_z)
                                continue;
                       image_gradient_elem[z][y][x]=image[z+1][y][x]- image[z][y][x];

                        }
                        if(direction=="y"){
                            if(y+1>max_y)
                                continue;
                       image_gradient_elem[z][y][x]=image[z][y+1][x]- image[z][y][x];
//                       std::cout<<"grady ="<<image[z][y+1][x]- image[z][y][x]<<std::endl;
                        }
                        if(direction=="x"){
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
PLSPrior<elemT>::compute_normalis_image_gradient(DiscretisedDensity<3,elemT> &norm_im_grad,
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
                     norm_im_grad[z][y][x]   = sqrt (square(image_grad_z[z][y][x]) + square(image_grad_y[z][y][x]) + square(image_grad_x[z][y][x]) + square(this->scale_par));
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


    DiscretisedDensity<3,elemT> &anat_im_grad_z=*this->get_anat_grad_sptr ("z").get ();
    DiscretisedDensity<3,elemT> &anat_im_grad_y=*this->get_anat_grad_sptr ("y").get ();
    DiscretisedDensity<3,elemT> &anat_im_grad_x=*this->get_anat_grad_sptr ("x").get ();

    DiscretisedDensity<3,elemT> &norm=*this->anatomical_sptr.get ()->get_empty_copy ();

    compute_image_gradient_element (pet_im_grad_z,"z",pet_image);
    compute_image_gradient_element (pet_im_grad_y,"y",pet_image);
    compute_image_gradient_element (pet_im_grad_x,"x",pet_image);


    compute_normalis_image_gradient (norm,anat_im_grad_x,anat_im_grad_y,anat_im_grad_z );



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
                     inner_product[z][y][x]   = ((pet_im_grad_z[z][y][x]*anat_im_grad_z[z][y][x]/norm[z][y][x]) +
                                                 (pet_im_grad_y[z][y][x]*anat_im_grad_y[z][y][x]/norm[z][y][x]) +
                                                 (pet_im_grad_x[z][y][x]*anat_im_grad_x[z][y][x]/norm[z][y][x]));

                     penalty[z][y][x]= sqrt (this->scale_par*this->scale_par + pet_im_grad_z[z][y][x]*pet_im_grad_z[z][y][x] +
                                                                               pet_im_grad_y[z][y][x]*pet_im_grad_y[z][y][x] +
                                                                               pet_im_grad_x[z][y][x]*pet_im_grad_x[z][y][x] -
                                                inner_product[z][y][x]*inner_product[z][y][x]);
                    }
                 }
          }
        //write_to_file("penalty", penalty);
        //write_to_file("innerp", inner_product);
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


  DiscretisedDensity<3,elemT> &pet_im_grad_z=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &pet_im_grad_y=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &pet_im_grad_x=*this->anatomical_sptr.get ()->get_empty_copy ();

  DiscretisedDensity<3,elemT> &inner_product=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &penalty=*this->anatomical_sptr.get ()->get_empty_copy ();

  compute_inner_product_and_penalty (inner_product,
                                     penalty,
                                     pet_im_grad_z,
                                     pet_im_grad_y,
                                     pet_im_grad_x,
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

                        elemT current = penalty[z][y][x];

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
  assert(  prior_gradient.has_same_characteristics(current_image_estimate));
  if (this->penalisation_factor==0)
  {
    prior_gradient.fill(0);
    return;
  }

  DiscretisedDensity<3,elemT> &pet_im_grad_z=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &pet_im_grad_y=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &pet_im_grad_x=*this->anatomical_sptr.get ()->get_empty_copy ();

  DiscretisedDensity<3,elemT> &anat_im_grad_z=*this->get_anat_grad_sptr ("z").get ();
  DiscretisedDensity<3,elemT> &anat_im_grad_y=*this->get_anat_grad_sptr ("y").get ();
  DiscretisedDensity<3,elemT> &anat_im_grad_x=*this->get_anat_grad_sptr ("x").get ();
  DiscretisedDensity<3,elemT> &norm=*this->get_norm_sptr ().get ();

  DiscretisedDensity<3,elemT> &inner_product=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &penalty=*this->anatomical_sptr.get ()->get_empty_copy ();

  DiscretisedDensity<3,elemT> &gradientz=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &gradienty=*this->anatomical_sptr.get ()->get_empty_copy ();
  DiscretisedDensity<3,elemT> &gradientx=*this->anatomical_sptr.get ()->get_empty_copy ();

  compute_inner_product_and_penalty (inner_product,
                                     penalty,
                                     pet_im_grad_z,
                                     pet_im_grad_y,
                                     pet_im_grad_x,
                                     current_image_estimate);

//write_to_file("pgradx", pet_im_grad_x);
//write_to_file("current", current_image_estimate);
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("PLSPrior: kappa image has not the same index range as the reconstructed image\n");
 DiscretisedDensity<3,elemT> &gradient = *this->anatomical_sptr.get ()->get_empty_copy ();

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

              if(x+1>max_x || y+1>max_y ||(z+1>max_z && only_2D==0))
                  continue;

                /* formula:
                  sum_x,y,z
                   div * (pet_im_grad[z][y][x]-inner_product[z][y][x]*anat_im_grad[z][y][x]/norm[z][y][x])*
                   (*kappa_ptr)[z][y][x] /penalty[z][y][x];
                */

              if(only_2D==1){
                  gradientx[z][y][x+1] =
                     ((pet_im_grad_x[z][y][x+1]-anat_im_grad_x[z][y][x+1]*inner_product[z][y][x+1]/norm[z][y][x+1])/penalty[z][y][x+1] -
                     (pet_im_grad_x[z][y][x]-anat_im_grad_x[z][y][x]*inner_product[z][y][x]/norm[z][y][x])/penalty[z][y][x]);

                  gradienty[z][y+1][x] =
                      ((pet_im_grad_y[z][y+1][x]-anat_im_grad_y[z][y+1][x]*inner_product[z][y+1][x]/norm[z][y+1][x])/penalty[z][y+1][x] -
                      ((pet_im_grad_y[z][y][x]-anat_im_grad_y[z][y][x]*inner_product[z][y][x]/norm[z][y][x])/penalty[z][y][x]) );
              }
              else{

                  gradientx[z][y][x+1] =
                          ((pet_im_grad_x[z][y][x+1]-anat_im_grad_x[z][y][x+1]*inner_product[z][y][x+1]/norm[z][y][x+1])/penalty[z][y][x+1] -
                          (pet_im_grad_x[z][y][x]-anat_im_grad_x[z][y][x]*inner_product[z][y][x]/norm[z][y][x])/penalty[z][y][x]);

                  gradienty[z][y+1][x] =
                          ((pet_im_grad_y[z][y+1][x]-anat_im_grad_y[z][y+1][x]*inner_product[z][y+1][x]/norm[z][y+1][x])/penalty[z][y+1][x] -
                          ((pet_im_grad_y[z][y][x]-anat_im_grad_y[z][y][x]*inner_product[z][y][x]/norm[z][y][x])/penalty[z][y][x]) );

                  gradientz[z+1][y][x] =
                      ((pet_im_grad_z[z+1][y][x]-anat_im_grad_z[z+1][y][x]*inner_product[z+1][y][x]/norm[z][y+1][x])/penalty[z+1][y][x] -
                          ((pet_im_grad_y[z][y][x]-anat_im_grad_y[z][y][x]*inner_product[z][y][x]/norm[z][y][x])/penalty[z][y][x]) );
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

              gradient[z][y][x] = -(gradientz[z][y][x] + gradienty[z][y][x] + gradientx[z][y][x]);

//                        std::cout<<"grad ="<<gradient[z][y][x]<<std::endl;

              if (do_kappa)
                  gradient[z][y][x] *=
                  (*kappa_ptr)[z][y][x] ;



                        prior_gradient[z][y][x]= gradient[z][y][x] * this->penalisation_factor;
            }}}

//write_to_file("pgrad", prior_gradient);

  info(boost::format("Prior gradient max %1%, min %2%\n") % prior_gradient.find_max() % prior_gradient.find_min());

  static int count = 0;
  ++count;
  if (gradient_filename_prefix.size()>0)
    {
      char *filename = new char[gradient_filename_prefix.size()+100];
      sprintf(filename, "%s%d.v", gradient_filename_prefix.c_str(), count);
      //write_to_file(filename, prior_gradient);
      delete[] filename;
    }
}

template <typename elemT>
void
PLSPrior<elemT>::
compute_Hessian(DiscretisedDensity<3,elemT>& prior_Hessian_for_single_densel,
                const BasicCoordinate<3,int>& coords,
                const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  assert(  prior_Hessian_for_single_densel.has_same_characteristics(current_image_estimate));
  prior_Hessian_for_single_densel.fill(0);
  if (this->penalisation_factor==0)
  {
    return;
  }


  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);

  DiscretisedDensityOnCartesianGrid<3,elemT>& prior_Hessian_for_single_densel_cast =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,elemT> &>(prior_Hessian_for_single_densel);

  if (weights.get_length() ==0)
  {
    compute_weights(weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }


  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa && kappa_ptr->has_same_characteristics(current_image_estimate))
    error("PLSPrior: kappa image has not the same index range as the reconstructed image\n");

  const int z = coords[1];
  const int y = coords[2];
  const int x = coords[3];
  const int min_dz = max(weights.get_min_index(), prior_Hessian_for_single_densel.get_min_index()-z);
  const int max_dz = min(weights.get_max_index(), prior_Hessian_for_single_densel.get_max_index()-z);

  const int min_dy = max(weights[0].get_min_index(), prior_Hessian_for_single_densel[z].get_min_index()-y);
  const int max_dy = min(weights[0].get_max_index(), prior_Hessian_for_single_densel[z].get_max_index()-y);

  const int min_dx = max(weights[0][0].get_min_index(), prior_Hessian_for_single_densel[z][y].get_min_index()-x);
  const int max_dx = min(weights[0][0].get_max_index(), prior_Hessian_for_single_densel[z][y].get_max_index()-x);

  elemT diagonal = 0;
  for (int dz=min_dz;dz<=max_dz;++dz)
    for (int dy=min_dy;dy<=max_dy;++dy)
      for (int dx=min_dx;dx<=max_dx;++dx)
      {
        // dz==0,dy==0,dx==0 will have weight 0, so we can just include it in the loop
        elemT current =
          weights[dz][dy][dx];

        if (do_kappa)
          current *=
          (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];

        diagonal += current;
        prior_Hessian_for_single_densel_cast[z+dz][y+dy][x+dx] = -current*this->penalisation_factor;
      }

      prior_Hessian_for_single_densel[z][y][x]= diagonal * this->penalisation_factor;
}

template <typename elemT>
void
PLSPrior<elemT>::parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature,
                        const DiscretisedDensity<3,elemT> &current_image_estimate)
{

  assert( parabolic_surrogate_curvature.has_same_characteristics(current_image_estimate));
  if (this->penalisation_factor==0)
  {
    parabolic_surrogate_curvature.fill(0);
    return;
  }


  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);

  if (weights.get_length() ==0)
  {
    compute_weights(weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }

  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("PLSPrior: kappa image has not the same index range as the reconstructed image\n");

  const int min_z = current_image_estimate.get_min_index();
  const int max_z = current_image_estimate.get_max_index();
  for (int z=min_z; z<=max_z; z++)
    {
      const int min_dz = max(weights.get_min_index(), min_z-z);
      const int max_dz = min(weights.get_max_index(), max_z-z);

      const int min_y = current_image_estimate[z].get_min_index();
      const int max_y = current_image_estimate[z].get_max_index();

      for (int y=min_y;y<= max_y;y++)
        {
          const int min_dy = max(weights[0].get_min_index(), min_y-y);
          const int max_dy = min(weights[0].get_max_index(), max_y-y);

          const int min_x = current_image_estimate[z][y].get_min_index();
          const int max_x = current_image_estimate[z][y].get_max_index();
          for (int x=min_x;x<= max_x;x++)
            {
              const int min_dx = max(weights[0][0].get_min_index(), min_x-x);
              const int max_dx = min(weights[0][0].get_max_index(), max_x-x);

                elemT gradient = 0;
                for (int dz=min_dz;dz<=max_dz;++dz)
                  for (int dy=min_dy;dy<=max_dy;++dy)
                    for (int dx=min_dx;dx<=max_dx;++dx)
                      {
                        // 1 comes from omega = psi'(t)/t = 2*t/2t =1
                        elemT current =
                          weights[dz][dy][dx] *1;

                         if (do_kappa)
                          current *=
                            (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];

                        gradient += current;
                      }

                parabolic_surrogate_curvature[z][y][x]= gradient * this->penalisation_factor;
              }
          }
    }

  info(boost::format("parabolic_surrogate_curvature max %1%, min %2%\n") % parabolic_surrogate_curvature.find_max() % parabolic_surrogate_curvature.find_min());
  /*{
    static int count = 0;
    ++count;
    char filename[20];
    sprintf(filename, "normalised_gradient%d.v",count);
    write_basic_interfile(filename, parabolic_surrogate_curvature);
  }*/
}

template <typename elemT>
Succeeded
PLSPrior<elemT>::
add_multiplication_with_approximate_Hessian(DiscretisedDensity<3,elemT>& output,
                                            const DiscretisedDensity<3,elemT>& input) const
{
  // TODO this function overlaps enormously with parabolic_surrogate_curvature
  // the only difference is that parabolic_surrogate_curvature uses input==1

  assert( output.has_same_characteristics(input));
  if (this->penalisation_factor==0)
  {
    return Succeeded::yes;
  }

  DiscretisedDensityOnCartesianGrid<3,elemT>& output_cast =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,elemT> &>(output);

  if (weights.get_length() ==0)
  {
    compute_weights(weights, output_cast.get_grid_spacing(), this->only_2D);
  }

  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa && !kappa_ptr->has_same_characteristics(input))
    error("PLSPrior: kappa image has not the same index range as the reconstructed image\n");

  const int min_z = output.get_min_index();
  const int max_z = output.get_max_index();
  for (int z=min_z; z<=max_z; z++)
    {
      const int min_dz = max(weights.get_min_index(), min_z-z);
      const int max_dz = min(weights.get_max_index(), max_z-z);

      const int min_y = output[z].get_min_index();
      const int max_y = output[z].get_max_index();

      for (int y=min_y;y<= max_y;y++)
        {
          const int min_dy = max(weights[0].get_min_index(), min_y-y);
          const int max_dy = min(weights[0].get_max_index(), max_y-y);

          const int min_x = output[z][y].get_min_index();
          const int max_x = output[z][y].get_max_index();

          for (int x=min_x;x<= max_x;x++)
            {
              const int min_dx = max(weights[0][0].get_min_index(), min_x-x);
              const int max_dx = min(weights[0][0].get_max_index(), max_x-x);

                elemT result = 0;
                for (int dz=min_dz;dz<=max_dz;++dz)
                  for (int dy=min_dy;dy<=max_dy;++dy)
                    for (int dx=min_dx;dx<=max_dx;++dx)
                      {
                        elemT current =
                          weights[dz][dy][dx] * input[z+dz][y+dy][x+dx];

                         if (do_kappa)
                          current *=
                            (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];
                         result += current;
                      }

                output[z][y][x] += result * this->penalisation_factor;
            }
        }
    }
  return Succeeded::yes;
}

#  ifdef _MSC_VER
// prevent warning message on reinstantiation,
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


template class PLSPrior<float>;

END_NAMESPACE_STIR

