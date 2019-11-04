//
//
/*
    Copyright (C) 2002- 2011, Hammersmith Imanet Ltd
    Copyright CTI
    This file is part of STIR.

    Some parts of this file originate in CTI code, distributed as
    part of the matrix library from Louvain-la-Neuve, and hence carries
    its restrictive license. Affected parts are the dead-time correction
    in get_dead_time_efficiency and geo_Z_corr related code.

    Most of this file is free software; you can redistribute that part and/or modify
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
  \ingroup recon_buildblock
  \ingroup

  \brief Implementation for class stir::BinNormalisationSPECT

  \author Kris Thielemans
  \author Daniel Deidda
*/

// enable if you want results identical to Peter Bloomfield's normalisation code
// (and hence old versions of Bkproj_3d)
// #define SAME_AS_PETER

#include "stir/recon_buildblock/BinNormalisationSPECT.h"
#include "stir/DetectionPosition.h"
#include "stir/DetectionPositionPair.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/IndexRange3D.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"
#include "stir/display.h"
#include "stir/is_null_ptr.h"
#include <algorithm>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
#endif

START_NAMESPACE_STIR


const char * const 
BinNormalisationSPECT::registered_name = "SPECT";

//
// helper functions used in this class.
//



void 
BinNormalisationSPECT::set_defaults()
{
  this->normalisation_spect_filename = "";
  this->_use_detector_efficiencies = false;
  this->_use_dead_time = false;
  this->_use_uniformity_factors = false;
  this->num_detector_heads = 3;
  this->half_life = 6*60*60; //seconds
  this->_use_cor_factors = false;

}

void 
BinNormalisationSPECT::
initialise_keymap()
{
  this->parser.add_start_key("Bin Normalisation SPECT");
  this->parser.add_key("normalisation_SPECT_filename", &this->normalisation_spect_filename);
  this->parser.add_key("use_detector_efficiencies", &this->_use_detector_efficiencies);
//  this->parser.add_key("use_COR_factors", &this->_use_cor_factors);
  this->parser.add_key("use_uniformity_factors", &this->_use_uniformity_factors);
  this->parser.add_key("folder_prefix", &this->folder_prefix);
  this->parser.add_key("rel_angle", &this->rel_angle);
  this->parser.add_key("half_life", &this->half_life);
  this->parser.add_key("view_time_interval", &this->view_time_interval);
  this->parser.add_key("num detector heads", &this->num_detector_heads);
  this->parser.add_key("use_decay_correction", &this->_use_decay_correction);

  this->parser.add_stop_key("End Bin Normalisation SPECT");
}

bool 
BinNormalisationSPECT::
post_processing()
{
  if(use_uniformity_factors()){
      uniformity.resize(IndexRange3D(0,2,0,1023,0,1023));
  read_uniformity_table(uniformity);}

  if(use_COR_factors()){
      cor.resize(IndexRange3D(0,2,0,719,0,719));
      read_cor_table(cor);
  }

  read_norm_data(normalisation_spect_filename);
  return false;
}


BinNormalisationSPECT::
BinNormalisationSPECT()
{
  set_defaults();
}

Succeeded
BinNormalisationSPECT::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v)
{
  return BinNormalisation::set_up(proj_data_info_ptr_v);
}

BinNormalisationSPECT::
BinNormalisationSPECT(const std::string& filename)
{
  read_norm_data(filename);
}

void
BinNormalisationSPECT::
read_norm_data(const std::string& filename)
{// to think about this here I would need to read each table for uniformity or cor
  }

void BinNormalisationSPECT::apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const{

    this->check(*viewgrams.get_proj_data_info_sptr());
    int view_num=viewgrams.get_basic_view_num();
    int max_tang=viewgrams.get_max_tangential_pos_num();
    int zoom=1024/(2*(max_tang+1));
    double normalisation=1;

    if(zoom!=1 && !resampled && use_uniformity_factors()){

        resample_uniformity(//down_sampled_uniformity,
                            uniformity,
                            max_tang,
                            zoom);
    }

    if(view_num==0)
    set_num_views(viewgrams.get_proj_data_info_sptr()->get_num_views());

    int head_num=(int)view_num/(num_views/num_detector_heads);

    double rel_time;
    rel_time=(this->view_time_interval)*
            (view_num+1-head_num*
            (num_views/num_detector_heads));

    for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
    {
      Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
      for (bin.axial_pos_num()= iter->get_min_axial_pos_num();
       bin.axial_pos_num()<=iter->get_max_axial_pos_num();
       ++bin.axial_pos_num())
        for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num();
         bin.tangential_pos_num()<=iter->get_max_tangential_pos_num();
         ++bin.tangential_pos_num()){

            /*####################################################################################################
             *####################################   uniformity factors  #########################################*/

                        if (use_uniformity_factors()){
                            if(zoom!=1)
                            normalisation=
                            normalisation*down_sampled_uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];
                            else
                            normalisation=
                            normalisation*uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];
                        }
            /*####################################################################################################
             *####################################     decay factors     #########################################*/

                        if (use_decay_correction_factors()){
                            normalisation=
                            normalisation/decay_correction_factor(half_life, rel_time);
                        }
          (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] /=
            (std::max(1.E-20F, get_bin_efficiency(bin, start_time, end_time))*
             normalisation);
           normalisation=1;
        }
    }
}

void BinNormalisationSPECT::undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const{

    this->check(*viewgrams.get_proj_data_info_sptr());
//    int head_to_sino_index=num_pixel_in_detector_head_row/
//        (viewgrams.get_max_tangential_pos_num()-viewgrams.get_min_tangential_pos_num()+1);
    int view_num=viewgrams.get_basic_view_num();
    int max_tang=viewgrams.get_max_tangential_pos_num();
    int zoom=1024/(2*(max_tang+1));
    double normalisation=1;
//    NCOR_viewgrams=viewgrams;
//    RelatedViewgrams<float>::iterator NCOR_iter=NCOR_viewgrams.begin();
//std::cout<<"uni "<<down_sampled_uniformity.empty()<<","<<zoom<<std::endl;
if(zoom!=1 && !resampled && use_uniformity_factors()){

    resample_uniformity(//down_sampled_uniformity,
                        uniformity,
                        max_tang,
                        zoom);
}

    if(view_num==0)
    set_num_views(viewgrams.get_proj_data_info_sptr()->get_num_views());

    int head_num=(int)view_num/(num_views/num_detector_heads);

    double rel_time;
    rel_time=(this->view_time_interval)*
            (view_num+1-head_num*
            (num_views/num_detector_heads));

    for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
    {//NCOR_iter=iter;
      Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
      for (bin.axial_pos_num()= iter->get_min_axial_pos_num();
       bin.axial_pos_num()<=iter->get_max_axial_pos_num();
       ++bin.axial_pos_num())
        for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num();
         bin.tangential_pos_num()<=iter->get_max_tangential_pos_num();
         ++bin.tangential_pos_num()){

/*####################################################################################################
 *####################################   uniformity factors  #########################################*/

            if (use_uniformity_factors()){
                if(zoom!=1)
                normalisation=
                normalisation*down_sampled_uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];
                else
                normalisation=
                normalisation*uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];
            }
/*####################################################################################################
 *####################################     decay factors     #########################################*/

            if (use_decay_correction_factors()){
                normalisation=
                normalisation/decay_correction_factor(half_life, rel_time);
            }


            (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()]*=
        (this->get_bin_efficiency(bin,start_time, end_time)*normalisation);
            normalisation=1;

        }
    }
}

void
BinNormalisationSPECT::
read_cor_table(Array<3,float>& cor) const
{//std::ofstream cor_table("cor.dat",std::ios::out);
    for(int n=1; n<=3; n++ ){
        std::ifstream input(this->folder_prefix+std::to_string(n)+"/COR_table.dat");
        std::string str;

        int i=0;

        while (std::getline(input, str)){
//            value=std::stof(str);
//            std::cout<<"uni "<<n<<", "<<i<<", "<<j<<", "
//                    <<value<<std::endl;
             input>>cor[n-1][i][0]>>cor[n-1][i][1];
//             std::cout<<"uni "<<n<<","<<i<<", "<<cor[n-1][i][0]<<cor[n-1][i][1]<<", "<<std::endl;
//             cor_table<<cor[n-1][i][0]<<" "<<cor[n-1][i][1]<<std::endl;
             i=i+1;
            }
    }
}


void
BinNormalisationSPECT::
read_uniformity_table(Array<3,float>& uniformity) const
{//std::ofstream unif_table("uniformity.dat",std::ios::out);
    for(int n=1; n<=3; n++ ){
        std::ifstream input(this->folder_prefix+std::to_string(n)+"/uniformity_table.dat");
        std::string str;
        float value;
        int i=0,j=0;

        while (std::getline(input, str)){
            value=std::stof(str);

            if(j>1023){
                j=0;
                i=i+1;
            }
            if(i>1023)
                i=0;
//            std::cout<<"uni "<<n<<", "<<i<<", "<<j<<", "
//                    <<value<<std::endl;
             uniformity[n-1][i][j]=value;
//             unif_table<<uniformity[n-1][i][j]<<std::endl;
             j=j+1;
            }
    }
}

void
BinNormalisationSPECT::
resample_uniformity(//Array<3,float>& down_sampled_uniformity,
                    Array<3,float> uniformity,
                    const int max_tang,
                    const int zoom) const
{
down_sampled_uniformity.resize(IndexRange3D(0, 2, 0, 2*max_tang+1, 0, 2*max_tang+1));
for(int n=0;n<=2;n++){
    for(int i=0;i<=2*max_tang+1;i++){
        for(int j=0;j<=2*max_tang+1;j++){
            for(int l=0;l<=zoom-1;l++){
                for(int k=0;k<=zoom-1;k++){// maybe resize uniformity

                    down_sampled_uniformity[n][i][j]=down_sampled_uniformity[n][i][j] +
                                            uniformity[n][zoom*i+l][zoom*j+k]/square(zoom);
//                    std::cout<<"uni"<<uniformity[n][zoom*i+l][zoom*j+k]/square(zoom)<<","
//                             <<down_sampled_uniformity[n][i][j]<<std::endl;

                }
            }
        }
    }
}
resampled=1;
//set_uniformity(down_sampled_uniformity);
}

bool 
BinNormalisationSPECT::
use_detector_efficiencies() const
{
  return this->_use_detector_efficiencies;
}

bool
BinNormalisationSPECT::
use_decay_correction_factors() const
{
  return this->_use_decay_correction;
}

bool 
BinNormalisationSPECT::
use_dead_time() const
{
  return this->_use_dead_time;
}

bool 
BinNormalisationSPECT::
use_uniformity_factors() const
{
  return this->_use_uniformity_factors;
}

double
BinNormalisationSPECT::
get_half_life() const
{
  return this->half_life;
}

bool
BinNormalisationSPECT::
use_COR_factors() const
{
  return this->_use_cor_factors;
}

END_NAMESPACE_STIR