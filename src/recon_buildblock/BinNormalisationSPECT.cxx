//
//
/*
    Copyright (C) 2019, UCL
    Copyright (C) 2019, NPL
    This file is part of STIR.
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

#include "stir/recon_buildblock/BinNormalisationSPECT.h"
#include "stir/DetectionPosition.h"
#include "stir/DetectionPositionPair.h"
#include "stir/shared_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/ProjData.h"
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
using std::ostringstream;
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
  this->uniformity_filename = "";
  this->_use_detector_efficiencies = false;
  this->_use_dead_time = false;
  this->_use_uniformity_factors = false;
  this->num_detector_heads = 3;
  this->half_life = 6*60*60; //seconds
  this->resampled=0;

}

void 
BinNormalisationSPECT::
initialise_keymap()
{
  this->parser.add_start_key("Bin Normalisation SPECT");
  this->parser.add_key("uniformity_filename", &this->uniformity_filename);
  this->parser.add_key("use_detector_efficiencies", &this->_use_detector_efficiencies);
  this->parser.add_key("use_uniformity_factors", &this->_use_uniformity_factors);
  this->parser.add_key("folder_prefix", &this->folder_prefix);
  this->parser.add_key("rel_angle", &this->rel_angle);
  this->parser.add_key("half_life", &this->half_life);
  this->parser.add_key("view_time_interval", &this->view_time_interval);
  this->parser.add_key("num detector heads", &this->num_detector_heads);
  this->parser.add_key("projdata filename", &this->projdata_filename);
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

  norm_proj_data_info_ptr=ProjData::read_from_file(projdata_filename);
  max_tang=norm_proj_data_info_ptr->get_max_tangential_pos_num();
//  read_norm_data(normalisation_spect_filename);
  return false;
}


BinNormalisationSPECT::
BinNormalisationSPECT()
{
  set_defaults();
}

Succeeded
BinNormalisationSPECT::
set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr_v)
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
{// to think about this
  }

float BinNormalisationSPECT::get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const {
    int zoom=1024/(2*(max_tang+1));
    double normalisation=1;

    if(zoom!=1 && !resampled && use_uniformity_factors()){

        resample_uniformity(//down_sampled_uniformity,
                            uniformity,
                            max_tang,
                            zoom);
    }

    if(bin.view_num()==0)
    set_num_views(norm_proj_data_info_ptr->get_num_views());

    int head_num=(int)bin.view_num()/(num_views/num_detector_heads);
    double rel_time;
    rel_time=(this->view_time_interval)*
             (bin.view_num()+1-head_num*
             (num_views/num_detector_heads));
    /*####################################################################################################
     *####################################   uniformity factors  #########################################*/

                if (use_uniformity_factors()){
                    if(uniformity_filename=="")
                        error("You need to define the uniformity filename and the folder prefix");
                    if(zoom!=1)
                    normalisation=
                    normalisation*down_sampled_uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];
                    else{
                    normalisation=
                    normalisation*uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];}
                }
    /*####################################################################################################
     *####################################     decay factors     #########################################*/

                if (use_decay_correction_factors()){
                    normalisation=
                    normalisation/decay_correction_factor(half_life, rel_time);
                }
//std::cout<<"value"<<uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1]<<" "<<normalisation<<std::endl;
return normalisation;
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
                            if(uniformity_filename=="")
                                error("You need to define the uniformity filename and the folder prefix");
                            if(zoom!=1)
                                normalisation=normalisation*down_sampled_uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];
                            else
                                normalisation=normalisation*uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];
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
                if(uniformity_filename=="")
                    error("You need to define the uniformity filename and the folder prefix");
                if(zoom!=1)
                    normalisation=normalisation*down_sampled_uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];
                else
                    normalisation=normalisation*uniformity[head_num][bin.axial_pos_num()][bin.tangential_pos_num()+max_tang+1];
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
read_uniformity_table(Array<3,float>& uniformity) const
{//std::ofstream unif_table("uniformity.dat",std::ios::out);
    for(int n=1; n<=num_detector_heads; n++ ){
      
              const std::string n_string = boost::lexical_cast<std::string>(n);
              const std::string filename(this->folder_prefix+n_string+"/"+uniformity_filename);
              
              std::ifstream input(filename.c_str());
              
              if (!input)
                  error("Could not open Uniformity correction table!");
              input.read(const_cast<char *>(reinterpret_cast<char *>(&map)), sizeof(map));
              input.close();
              for(int j=1;j<=1023;j++)
                  for(int i=1;i<=1023;i++){
                      uniformity[n-1][j][i]=map[j+i*1024];
//                      std::cout<<"value"<<uniformity[n-1][j][i]<<std::endl;
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

END_NAMESPACE_STIR
