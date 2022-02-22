//
//
/*
 This file is part of STIR.
 
    Copyright (C) 2001- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2020, University College London
    Copyright (C) 2016-2017, PETsys Electronics
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
 \file
 \ingroup buildblock
 
 \brief Preliminary things for ML normalisation factor estimation

 Algorithms are described in

 1. Darren Hogg, Kris Thielemans, Terence J. Spinks, and Nicolas Spyrou. 2001.
 <cite>Maximum-Likelihood Estimation of Normalisation Factors for PET</cite>
 In 2001 IEEE Nuclear Science Symposium Conference Record, 4:2065-2069. San Diego, CA, USA: IEEE. https://doi.org/10.1109/nssmic.2001.1009231.

 2. Jacobson, Matthew W., and Kris Thielemans. 2008. 
 <cite>Optimizability of Loglikelihoods for the Estimation of Detector Efficiencies and Singles Rates in PET</cite>
 In 2008 IEEE Nuclear Science Symposium and Medical Imaging Conference (2008 NSS/MIC), 4580-4586. IEEE. https://doi.org/10.1109/nssmic.2008.4774352.

 3.  Tahereh Niknejad, Stefaan Tavernier, Joao Varela, and Kris Thielemans,
    <cite>Validation of 3D Model-Based Maximum-Likelihood Estimation of Normalisation Factors for Partial Ring Positron Emission Tomography</cite>
    In 2016 IEEE Nuclear Science Symposium, Medical Imaging Conference and Room-Temperature Semiconductor Detector Workshop (NSS/MIC/RTSD), 1-5.
    <a href="https://doi.org/10.1109/NSSMIC.2016.8069577">DOI: 10.1109/NSSMIC.2016.8069577</a>.
 
 \author Kris Thielemans
 \author Tahereh Niknejad
 
 */
#ifndef __stir_ML_norm_H__
#define __stir_ML_norm_H__


#include "stir/ProjData.h"
#include "stir/Array.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/IndexRange2D.h"
#include "stir/Sinogram.h"
#include <iostream>

START_NAMESPACE_STIR

typedef Array<2,float> GeoData;
typedef Array<2,float> BlockData;

class DetPairData : private Array<2,float>
{
public:
    
    DetPairData();
    DetPairData(const IndexRange<2>& range);
    DetPairData& operator=(const DetPairData&);
    
    float& operator()(const int a, const int b);
    
    float operator()(const int a, const int b) const;
    
    void fill(const float d);
    
    int get_min_index() const;
    int get_max_index() const;
    int get_min_index(const int a) const;
    int get_max_index(const int a) const;
    bool is_in_data(const int a, const int b) const;
    float sum() const;
    float sum(const int a) const;
    float find_max() const;
    float find_min() const;
    int get_num_detectors() const;
    void grow(const IndexRange<2>&);
    
private:
    typedef Array<2,float> base_type;
    int num_detectors;
};

//! Makes a DetPairData of appropriate dimensions and fills it with 0
void make_det_pair_data(DetPairData& det_pair_data,
                        const ProjDataInfo& proj_data_info,
                        const int segment_num,
                        const int ax_pos_num);
void make_det_pair_data(DetPairData& det_pair_data,
                        const ProjData& proj_data,
                        const int segment_num,
                        const int ax_pos_num);
void set_det_pair_data(ProjData& proj_data,
                       const DetPairData& det_pair_data,
                       const int segment_num,
                       const int ax_pos_num);
void apply_block_norm(DetPairData& det_pair_data,
                      const BlockData& geo_data,
                      const bool apply= true);
void apply_geo_norm(DetPairData& det_pair_data,
                    const GeoData& geo_data,
                    const bool apply= true);
void apply_efficiencies(DetPairData& det_pair_data,
                        const Array<1,float>& efficiencies,
                        const bool apply=true);


void make_fan_sum_data(Array<1,float>& data_fan_sums, const DetPairData& det_pair_data);

void make_geo_data(GeoData& geo_data, const DetPairData& det_pair_data);

void make_block_data(BlockData& block_data, const DetPairData& det_pair_data);

void iterate_efficiencies(Array<1,float>& efficiencies,
                          const Array<1,float>& data_fan_sums,
                          const DetPairData& model);

void iterate_geo_norm(GeoData& norm_geo_data,
                      const GeoData& measured_geo_data,
                      const DetPairData& model);

void iterate_block_norm(BlockData& norm_block_data,
                        const BlockData& measured_block_data,
                        const DetPairData& model);

//******3 D
typedef Array<2,float> DetectorEfficiencies;

class GeoData3D : public Array<4,float>
{
public:
    GeoData3D();
    GeoData3D(const int num_axial_crystals_per_block, const int half_num_transaxial_crystals_per_block, const int num_rings, const int num_detectors_per_ring);
    virtual  ~GeoData3D();
    GeoData3D& operator=(const GeoData3D&);

    
    float& operator()(const int ra, const int a, const int rb, const int b);
    
    float operator()(const int ra, const int a, const int rb, const int b) const;

    void fill(const float d);
    
    int get_min_ra() const;
    int get_max_ra() const;
    int get_min_a() const;
    int get_max_a() const;
    int get_min_b(const int a) const;
    int get_max_b(const int a) const;
    int get_min_rb(const int ra) const;
    int get_max_rb(const int ra) const;
    bool is_in_data(const int ra, const int a, const int rb, const int b) const;
    float sum() const;
    float sum(const int ra, const int a) const;
    float find_max() const;
    float find_min() const;
    int get_num_axial_crystals_per_block() const;
    int get_half_num_transaxial_crystals_per_block() const;

private:
    friend std::ostream& operator<<(std::ostream&, const GeoData3D&);
    friend std::istream& operator>>(std::istream&, GeoData3D&);
    typedef Array<4,float> base_type;
    int num_axial_crystals_per_block;
    int half_num_transaxial_crystals_per_block;
    int num_rings;
    int num_detectors_per_ring;
    
};

class FanProjData : private Array<4,float>
{
public:
    
    FanProjData();
    FanProjData(const int num_rings, const int num_detectors_per_ring, const int max_ring_diff, const int fan_size);
    virtual ~FanProjData();
    FanProjData& operator=(const FanProjData&);
    
    float& operator()(const int ra, const int a, const int rb, const int b);
    
    float operator()(const int ra, const int a, const int rb, const int b) const;
    
    void fill(const float d);
    
    int get_min_ra() const;
    int get_max_ra() const;
    int get_min_a() const;
    int get_max_a() const;
    int get_min_b(const int a) const;
    int get_max_b(const int a) const;
    int get_min_rb(const int ra) const;
    int get_max_rb(const int ra) const;
    bool is_in_data(const int ra, const int a, const int rb, const int b) const;
    float sum() const;
    float sum(const int ra, const int a) const;
    float find_max() const;
    float find_min() const;
    int get_num_detectors_per_ring() const;
    int get_num_rings() const;
    
    
private:
    friend std::ostream& operator<<(std::ostream&, const FanProjData&);
    friend std::istream& operator>>(std::istream&, FanProjData&);
    typedef Array<4,float> base_type;
    //FanProjData(const IndexRange<4>& range);
    //void grow(const IndexRange<4>&);
    int num_rings;
    int num_detectors_per_ring;
    int max_ring_diff;
    int half_fan_size;
};

typedef FanProjData BlockData3D;

shared_ptr<const ProjDataInfoCylindricalNoArcCorr>
get_fan_info(int& num_rings, int& num_detectors_per_ring, 
	     int& max_ring_diff, int& fan_size, 
	     const ProjDataInfo& proj_data_info);

shared_ptr<const ProjDataInfoBlocksOnCylindricalNoArcCorr>
get_fan_info_block(int& num_rings, int& num_detectors_per_ring,
             int& max_ring_diff, int& fan_size,
             const ProjDataInfo& proj_data_info);

void make_fan_data_remove_gaps(FanProjData& fan_data,
                   const ProjData& proj_data);

void set_fan_data_add_gaps(ProjData& proj_data,
                  const FanProjData& fan_data,
                  const float gap_value=0.F);

void apply_block_norm(FanProjData& fan_data,
                      const BlockData3D& block_data,
                      const bool apply= true);

void apply_geo_norm(FanProjData& fan_data,
                    const GeoData3D& geo_data,
                    const bool apply= true);


void apply_efficiencies(FanProjData& fan_data,
                        const DetectorEfficiencies& efficiencies,
                        const bool apply=true);

void make_fan_sum_data(Array<2,float>& data_fan_sums, const FanProjData& fan_data);

void make_fan_sum_data(Array<2,float>& data_fan_sums,
                       const ProjData& proj_data);

void make_fan_sum_data(Array<2,float>& data_fan_sums,
                       const DetectorEfficiencies& efficiencies,
                       const int max_ring_diff, const int half_fan_size);

void make_geo_data(GeoData3D& geo_data, const FanProjData& fan_data);


void make_block_data(BlockData3D& block_data, const FanProjData& fan_data);


void iterate_efficiencies(DetectorEfficiencies& efficiencies,
                          const Array<2,float>& data_fan_sums,
                          const FanProjData& model);

// version without model
void iterate_efficiencies(DetectorEfficiencies& efficiencies,
                          const Array<2,float>& data_fan_sums,
                          const int max_ring_diff, const int half_fan_size);

void iterate_geo_norm(GeoData3D& geo_data,
                      const GeoData3D& measured_geo_data,
                      const FanProjData& model);

void iterate_block_norm(BlockData3D& norm_block_data,
                        const BlockData3D& measured_block_data,
                        const FanProjData& model);

inline double KL(const double a, const double b, const double threshold_a = 0)
{
    assert(a>=0);
    assert(b>=0);
    double res = a<=threshold_a ? b : (a*(log(a)-log(b)) + b - a);
#ifndef NDEBUG
    if (res != res)
        warning("KL nan at a=%g b=%g, threshold %g\n",a,b,threshold_a);
    if (res > 1.E20)
        warning("KL large at a=%g b=%g, threshold %g\n",a,b,threshold_a);
#endif
    assert(res>=-1.e-4);
    return res;
}

template <int num_dimensions, typename elemT>
double KL(const Array<num_dimensions, elemT>& a, const Array<num_dimensions, elemT>& b, const double threshold_a = 0)
{
    double sum = 0;
    typename Array<num_dimensions, elemT>::const_full_iterator iter_a = a.begin_all();
    typename Array<num_dimensions, elemT>::const_full_iterator iter_b = b.begin_all();
    while (iter_a != a.end_all())
    {
        sum += static_cast<double>(KL(*iter_a++, *iter_b++, threshold_a));
    }
    return static_cast<double>(sum);
}

double KL(const DetPairData& d1, const DetPairData& d2, const double threshold);

double KL(const FanProjData& d1, const FanProjData& d2, const double threshold);

//double KL(const GeoData3D& d1, const GeoData3D& d2, const double threshold);

END_NAMESPACE_STIR
#endif
