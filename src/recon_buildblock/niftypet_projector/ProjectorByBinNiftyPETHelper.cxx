//
//
/*!

  \file
  \ingroup projection

  \brief non-inline implementations for stir::ProjectorByBinNiftyPETHelper

  \author Richard Brown


*/
/*
    Copyright (C) 2019-2020, University College London
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

#include "stir/recon_buildblock/niftypet_projector/ProjectorByBinNiftyPETHelper.h"
#include <fstream>
#include <math.h>
#include <boost/format.hpp>
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/is_null_ptr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_array_functions.h"
#include "stir/ProjDataInMemory.h"
#include "def.h"
#include "driver_types.h"
#include "auxmath.h"
#include "prjb.h"
#include "prjf.h"
#include "recon.h"
#include "lmproc.h"
#include "scanner_0.h"
#include "rnd.h"

START_NAMESPACE_STIR

ProjectorByBinNiftyPETHelper::~ProjectorByBinNiftyPETHelper()
{
    delete [] _crs;
    delete [] _s2c;
    delete [] _li2rng;
    delete [] _li2sn;
    delete [] _li2nos;
}

static void delete_axialLUT(axialLUT *axlut_ptr)
{
    if (!axlut_ptr) return;
    delete [] axlut_ptr->li2rno;
    delete [] axlut_ptr->li2sn;
    delete [] axlut_ptr->li2nos;
    delete [] axlut_ptr->sn1_rno;
    delete [] axlut_ptr->sn1_sn11;
    delete [] axlut_ptr->sn1_ssrb;
    delete [] axlut_ptr->sn1_sn11no;
}

static void delete_txLUT(txLUTs *txluts_ptr)
{
    if (!txluts_ptr) return;
    free(txluts_ptr->s2cF);
    free(txluts_ptr->c2sF);
    free(txluts_ptr->cr2s);
    free(txluts_ptr->s2c);
    free(txluts_ptr->s2cr);
    free(txluts_ptr->aw2sn);
    free(txluts_ptr->aw2ali);
    free(txluts_ptr->crsr);
    free(txluts_ptr->msino);
    free(txluts_ptr->cij);
}

static
shared_ptr<Cnst> get_cnst(const Scanner &scanner, const bool cuda_verbose, const char cuda_device, const char span)
{
    shared_ptr<Cnst> cnt_sptr = MAKE_SHARED<Cnst>();

    cnt_sptr->DEVID = cuda_device; // device (GPU) ID.  allows choosing the device on which to perform calculations
    cnt_sptr->VERBOSE = cuda_verbose;

    if (scanner.get_type() == Scanner::Siemens_mMR) {
        if (!(span==0 || span==1 || span==11))
            throw std::runtime_error("ProjectorByBinNiftyPETHelper::getcnst() "
                                 "only spans 0, 1 and 11 supported for scanner type: " + scanner.get_name());

        cnt_sptr->A = NSANGLES; //sino angles
        cnt_sptr->W = NSBINS; // sino bins for any angular index
        cnt_sptr->aw = AW; //sino bins (active only)

        cnt_sptr->NCRS = nCRS; //number of crystals
        cnt_sptr->NCRSR = nCRSR; //reduced number of crystals by gaps
        cnt_sptr->NRNG = NRINGS;  //number of axial rings
        cnt_sptr->D = -1;  //number of linear indexes along Michelogram diagonals                         /*unknown*/
        cnt_sptr->Bt = -1; //number of buckets transaxially                                               /*unknown*/

        cnt_sptr->B = NBUCKTS; //number of buckets (total)
        cnt_sptr->Cbt = 32552; //number of crystals in bucket transaxially                                /*unknown*/
        cnt_sptr->Cba = 3; //number of crystals in bucket axially                                         /*unknown*/

        cnt_sptr->NSN1 = NSINOS; //number of sinos in span-1
        cnt_sptr->NSN11 = NSINOS11; //in span-11
        cnt_sptr->NSN64 = NRINGS*NRINGS; //with no MRD limit

        cnt_sptr->SPN = span; //span-1 (s=1) or span-11 (s=11, default) or SSRB (s=0)
        cnt_sptr->NSEG0 = SEG0;

        cnt_sptr->RNG_STRT = 0;
        cnt_sptr->RNG_END  = NRINGS;

        cnt_sptr->TGAP = 9; // get the crystal gaps right in the sinogram, period and offset given      /*unknown*/
        cnt_sptr->OFFGAP = 1;                                                                           /*unknown*/

        cnt_sptr->NSCRS = 21910;  // number of scatter crystals used in scatter estimation              /*unknown*/
        std::vector<short> sct_irng = {0, 10, 19, 28, 35, 44, 53, 63}; // scatter ring definition
        cnt_sptr->NSRNG = int(sct_irng.size());
        cnt_sptr->MRD = mxRD; // maximum ring difference

        cnt_sptr->ALPHA = aLPHA; //angle subtended by a crystal
        float R = 32.8f; // ring radius
        cnt_sptr->RE = R + 0.67f; // effective ring radius accounting for the depth of interaction
        cnt_sptr->AXR = SZ_RING; //axial crystal dim

        cnt_sptr->COSUPSMX = 0.725f; //cosine of max allowed scatter angle
        cnt_sptr->COSSTP = (1-cnt_sptr->COSUPSMX)/(255);; //cosine step

        cnt_sptr->TOFBINN = 1; // number of TOF bins
        cnt_sptr->TOFBINS = 3.9e-10f; // size of TOF bin in [ps]
        float CLGHT = 29979245800.f; // speed of light [cm/s]
        cnt_sptr->TOFBIND = cnt_sptr->TOFBINS * CLGHT; // size of TOF BIN in cm of travelled distance
        cnt_sptr->ITOFBIND = 1.f / cnt_sptr->TOFBIND; // inverse of above

        cnt_sptr->BTP = 0; //0: no bootstrapping, 1: no-parametric, 2: parametric (recommended)
        cnt_sptr->BTPRT = 1.f; // ratio of bootstrapped/original events in the target sinogram (1.0 default)

        cnt_sptr->ETHRLD = 0.05f; // intensity percentage threshold of voxels to be considered in the image
    }
    else
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::getcnst() "
                                 "not implemented for scanner type: " + scanner.get_name());
    return cnt_sptr;
}

static inline unsigned to_1d_idx(const unsigned nrow, const unsigned ncol, const unsigned row, const unsigned col)
{
    return col + ncol*row;
}

template<class dataType>
dataType* create_heap_array(const unsigned numel)
{
    return new dataType[numel];
}

template<class dataType>
dataType* create_heap_array(const unsigned numel, const dataType val)
{
    dataType *array = create_heap_array<dataType>(numel);
    memset(array, val, numel * sizeof(dataType));
    return array;
}

/// Converted from mmraux.py axial_lut
static void get_axLUT_sptr(shared_ptr<axialLUT> &axlut_sptr, float *&li2rng, short *&li2sn_s, char *&li2nos_c, const Cnst &cnt)
{ 
    const int NRNG = cnt.NRNG;
    int NRNG_c, NSN1_c;

    if (cnt.SPN == 1) {
        // number of rings calculated for the given ring range (optionally we can use only part of the axial FOV)
        NRNG_c = cnt.RNG_END - cnt.RNG_STRT;
        // number of sinos in span-1
        NSN1_c = NRNG_c*NRNG_c;
        // correct for the max. ring difference in the full axial extent (don't use ring range (1,63) as for this case no correction)
        if (NRNG_c==64)
            NSN1_c -= 12;
    }
    else {
        NRNG_c = NRNG;
        NSN1_c = cnt.NSN1;
        if (cnt.RNG_END!=NRNG || cnt.RNG_STRT!=0)
            throw std::runtime_error("ProjectorByBinNiftyPETHelper::get_axLUT: the reduced axial FOV only works in span-1.");
    }

    // ring dimensions
    float *rng = create_heap_array<float>(NRNG*2);
    float z = -.5f*float(NRNG)*cnt.AXR;
    for (unsigned i=0; i<unsigned(NRNG); ++i) {
        rng[to_1d_idx(NRNG,2,i,0)] = z;
        z += cnt.AXR;
        rng[to_1d_idx(NRNG,2,i,1)] = z;
    }

    // --create mapping from ring difference to segment number
    // ring difference range
    std::vector<int> rd(2*cnt.MRD+1);
    for (unsigned i=0; i<rd.size(); ++i)
        rd[i] = i - cnt.MRD;
    // ring difference to segment
    int *rd2sg = create_heap_array<int>(rd.size()*2, -1);
    // minimum and maximum ring difference for each segment
    std::vector<int> minrd = {-5,-16, 6,-27,17,-38,28,-49,39,-60,50};
    std::vector<int> maxrd = { 5, -6,16,-17,27,-28,38,-39,49,-50,60};
    for (unsigned i=0; i<rd.size(); ++i) {
        for (unsigned iseg=0; iseg<minrd.size(); ++iseg) {
            if (rd[i]>=minrd[iseg] && rd[i]<=maxrd[iseg]) {
                rd2sg[to_1d_idx(rd.size(),2,i,0)] = rd[i];
                rd2sg[to_1d_idx(rd.size(),2,i,1)] = iseg;
            }
        }
    }

    // create two Michelograms for segments (Mseg)
    // and absolute axial position for individual sinos (Mssrb) which is single slice rebinning
    int *Mssrb = create_heap_array<int>(NRNG*NRNG, -1);
    int *Mseg  = create_heap_array<int>(NRNG*NRNG, -1);
    for (int r1=cnt.RNG_STRT; r1<cnt.RNG_END; ++r1) {
        for (int r0=cnt.RNG_STRT; r0<cnt.RNG_END; ++r0) {
            if (abs(r0-r1)>cnt.MRD)
                continue;
            int ssp = r0+r1; // segment sino position (axially: 0-126)
            int rdd = r1-r0;
            int jseg = -1;
            for (unsigned i=0; i<rd.size(); ++i)
                if (rd2sg[to_1d_idx(rd.size(),2,i,0)] == rdd)
                    jseg = rd2sg[to_1d_idx(rd.size(),2,i,1)];
            Mssrb[to_1d_idx(NRNG,NRNG,r1,r0)] = ssp;
            Mseg[to_1d_idx(NRNG,NRNG,r1,r0)] = jseg; // negative segments are on top diagonals
        }
    }

    // create a Michelogram map from rings to sino number in span-11 (1..837)
    int *Msn  = create_heap_array<int>(NRNG*NRNG, -1);
    // number of span-1 sinos per sino in span-11
    int *Mnos = create_heap_array<int>(NRNG*NRNG, -1);
    std::vector<int> seg = {127,115,115,93,93,71,71,49,49,27,27};
    int *msk = create_heap_array<int>(NRNG*NRNG, 0);
    int *Mtmp = create_heap_array<int>(NRNG*NRNG);
    int i=0;
    for (unsigned iseg=0; iseg<seg.size(); ++iseg) {
        // msk = (Mseg==iseg)
        for (unsigned a=0; a<unsigned(NRNG*NRNG); ++a)
            msk[a] = Mseg[a]==int(iseg)? 1 : 0;
        // Mtmp = np.copy(Mssrb)
        // Mtmp[~msk] = -1
        for (unsigned a=0; a<unsigned(NRNG*NRNG); ++a)
            Mtmp[a] = msk[a] ? Mssrb[a] : -1;

        // uq = np.unique(Mtmp[msk])
        std::vector<int> uq;
        for (unsigned a=0; a<unsigned(NRNG*NRNG); ++a)
            if (msk[a] && std::find(uq.begin(), uq.end(),Mtmp[a]) == uq.end())
                uq.push_back(Mtmp[a]);
        // for u in range(0,len(uq)):
        for (unsigned u=0; u<uq.size(); ++u) {
            // Msn [ Mtmp==uq[u] ] = i
            for (unsigned a=0; a<unsigned(NRNG*NRNG); ++a)
                if (Mtmp[a]==uq[u])
                    Msn[a] = i;
            // Mnos[ Mtmp==uq[u] ] = np.sum(Mtmp==uq[u])
            int sum = 0;
            for (unsigned a=0; a<unsigned(NRNG*NRNG); ++a)
                if (Mtmp[a]==uq[u])
                    ++sum;
            for (unsigned a=0; a<unsigned(NRNG*NRNG); ++a)
                if (Mtmp[a]==uq[u])
                    Mnos[a] = sum;
            ++i;
        }
    }

    //====full LUT
    short *sn1_rno     = create_heap_array<short>(NSN1_c*2, 0);
    short *sn1_ssrb    = create_heap_array<short>(NSN1_c, 0);
    short *sn1_sn11    = create_heap_array<short>(NSN1_c, 0);
    char *sn1_sn11no   = create_heap_array<char>(NSN1_c, 0);
    int sni = 0; // full linear index, up to 4084
    // michelogram of sino numbers for spn-1
    short *Msn1 = create_heap_array<short>(NRNG*NRNG, -1);
    for (unsigned ro=0; ro<unsigned(NRNG); ++ro) {
        unsigned oblique = ro==0? 1 : 2;
        // for m in range(oblique):
        for (unsigned m=0; m<oblique; ++m) {
            // strt = NRNG*(ro+Cnt['RNG_STRT']) + Cnt['RNG_STRT']
            int strt = NRNG*(ro+cnt.RNG_STRT) + cnt.RNG_STRT;
            int stop = (cnt.RNG_STRT+NRNG_c)*NRNG;
            int step = NRNG+1;

            // goes along a diagonal started in the first row at r1
            // for li in range(strt, stop, step):
            for (int li=strt; li<stop; li+=step) {
                int r1, r0;
                // linear indecies of michelogram --> subscript indecies for positive and negative RDs
                if (m==0) {
                    r1 = floor(float(li)/float(NRNG));
                    r0 = li - r1*NRNG;
                }
                // for positive now (? or vice versa)
                else {
                    r0 = floor(float(li)/float(NRNG));
                    r1 = li - r0*NRNG;
                }
                // avoid case when RD>MRD
                if (Msn[to_1d_idx(NRNG,NRNG,r1,r0)]<0)
                    continue;

                sn1_rno[to_1d_idx(NSN1_c,2, sni,0)] = r0;
                sn1_rno[to_1d_idx(NSN1_c,2, sni,1)] = r1;

                sn1_ssrb[sni] = Mssrb[to_1d_idx(NRNG,NRNG,r1,r0)];
                sn1_sn11[sni] = Msn[to_1d_idx(NRNG,NRNG,r0,r1)];

                sn1_sn11no[sni] = Mnos[to_1d_idx(NRNG,NRNG,r0,r1)];

                Msn1[to_1d_idx(NRNG,NRNG,r0,r1)] = sni;
                //--
                sni += 1;
            }
        }
    }

    // span-11 sino to SSRB
    // sn11_ssrb = np.zeros(Cnt['NSN11'], dtype=np.int32);
    int *sn11_ssrb = create_heap_array<int>(cnt.NSN11, -1);
    // sn1_ssrno = np.zeros(Cnt['NSEG0'], dtype=np.int8)
    char *sn1_ssrno = create_heap_array<char>(cnt.NSEG0, 0);
    // for i in range(NSN1_c):
    for (unsigned i=0; i<unsigned(NSN1_c); ++i) {
        sn11_ssrb[sn1_sn11[i]] = sn1_ssrb[i];
        sn1_ssrno[sn1_ssrb[i]] += 1;
    }

    // sn11_ssrno = np.zeros(Cnt['NSEG0'], dtype=np.int8)
    char *sn11_ssrno = create_heap_array<char>(cnt.NSEG0, 0);
    // for i in range(Cnt['NSN11']):
    for (unsigned i=0; i<unsigned(cnt.NSN11); ++i)
        // if sn11_ssrb[i]>0: sn11_ssrno[sn11_ssrb[i]] += 1
        if (sn11_ssrb[i]>0)
            sn11_ssrno[sn11_ssrb[i]] += 1;

    // sn11_ssrb = sn11_ssrb[sn11_ssrb>=0]
    for (unsigned i=0; i<unsigned(cnt.NSN11); ++i)
        if (sn11_ssrb[i]<0)
            sn11_ssrb[i] = 0;

    // ---------------------------------------------------------------------
    // linear index (along diagonals of Michelogram) to rings
    // the number of Michelogram elements considered in projection calculations
    int NLI2R_c = int(float(NRNG_c*NRNG_c)/2.f + float(NRNG_c)/2.f);

    // if the whole scanner is used then account for the MRD and subtract 6 ring permutations
    if (NRNG_c==NRNG)
        NLI2R_c -= 6;

    int    *li2r   = create_heap_array<int>(NLI2R_c*2);
    // the same as above but to sinos in span-11
    int    *li2sn  = create_heap_array<int>(NLI2R_c*2);
    short  *li2sn1 = create_heap_array<short>(NLI2R_c*2);
    li2rng = create_heap_array<float>(NLI2R_c*2);
    // ...to number of sinos (nos)
    int *li2nos = create_heap_array<int>(NLI2R_c);

    int dli = 0;
    for (unsigned ro=0; ro<unsigned(NRNG_c); ++ro) {
        // selects the sub-Michelogram of the whole Michelogram
        unsigned strt = NRNG*(ro+cnt.RNG_STRT) + cnt.RNG_STRT;
        unsigned stop = (cnt.RNG_STRT+NRNG_c)*NRNG;
        unsigned step = NRNG+1;

        // goes along a diagonal started in the first row at r2o
        for (unsigned li=strt; li<stop; li+=step) {
            // from the linear indexes of Michelogram get the subscript indexes
            unsigned r1 = floor(float(li)/float(NRNG));
            unsigned r0 = li - r1*NRNG;
            if (Msn[to_1d_idx(NRNG,NRNG, r1,r0)]<0)
                continue;

            li2r[to_1d_idx(NLI2R_c,2, dli,0)] = r0;
            li2r[to_1d_idx(NLI2R_c,2, dli,1)] = r1;
            //--//rng[to_1d_idx(NRNG,2,i,1)] = z;
            li2rng[to_1d_idx(NLI2R_c,2, dli,0)] = rng[to_1d_idx(NRNG,2,r0,0)];
            li2rng[to_1d_idx(NLI2R_c,2, dli,1)] = rng[to_1d_idx(NRNG,2,r1,0)];
            //--
            li2sn[to_1d_idx(NLI2R_c,2, dli,0)] = Msn[to_1d_idx(NRNG,NRNG,r0,r1)];
            li2sn[to_1d_idx(NLI2R_c,2, dli,1)] = Msn[to_1d_idx(NRNG,NRNG,r1,r0)];

            li2sn1[to_1d_idx(NLI2R_c,2, dli,0)] = Msn1[to_1d_idx(NRNG,NRNG,r0,r1)];
            li2sn1[to_1d_idx(NLI2R_c,2, dli,1)] = Msn1[to_1d_idx(NRNG,NRNG,r1,r0)];

            li2nos[dli] = Mnos[to_1d_idx(NRNG,NRNG,r1,r0)];

            ++dli;
        }
    }

    // Need some results in a different data type
    li2sn_s = create_heap_array<short>(NLI2R_c*2);
    for (unsigned i=0; i<unsigned(NLI2R_c*2); ++i)
        li2sn_s[i] = short(li2sn[i]);
    li2nos_c = create_heap_array<char>(NLI2R_c);
    for (unsigned i=0; i<unsigned(NLI2R_c); ++i)
        li2nos_c[i] = char(li2nos[i]);

    // Delete temporary variables
    delete [] rng;
    delete [] rd2sg;
    delete [] Mssrb;
    delete [] Mseg;
    delete [] Msn;
    delete [] Mnos;
    delete [] msk;
    delete [] Mtmp;
    delete [] Msn1;
    delete [] sn11_ssrb;
    delete [] sn1_ssrno;
    delete [] sn11_ssrno;
    delete [] li2sn1;

    // Fill in struct
    axlut_sptr = shared_ptr<axialLUT>(new axialLUT, delete_axialLUT);
    axlut_sptr->li2rno     = li2r;        // int   linear indx to ring indx
    axlut_sptr->li2sn      = li2sn;       // int   linear michelogram index (along diagonals) to sino index
    axlut_sptr->li2nos     = li2nos;      // int   linear indx to no of sinos in span-11
    axlut_sptr->sn1_rno    = sn1_rno;     // short
    axlut_sptr->sn1_sn11   = sn1_sn11;    // short
    axlut_sptr->sn1_ssrb   = sn1_ssrb;    // short
    axlut_sptr->sn1_sn11no = sn1_sn11no;  // char
    // array sizes
    axlut_sptr->Nli2rno[0] = NLI2R_c;
    axlut_sptr->Nli2rno[1] = 2;
    axlut_sptr->Nli2sn[0]  = NLI2R_c;
    axlut_sptr->Nli2sn[1]  = 2;
    axlut_sptr->Nli2nos    = NLI2R_c;
}

static
void get_txLUT_sptr(shared_ptr<txLUTs> &txlut_sptr, float *&crs, short *&s2c, Cnst &cnt)
{
    txlut_sptr = shared_ptr<txLUTs>(new txLUTs, delete_txLUT);
    *txlut_sptr = get_txlut(cnt);

    s2c = create_heap_array<short>(txlut_sptr->naw*2);
    for (unsigned i=0; i<unsigned(txlut_sptr->naw); ++i) {
        s2c[ 2*i ] = txlut_sptr->s2c[i].c0;
        s2c[2*i+1] = txlut_sptr->s2c[i].c1;
    }
    // from mmraux.py
    const float bw = 3.209f; // block width
    // const float dg = 0.474f; // block gap [cm]
    const int NTBLK = 56;
    const float alpha = 2*M_PI/float(NTBLK); // 2*pi/NTBLK
    crs = create_heap_array<float>(4 * cnt.NCRS);
    float phi = 0.5f*M_PI - alpha/2.f - 0.001f;
    for (int bi=0; bi<NTBLK; ++bi) {
        //-tangent point (ring against detector block)
        // ye = RE*np.sin(phi)
        // xe = RE*np.cos(phi)
        float y = cnt.RE * sin(phi);
        float x = cnt.RE * cos(phi);
        //-vector for the face of crystals
        float pv[2] = {-y, x};
        float pv_ = pow(pv[0]*pv[0] + pv[1]*pv[1], 0.5f);
        pv[0] /= pv_;
        pv[1] /= pv_;
        // update phi for next block
        phi -= alpha;
        //-end block points
        float xcp = x + (bw/2)*pv[0];
        float ycp = y + (bw/2)*pv[1];
        for (unsigned n=1; n<9; ++n) {
            int c = bi*9 + n - 1;
            crs[to_1d_idx(4,cnt.NCRS, 0,c)] = xcp;
            crs[to_1d_idx(4,cnt.NCRS, 1,c)] = ycp;
            float xc = x + (bw/2-float(n)*bw/8)*pv[0];
            float yc = y + (bw/2-float(n)*bw/8)*pv[1];
            crs[to_1d_idx(4,cnt.NCRS, 2,c)] = xc;
            crs[to_1d_idx(4,cnt.NCRS, 3,c)] = yc;
            xcp = xc;
            ycp = yc;
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
set_up()
{

    // Intensities from projections do not have to match between
    // reconstruction packages. To account for that we need to
    // divide by this value after forward projection and multiply
    // after back projection.
    // This value is fixed for the mMR, but may have to change
    // if other scanners are incorporated.
    _niftypet_to_stir_ratio = 1.f;//.25f;

    if (_span < 0)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_up() "
                                "sinogram span not set.");

    if (_att < 0)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_up() "
                                "emission or transmission mode (att) not set.");

    if (_scanner_type == Scanner::Unknown_scanner)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_up() "
                                "scanner type not set.");

    // Get consts
    _cnt_sptr = get_cnst(_scanner_type, _verbose, _devid, _span);

    // Get txLUT
    get_txLUT_sptr(_txlut_sptr, _crs, _s2c, *_cnt_sptr);

    // Get axLUT
    get_axLUT_sptr(_axlut_sptr, _li2rng, _li2sn, _li2nos, *_cnt_sptr);

    switch(_cnt_sptr->SPN){
        case 11:
            _nsinos = _cnt_sptr->NSN11; break;
        case 1:
            _nsinos = _cnt_sptr->NSEG0; break;
        default:
            throw std::runtime_error("Unsupported span");
     }

    // isub
    _isub = std::vector<int>(unsigned(AW));
    for (unsigned i = 0; i<unsigned(AW); i++) _isub[i] = int(i);

    _already_set_up = true;
}

void
ProjectorByBinNiftyPETHelper::
check_set_up() const
{
    if (!_already_set_up)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::check_set_up() "
                                 "Make sure filenames have been set and set_up has been run.");
}

std::vector<float>
ProjectorByBinNiftyPETHelper::
create_niftyPET_image()
{
    return std::vector<float>(SZ_IMZ*SZ_IMX*SZ_IMY,0);
}

std::vector<float>
ProjectorByBinNiftyPETHelper::
create_niftyPET_sinogram_no_gaps() const
{
    check_set_up();
    return std::vector<float>(_isub.size() * static_cast<unsigned long>(_nsinos), 0);
}

std::vector<float>
ProjectorByBinNiftyPETHelper::
create_niftyPET_sinogram_with_gaps() const
{
    return std::vector<float>(NSBINS*NSANGLES*unsigned(_nsinos), 0);
}

void get_stir_indices_and_dims(int stir_dim[3], Coordinate3D<int> &min_indices, Coordinate3D<int> &max_indices, const DiscretisedDensity<3,float >&stir)
{
    if (!stir.get_regular_range(min_indices, max_indices))
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_input - "
                                 "expected image to have regular range.");
    for (int i=0; i<3; ++i)
        stir_dim[i] = max_indices[i + 1] - min_indices[i + 1] + 1;
}

unsigned convert_niftypet_im_3d_to_1d_idx(const unsigned x, const unsigned y, const unsigned z)
{
    return z*SZ_IMX*SZ_IMY + y*SZ_IMX + x;
}

unsigned
ProjectorByBinNiftyPETHelper::
convert_niftypet_proj_3d_to_1d_idx(const unsigned ang, const unsigned bins, const unsigned sino) const
{
    return sino*NSANGLES*NSBINS + ang*NSBINS + bins;
}

void
ProjectorByBinNiftyPETHelper::
permute(std::vector<float> &output_array, const std::vector<float> &orig_array, const unsigned output_dims[3], const unsigned permute_order[3]) const
{
#ifndef NDEBUG
    // Check that in the permute order, each number is between 0 and 2 (can't be <0 because it's unsigned)
    for (unsigned i=0; i<3; ++i)
        if (permute_order[i]>2)
            throw std::runtime_error("Permute order values should be between 0 and 2.");
    // Check that each number is unique
    for (unsigned i=0; i<3; ++i)
        for (unsigned j=i+1; j<3; ++j)
            if (permute_order[i] == permute_order[j])
                throw std::runtime_error("Permute order values should be unique.");
    // Check that size of output_dims==arr.size()
    assert(orig_array.size() == output_dims[0]*output_dims[1]*output_dims[2]);
    // Check that output array is same size as input array
    assert(orig_array.size() == output_array.size());
#endif

    // Calculate old dimensions
    unsigned old_dims[3];
    for (unsigned i=0; i<3; ++i)
        old_dims[permute_order[i]] = output_dims[i];

    // Loop over all elements
    unsigned old_3d_idx[3], new_3d_idx[3], new_1d_idx;
    for (unsigned old_1d_idx=0; old_1d_idx<orig_array.size(); ++old_1d_idx) {

        // From the 1d index, generate the old 3d index
        old_3d_idx[2] =  old_1d_idx %  old_dims[2];
        old_3d_idx[1] = (old_1d_idx /  old_dims[2]) % old_dims[1];
        old_3d_idx[0] =  old_1d_idx / (old_dims[2]  * old_dims[1]);

        // Get the corresponding new 3d index
        for (unsigned i=0; i<3; ++i)
            new_3d_idx[i] = old_3d_idx[permute_order[i]];

        // Get the new 1d index from the new 3d index
        new_1d_idx = new_3d_idx[0]*output_dims[2]*output_dims[1] + new_3d_idx[1]*output_dims[2] + new_3d_idx[2];

        // Fill the data
        output_array[new_1d_idx] = orig_array[old_1d_idx];
    }
}

void
ProjectorByBinNiftyPETHelper::
remove_gaps(std::vector<float> &sino_no_gaps, const std::vector<float> &sino_w_gaps) const
{
    check_set_up();
    assert(!sino_no_gaps.empty());

    if (_verbose)
        getMemUse();

    ::remove_gaps(sino_no_gaps.data(),
                  const_cast<std::vector<float>&>(sino_w_gaps).data(),
                  _nsinos,
                  _txlut_sptr->aw2ali,
                  *_cnt_sptr);
}

void
ProjectorByBinNiftyPETHelper::
put_gaps(std::vector<float> &sino_w_gaps, const std::vector<float> &sino_no_gaps) const
{
    check_set_up();
    assert(!sino_w_gaps.empty());

    std::vector<float> unpermuted_sino_w_gaps = this->create_niftyPET_sinogram_with_gaps();

    if (_verbose)
        getMemUse();

    ::put_gaps(unpermuted_sino_w_gaps.data(),
               const_cast<std::vector<float>&>(sino_no_gaps).data(),
               _txlut_sptr->aw2ali,
               *_cnt_sptr);

    // Permute the data (as this is done on the NiftyPET python side after put gaps
    unsigned output_dims[3] = {837, 252, 344};
    unsigned permute_order[3] = {2,0,1};
    this->permute(sino_w_gaps,unpermuted_sino_w_gaps,output_dims,permute_order);
}

void
ProjectorByBinNiftyPETHelper::
back_project(std::vector<float> &image, const std::vector<float> &sino_no_gaps) const
{
    check_set_up();
    assert(!image.empty());

    std::vector<float> unpermuted_image = this->create_niftyPET_image();

    if (_verbose)
        getMemUse();

    gpu_bprj(unpermuted_image.data(),
             const_cast<std::vector<float>&>(sino_no_gaps).data(),
             _li2rng,
             _li2sn,
             _li2nos,
             _s2c,
             _txlut_sptr->aw2ali,
             _crs,
             const_cast<std::vector<int>&>(_isub).data(),
             int(_isub.size()),
             AW,
             4, // n0crs
             nCRS,
             *_cnt_sptr);

    // Permute the data (as this is done on the NiftyPET python side after back projection
    unsigned output_dims[3] = {127,320,320};
    unsigned permute_order[3] = {2,0,1};
    this->permute(image,unpermuted_image,output_dims,permute_order);

    // Scale to account for niftypet-to-stir ratio
    for (unsigned i=0; i<image.size(); ++i)
        image[i] *= _niftypet_to_stir_ratio;
}

void
ProjectorByBinNiftyPETHelper::
forward_project(std::vector<float> &sino_no_gaps, const std::vector<float> &image) const
{
    check_set_up();
    assert(!sino_no_gaps.empty());

    // Permute the data (as this is done on the NiftyPET python side before forward projection
    unsigned output_dims[3] = {320,320,127};
    unsigned permute_order[3] = {1,2,0};
    std::vector<float> permuted_image = this->create_niftyPET_image();
    this->permute(permuted_image,image,output_dims,permute_order);

    if (_verbose)
        getMemUse();

    gpu_fprj(sino_no_gaps.data(),
             permuted_image.data(),
             _li2rng,
             _li2sn,
             _li2nos,
             _s2c,
             _txlut_sptr->aw2ali,
             _crs,
             const_cast<std::vector<int>&>(_isub).data(),
             int(_isub.size()),
             AW,
             4, // n0crs
             nCRS,
             *_cnt_sptr,
             _att);

    // Scale to account for niftypet-to-stir ratio
    for (unsigned i=0; i<sino_no_gaps.size(); ++i)
        sino_no_gaps[i] /= _niftypet_to_stir_ratio;
}

shared_ptr<ProjData>
ProjectorByBinNiftyPETHelper::create_stir_sino()
{
    const int span=11;
    const int max_ring_diff=60;
    const int view_mash_factor=1;
    shared_ptr<ExamInfo> ei_sptr = MAKE_SHARED<ExamInfo>();
    ei_sptr->imaging_modality = ImagingModality::PT;
    shared_ptr<Scanner> scanner_sptr(Scanner::get_scanner_from_name("mMR"));
    int num_views = scanner_sptr->get_num_detectors_per_ring() / 2 / view_mash_factor;
    int num_tang_pos = scanner_sptr->get_max_num_non_arccorrected_bins();
    shared_ptr<ProjDataInfo> pdi_sptr = ProjDataInfo::construct_proj_data_info
            (scanner_sptr, span, max_ring_diff, num_views, num_tang_pos, false);
    shared_ptr<ProjDataInMemory> pd_sptr = MAKE_SHARED<ProjDataInMemory>(ei_sptr, pdi_sptr);
    return pd_sptr;
}

void
ProjectorByBinNiftyPETHelper::
lm_to_proj_data(shared_ptr<ProjData> &prompts_sptr, shared_ptr<ProjData> &delayeds_sptr, shared_ptr<ProjData> &randoms_sptr,
                const std::string &lm_binary_file, const int tstart, const int tstop) const
{
    check_set_up();
    
    // Get listmode info
    char *flm = create_heap_array<char>(lm_binary_file.length() + 1);
    strcpy(flm, lm_binary_file.c_str());
    getLMinfo(flm, *_cnt_sptr);
    free(lmprop.atag);
    free(lmprop.btag);
    free(lmprop.ele4chnk);
    free(lmprop.ele4thrd);
    free(lmprop.t2dfrm);

    // preallocate all the output arrays - in def.h VTIME=2 (), MXNITAG=5400 (max time 1h30)
    const int nitag = lmprop.nitag;
    const int pow_2_MXNITAG = pow(2,VTIME);
    int tn;
    if (nitag>MXNITAG)
        tn = MXNITAG/pow_2_MXNITAG;
    else
        tn = (nitag+pow_2_MXNITAG-1)/pow_2_MXNITAG;

    unsigned short frames(0);
    int nfrm(1);

    // structure of output data
    // var   | type               | python var | description                      | shape
    // ------+--------------------|------------+----------------------------------+-----------------------------------------------------------------
    // nitag | int                |            | gets set inside lmproc           | 
    // sne   | int                |            | gets set inside lmproc           |
    // snv   | unsigned int *     | pvs        | sino views                       | [ tn,           Cnt['NSEG0'],    Cnt['NSBINS']                  ]
    // hcp   | unsigned int *     | phc        | head curve prompts               | [ nitag                                                         ]
    // hcd   | unsigned int *     | dhc        | head curve delayeds              | [ nitag                                                         ]
    // fan   | unsigned int *     | fan        | fansums                          | [ nfrm,         Cnt['NRNG'],     Cnt['NCRS']                    ]
    // bck   | unsigned int *     | bck        | buckets (singles)                | [ 2,            nitag,           Cnt['NBCKT']                   ]
    // mss   | float *            | mss        | centre of mass (axially)         | [ nitag                                                         ]
    // ssr   | unsigned int *     | ssr        |                                  | [ Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']                  ]
    // psn   | void *             | psino      | if nfrm==1, unsigned int*        | [ nfrm,          nsinos,         Cnt['NSANGLES'], Cnt['NSBINS'] ]
    // dsn   | void *             | dsino      | if nfrm==1, unsigned int*        | [ nfrm,          nsinos,         Cnt['NSANGLES'], Cnt['NSBINS'] ]
    // psm   | unsigned long long |            | gets set inside lmproc           |
    // dsm   | unsigned long long |            | gets set inside lmproc           |
    // tot   | unsigned int       |            | gets set inside lmproc           |
    const unsigned int num_sino_elements = _nsinos * _cnt_sptr->A * _cnt_sptr->W;
    hstout dicout; 
    dicout.snv = create_heap_array<unsigned int>(tn * _cnt_sptr->NSEG0 * _cnt_sptr->W);
    dicout.hcp = create_heap_array<unsigned int>(nitag);
    dicout.hcd = create_heap_array<unsigned int>(nitag);
    dicout.fan = create_heap_array<unsigned int>(nfrm * _cnt_sptr->NRNG * _cnt_sptr->NCRS);
    dicout.bck = create_heap_array<unsigned int>(2 * nitag * _cnt_sptr->B);
    dicout.mss = create_heap_array<float>       (nitag);
    dicout.ssr = create_heap_array<unsigned int>(_cnt_sptr->NSEG0 * _cnt_sptr->A * _cnt_sptr->W);
    if (nfrm == 1)  {
        dicout.psn =  create_heap_array<unsigned short>(nfrm * num_sino_elements);
        dicout.dsn =  create_heap_array<unsigned short>(nfrm * num_sino_elements);
    }
    else
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::lm_to_proj_data: If nfrm>1, "
                                  "dicout.psn and dicout.dsn should be unsigned char*. Not "
                                  "tested, but should be pretty easy.");

    lmproc(dicout, // hstout (struct): output
           flm, // char *: binary filename (.s, .bf)
           &frames, // unsigned short *: think for one frame, frames = 0
           nfrm, // int: num frames
           tstart, // int
           tstop, // int
           _txlut_sptr->s2c, // *LORcc (struct)
           *_axlut_sptr, // axialLUT (struct)
           *_cnt_sptr); // Cnst (struct)

    // Convert prompts and delayeds to STIR sinogram
    const unsigned short *psn_int = (const unsigned short*)dicout.psn;
    const unsigned short *dsn_int = (const unsigned short*)dicout.dsn;
    std::vector<float> np_prompts = create_niftyPET_sinogram_with_gaps();
    std::vector<float> np_delayeds = create_niftyPET_sinogram_with_gaps();
    for (unsigned i=0; i<num_sino_elements; ++i) {
        np_prompts[i] = float(psn_int[i]);
        np_delayeds[i] = float(dsn_int[i]);
    }
    prompts_sptr = create_stir_sino();
    delayeds_sptr = create_stir_sino();
    convert_proj_data_niftyPET_to_stir(*prompts_sptr, np_prompts);
    convert_proj_data_niftyPET_to_stir(*delayeds_sptr, np_delayeds);

    // estimated crystal map of singles
    // cmap = np.zeros((Cnt['NCRS'], Cnt['NRNG']), dtype=np.float32)
    std::vector<float> cmap(_cnt_sptr->NCRS*_cnt_sptr->NRNG, 0);

    // Estiamte randoms from delayeds
    std::vector<float> np_randoms = create_niftyPET_sinogram_with_gaps();
    gpu_randoms(np_randoms.data(),     // float *rsn
                cmap.data(),           // float *cmap,
                dicout.fan,            // unsigned int * fansums,
                *_txlut_sptr,          // txLUTs txlut,
                _axlut_sptr->sn1_rno,  // short *sn1_rno,
                _axlut_sptr->sn1_sn11, // short *sn1_sn11,
                *_cnt_sptr             // const Cnst Cnt)
                );

    randoms_sptr = create_stir_sino();
    convert_proj_data_niftyPET_to_stir(*randoms_sptr, np_randoms);

    // Clear up
    delete [] flm;
    delete [] dicout.snv;
    delete [] dicout.hcp;
    delete [] dicout.hcd;
    delete [] dicout.fan;
    delete [] dicout.bck;
    delete [] dicout.mss;
    delete [] dicout.ssr;
    if (nfrm == 1)  {
        delete [] (unsigned int*)dicout.psn;
        delete [] (unsigned int*)dicout.dsn;
    }
    else
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::lm_to_proj_data: If nfrm>1, "
                                  "need to cast before deleting as is stored as void*.");
}

void check_im_sizes(const int stir_dim[3], const int np_dim[3])
{
    for (int i=0; i<3; ++i)
        if (stir_dim[i] != np_dim[i])
            throw std::runtime_error((boost::format(
                                      "ProjectorByBinNiftyPETHelper::check_im_sizes() - "
                                      "STIR image (%1%, %2%, %3%) should be == (%4%,%5%,%6%).")
                                      % stir_dim[0] % stir_dim[1] % stir_dim[2]
                                      % np_dim[0]   % np_dim[1]   % np_dim[2]).str());
}

void check_voxel_spacing(const DiscretisedDensity<3, float> &stir)
{
    // Requires image to be a VoxelsOnCartesianGrid
    const VoxelsOnCartesianGrid<float> &stir_vocg =
            dynamic_cast<const VoxelsOnCartesianGrid<float>&>(stir);
    const BasicCoordinate<3,float> stir_spacing = stir_vocg.get_grid_spacing();

    // Get NiftyPET image spacing (need to *10 for mm)
    float np_spacing[3] = { 10.f*SZ_VOXZ, 10.f*SZ_VOXY, 10.f*SZ_VOXY };

    for (unsigned i=0; i<3; ++i)
        if (std::abs(stir_spacing[int(i)+1] - np_spacing[i]) > 1e-4f)
            throw std::runtime_error((boost::format(
                                      "ProjectorByBinNiftyPETHelper::check_voxel_spacing() - "
                                      "STIR image (%1%, %2%, %3%) should be == (%4%,%5%,%6%).")
                                      % stir_spacing[1] % stir_spacing[2] % stir_spacing[3]
                                      % np_spacing[0]   % np_spacing[1]   % np_spacing[2]).str());
}

void
ProjectorByBinNiftyPETHelper::
convert_image_stir_to_niftyPET(std::vector<float> &np_vec, const DiscretisedDensity<3, float> &stir)
{
    // Get the dimensions of the input image
    Coordinate3D<int> min_indices;
    Coordinate3D<int> max_indices;
    int stir_dim[3];
    get_stir_indices_and_dims(stir_dim,min_indices,max_indices,stir);

    // NiftyPET requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
    // which at the time of writing was (127,320,320).
    const int np_dim[3] = {SZ_IMZ,SZ_IMX,SZ_IMY};
    check_im_sizes(stir_dim,np_dim);
    check_voxel_spacing(stir);

    // Copy data from STIR to NiftyPET image
    unsigned np_z, np_y, np_x, np_1d;
    for (int z = min_indices[1]; z <= max_indices[1]; z++) {
        for (int y = min_indices[2]; y <= max_indices[2]; y++) {
            for (int x = min_indices[3]; x <= max_indices[3]; x++) {
                // Convert the stir 3d index to a NiftyPET 1d index
                np_z = unsigned(z - min_indices[1]);
                np_y = unsigned(y - min_indices[2]);
                np_x = unsigned(x - min_indices[3]);
                np_1d = convert_niftypet_im_3d_to_1d_idx(np_x,np_y,np_z);
                np_vec[np_1d] = stir[z][y][x];
            }
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
convert_image_niftyPET_to_stir(DiscretisedDensity<3,float> &stir, const std::vector<float> &np_vec)
{
    // Get the dimensions of the input image
    Coordinate3D<int> min_indices;
    Coordinate3D<int> max_indices;
    int stir_dim[3];
    get_stir_indices_and_dims(stir_dim,min_indices,max_indices,stir);

    // NiftyPET requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
    // which at the time of writing was (127,320,320).
    const int np_dim[3] = {SZ_IMZ,SZ_IMX,SZ_IMY};
    check_im_sizes(stir_dim,np_dim);
    check_voxel_spacing(stir);

    // Copy data from NiftyPET to STIR image
    unsigned np_z, np_y, np_x, np_1d;
    for (int z = min_indices[1]; z <= max_indices[1]; z++) {
        for (int y = min_indices[2]; y <= max_indices[2]; y++) {
            for (int x = min_indices[3]; x <= max_indices[3]; x++) {
                // Convert the stir 3d index to a NiftyPET 1d index
                np_z = unsigned(z - min_indices[1]);
                np_y = unsigned(y - min_indices[2]);
                np_x = unsigned(x - min_indices[3]);
                np_1d = convert_niftypet_im_3d_to_1d_idx(np_x,np_y,np_z);
                stir[z][y][x] = np_vec[np_1d];
            }
        }
    }

    // After the back projection, we enforce a truncation outside of the FOV.
    // This is because the NiftyPET FOV is smaller than the STIR FOV and this
    // could cause some voxel values to spiral out of control.
    // truncate_rim(stir,17);
}

void
get_vals_for_proj_data_conversion(std::vector<int> &sizes, std::vector<int> &segment_sequence,
                                  int &num_sinograms, int &min_view, int &max_view,
                                  int &min_tang_pos, int &max_tang_pos,
                                  const ProjDataInfo& proj_data_info, const std::vector<float> &np_vec)
{
    const ProjDataInfoCylindricalNoArcCorr * info_sptr =
            dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *>(&proj_data_info);
    if (is_null_ptr(info_sptr))
        error("ProjectorByBinNiftyPETHelper: only works with cylindrical projection data without arc-correction");

    const int max_ring_diff   = info_sptr->get_max_ring_difference(info_sptr->get_max_segment_num());
    const int max_segment_num = info_sptr->get_max_segment_num();

    segment_sequence.resize(unsigned(2*max_ring_diff+1));
    sizes.resize(unsigned(2*max_ring_diff+1));
    segment_sequence[unsigned(0)]=0;
    sizes[0]=info_sptr->get_num_axial_poss(0);
    for (int segment_num=1; segment_num<=max_segment_num; ++segment_num) {
       segment_sequence[unsigned(2*segment_num-1)] = -segment_num;
       segment_sequence[unsigned(2*segment_num)] = segment_num;
       sizes [unsigned(2*segment_num-1)] =info_sptr->get_num_axial_poss(-segment_num);
       sizes [unsigned(2*segment_num)] =info_sptr->get_num_axial_poss(segment_num);
    }

    // Get dimensions of STIR sinogram
    min_view      = proj_data_info.get_min_view_num();
    max_view      = proj_data_info.get_max_view_num();
    min_tang_pos  = proj_data_info.get_min_tangential_pos_num();
    max_tang_pos  = proj_data_info.get_max_tangential_pos_num();


    num_sinograms = proj_data_info.get_num_axial_poss(0);
    for (int s=1; s<= proj_data_info.get_max_segment_num(); ++s)
        num_sinograms += 2* proj_data_info.get_num_axial_poss(s);

    int num_proj_data_elems = num_sinograms * (1+max_view-min_view) * (1+max_tang_pos-min_tang_pos);

    // Make sure they're the same size
    if (np_vec.size() != unsigned(num_proj_data_elems))
        error(boost::format(
                  "ProjectorByBinNiftyPETHelper::get_vals_for_proj_data_conversion "
                  "NiftyPET and STIR sinograms are different sizes (%1% for STIR versus %2% for NP")
              % num_proj_data_elems % np_vec.size());
}

void get_stir_segment_and_axial_pos_from_niftypet_sino(int &segment, int &axial_pos, const unsigned np_sino, const std::vector<int> &sizes, const std::vector<int> &segment_sequence)
{
    int z = int(np_sino);
    for (unsigned i=0; i<segment_sequence.size(); ++i) {
        if (z < sizes[i]) {
            axial_pos = z;
            segment = segment_sequence[i];
            return;
          }
        else {
            z -= sizes[i];
        }
    }
}

void get_niftypet_sino_from_stir_segment_and_axial_pos(unsigned &np_sino, const int segment, const int axial_pos, const std::vector<int> &sizes, const std::vector<int> &segment_sequence)
{
    np_sino = 0U;
    for (unsigned i=0; i<segment_sequence.size(); ++i) {
        if (segment == segment_sequence[i]) {
            np_sino += axial_pos;
            return;
          }
        else {
            np_sino += sizes[i];
        }
    }
    throw std::runtime_error("ProjectorByBinNiftyPETHelper::get_niftypet_sino_from_stir_segment_and_axial_pos(): Failed to find NiftyPET sinogram.");
}

void
ProjectorByBinNiftyPETHelper::
convert_viewgram_stir_to_niftyPET(std::vector<float> &np_vec, const Viewgram<float>& viewgram) const
{
    // Get the values (and LUT) to be able to switch between STIR and NiftyPET projDatas
    std::vector<int> sizes, segment_sequence;
    int num_sinograms, min_view, max_view, min_tang_pos, max_tang_pos;
    get_vals_for_proj_data_conversion(sizes, segment_sequence, num_sinograms, min_view, max_view,
                                      min_tang_pos, max_tang_pos, *viewgram.get_proj_data_info_sptr(), np_vec);

    const int segment = viewgram.get_segment_num();
    const int view = viewgram.get_view_num();

    // Loop over the STIR view and tangential position
    for (int ax_pos=viewgram.get_min_axial_pos_num(); ax_pos<=viewgram.get_max_axial_pos_num(); ++ax_pos) {

        unsigned np_sino;

        // Convert the NiftyPET sinogram to STIR's segment and axial position
        get_niftypet_sino_from_stir_segment_and_axial_pos(np_sino, segment, ax_pos, sizes, segment_sequence);

        for (int tang_pos=min_tang_pos; tang_pos<=max_tang_pos; ++tang_pos) {

            unsigned np_ang  = unsigned(view-min_view);
            unsigned np_bin  = unsigned(tang_pos-min_tang_pos);
            unsigned np_1d = convert_niftypet_proj_3d_to_1d_idx(np_ang,np_bin,np_sino);
            np_vec.at(np_1d) = viewgram.at(ax_pos).at(tang_pos);
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
convert_proj_data_stir_to_niftyPET(std::vector<float> &np_vec, const ProjData& stir) const
{
    const int min_view = stir.get_min_view_num();
    const int max_view = stir.get_max_view_num();
    const int min_segment = stir.get_min_segment_num();
    const int max_segment = stir.get_max_segment_num();

    for (int view=min_view; view<=max_view; ++view) {
        for (int segment=min_segment; segment<=max_segment; ++segment) {
            convert_viewgram_stir_to_niftyPET(np_vec, stir.get_viewgram(view,segment));
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
convert_proj_data_niftyPET_to_stir(ProjData &stir, const std::vector<float> &np_vec) const
{
    // Get the values (and LUT) to be able to switch between STIR and NiftyPET projDatas
    std::vector<int> sizes, segment_sequence;
    int num_sinograms, min_view, max_view, min_tang_pos, max_tang_pos;
    get_vals_for_proj_data_conversion(sizes, segment_sequence, num_sinograms, min_view, max_view,
                                      min_tang_pos, max_tang_pos, *stir.get_proj_data_info_sptr(), np_vec);

    int segment, axial_pos;
    // Loop over all NiftyPET sinograms
    for (unsigned np_sino = 0; np_sino < unsigned(num_sinograms); ++np_sino) {

        // Convert the NiftyPET sinogram to STIR's segment and axial position
        get_stir_segment_and_axial_pos_from_niftypet_sino(segment, axial_pos, np_sino, sizes, segment_sequence);

        // Get the corresponding STIR sinogram
        Sinogram<float> sino = stir.get_empty_sinogram(axial_pos,segment);

        // Loop over the STIR view and tangential position
        for (int view=min_view; view<=max_view; ++view) {
            for (int tang_pos=min_tang_pos; tang_pos<=max_tang_pos; ++tang_pos) {

                unsigned np_ang  = unsigned(view-min_view);
                unsigned np_bin  = unsigned(tang_pos-min_tang_pos);
                unsigned np_1d = convert_niftypet_proj_3d_to_1d_idx(np_ang,np_bin,np_sino);
                sino.at(view).at(tang_pos) = np_vec.at(np_1d);
            }
        }
        stir.set_sinogram(sino);
    }
}

END_NAMESPACE_STIR
