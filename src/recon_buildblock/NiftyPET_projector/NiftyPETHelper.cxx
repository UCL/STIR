//
//
/*!

  \file
  \ingroup projection
  \ingroup NiftyPET

  \brief non-inline implementations for stir::NiftyPETHelper

  \author Richard Brown


*/
/*
    Copyright (C) 2019-2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/NiftyPET_projector/NiftyPETHelper.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/is_null_ptr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInMemory.h"
#include "stir/IndexRange3D.h"
#include "stir/FilePath.h"
#include "stir/IO/stir_ecat_common.h"
#include "stir/error.h"
#include "stir/format.h"
// Non-STIR includes
#include <fstream>
#include <math.h>
#include "driver_types.h"
// NiftyPET includes
#include "def.h"
#include "auxmath.h"
#include "prjb.h"
#include "prjf.h"
#include "recon.h"
#include "lmproc.h"
#include "scanner_0.h"
#include "rnd.h"
#include "norm.h"

START_NAMESPACE_STIR

NiftyPETHelper::~NiftyPETHelper()
{}

static void
delete_axialLUT(axialLUT* axlut_ptr)
{
  if (!axlut_ptr)
    return;
  delete[] axlut_ptr->li2rno;
  delete[] axlut_ptr->li2sn;
  delete[] axlut_ptr->li2nos;
  delete[] axlut_ptr->sn1_rno;
  delete[] axlut_ptr->sn1_sn11;
  delete[] axlut_ptr->sn1_ssrb;
  delete[] axlut_ptr->sn1_sn11no;
}

static void
delete_txLUT(txLUTs* txluts_ptr)
{
  if (!txluts_ptr)
    return;
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

static shared_ptr<Cnst>
get_cnst(const Scanner& scanner, const bool cuda_verbose, const char cuda_device, const char span)
{
  shared_ptr<Cnst> cnt_sptr = MAKE_SHARED<Cnst>();

  cnt_sptr->DEVID = cuda_device; // device (GPU) ID.  allows choosing the device on which to perform calculations
  cnt_sptr->VERBOSE = cuda_verbose;

  if (scanner.get_type() == Scanner::Siemens_mMR)
    {
      if (!(span == 0 || span == 1 || span == 11))
        throw std::runtime_error("NiftyPETHelper::getcnst() "
                                 "only spans 0, 1 and 11 supported for scanner type: "
                                 + scanner.get_name());

      cnt_sptr->A = NSANGLES; // sino angles
      cnt_sptr->W = NSBINS;   // sino bins for any angular index
      cnt_sptr->aw = AW;      // sino bins (active only)

      cnt_sptr->NCRS = nCRS;   // number of crystals
      cnt_sptr->NCRSR = nCRSR; // reduced number of crystals by gaps
      cnt_sptr->NRNG = NRINGS; // number of axial rings
      cnt_sptr->D = -1;        // number of linear indexes along Michelogram diagonals                         /*unknown*/
      cnt_sptr->Bt = -1;       // number of buckets transaxially                                               /*unknown*/

      cnt_sptr->B = NBUCKTS; // number of buckets (total)
      cnt_sptr->Cbt = 32552; // number of crystals in bucket transaxially                                /*unknown*/
      cnt_sptr->Cba = 3;     // number of crystals in bucket axially                                         /*unknown*/

      cnt_sptr->NSN1 = NSINOS;           // number of sinos in span-1
      cnt_sptr->NSN11 = NSINOS11;        // in span-11
      cnt_sptr->NSN64 = NRINGS * NRINGS; // with no MRD limit

      cnt_sptr->SPN = span; // span-1 (s=1) or span-11 (s=11, default) or SSRB (s=0)
      cnt_sptr->NSEG0 = SEG0;

      cnt_sptr->RNG_STRT = 0;
      cnt_sptr->RNG_END = NRINGS;

      cnt_sptr->TGAP = 9;   // get the crystal gaps right in the sinogram, period and offset given      /*unknown*/
      cnt_sptr->OFFGAP = 1; /*unknown*/

      cnt_sptr->NSCRS = 21910; // number of scatter crystals used in scatter estimation              /*unknown*/
      std::vector<short> sct_irng = { 0, 10, 19, 28, 35, 44, 53, 63 }; // scatter ring definition
      cnt_sptr->NSRNG = int(sct_irng.size());
      cnt_sptr->MRD = mxRD; // maximum ring difference

      cnt_sptr->ALPHA = aLPHA;  // angle subtended by a crystal
      float R = 32.8f;          // ring radius
      cnt_sptr->RE = R + 0.67f; // effective ring radius accounting for the depth of interaction
      cnt_sptr->AXR = SZ_RING;  // axial crystal dim

      cnt_sptr->COSUPSMX = 0.725f;                         // cosine of max allowed scatter angle
      cnt_sptr->COSSTP = (1 - cnt_sptr->COSUPSMX) / (255); // cosine step

      cnt_sptr->TOFBINN = 1;                         // number of TOF bins
      cnt_sptr->TOFBINS = 3.9e-10f;                  // size of TOF bin in [ps]
      float CLGHT = 29979245800.f;                   // speed of light [cm/s]
      cnt_sptr->TOFBIND = cnt_sptr->TOFBINS * CLGHT; // size of TOF BIN in cm of travelled distance
      cnt_sptr->ITOFBIND = 1.f / cnt_sptr->TOFBIND;  // inverse of above

      cnt_sptr->BTP = 0;     // 0: no bootstrapping, 1: no-parametric, 2: parametric (recommended)
      cnt_sptr->BTPRT = 1.f; // ratio of bootstrapped/original events in the target sinogram (1.0 default)

      cnt_sptr->ETHRLD = 0.05f; // intensity percentage threshold of voxels to be considered in the image
    }
  else
    throw std::runtime_error("NiftyPETHelper::getcnst() "
                             "not implemented for scanner type: "
                             + scanner.get_name());
  return cnt_sptr;
}

static inline unsigned
to_1d_idx(const unsigned nrow, const unsigned ncol, const unsigned row, const unsigned col)
{
  return col + ncol * row;
}

template <class dataType>
dataType*
create_heap_array(const unsigned numel, const dataType val = dataType(0))
{
  dataType* array = new dataType[numel];
  std::fill(array, array + numel, val);
  return array;
}

/// Converted from mmraux.py axial_lut
static void
get_axLUT_sptr(shared_ptr<axialLUT>& axlut_sptr,
               std::vector<float>& li2rng,
               std::vector<short>& li2sn_s,
               std::vector<char>& li2nos_c,
               const Cnst& cnt)
{
  const int NRNG = cnt.NRNG;
  int NRNG_c, NSN1_c;

  if (cnt.SPN == 1)
    {
      // number of rings calculated for the given ring range (optionally we can use only part of the axial FOV)
      NRNG_c = cnt.RNG_END - cnt.RNG_STRT;
      // number of sinos in span-1
      NSN1_c = NRNG_c * NRNG_c;
      // correct for the max. ring difference in the full axial extent (don't use ring range (1,63) as for this case no
      // correction)
      if (NRNG_c == 64)
        NSN1_c -= 12;
    }
  else
    {
      NRNG_c = NRNG;
      NSN1_c = cnt.NSN1;
      if (cnt.RNG_END != NRNG || cnt.RNG_STRT != 0)
        throw std::runtime_error("NiftyPETHelper::get_axLUT: the reduced axial FOV only works in span=1.");
    }

  // ring dimensions
  std::vector<float> rng(NRNG * 2);
  float z = -.5f * float(NRNG) * cnt.AXR;
  for (unsigned i = 0; i < unsigned(NRNG); ++i)
    {
      rng[to_1d_idx(NRNG, 2, i, 0)] = z;
      z += cnt.AXR;
      rng[to_1d_idx(NRNG, 2, i, 1)] = z;
    }

  // --create mapping from ring difference to segment number
  // ring difference range
  std::vector<int> rd(2 * cnt.MRD + 1);
  for (unsigned i = 0; i < rd.size(); ++i)
    rd[i] = i - cnt.MRD;
  // ring difference to segment
  std::vector<int> rd2sg(rd.size() * 2, -1);
  // minimum and maximum ring difference for each segment
  std::vector<int> minrd = { -5, -16, 6, -27, 17, -38, 28, -49, 39, -60, 50 };
  std::vector<int> maxrd = { 5, -6, 16, -17, 27, -28, 38, -39, 49, -50, 60 };
  for (unsigned i = 0; i < rd.size(); ++i)
    {
      for (unsigned iseg = 0; iseg < minrd.size(); ++iseg)
        {
          if (rd[i] >= minrd[iseg] && rd[i] <= maxrd[iseg])
            {
              rd2sg[to_1d_idx(rd.size(), 2, i, 0)] = rd[i];
              rd2sg[to_1d_idx(rd.size(), 2, i, 1)] = iseg;
            }
        }
    }

  // create two Michelograms for segments (Mseg)
  // and absolute axial position for individual sinos (Mssrb) which is single slice rebinning
  std::vector<int> Mssrb(NRNG * NRNG, -1);
  std::vector<int> Mseg(NRNG * NRNG, -1);
  for (int r1 = cnt.RNG_STRT; r1 < cnt.RNG_END; ++r1)
    {
      for (int r0 = cnt.RNG_STRT; r0 < cnt.RNG_END; ++r0)
        {
          if (abs(r0 - r1) > cnt.MRD)
            continue;
          int ssp = r0 + r1; // segment sino position (axially: 0-126)
          int rdd = r1 - r0;
          int jseg = -1;
          for (unsigned i = 0; i < rd.size(); ++i)
            if (rd2sg[to_1d_idx(rd.size(), 2, i, 0)] == rdd)
              jseg = rd2sg[to_1d_idx(rd.size(), 2, i, 1)];
          Mssrb[to_1d_idx(NRNG, NRNG, r1, r0)] = ssp;
          Mseg[to_1d_idx(NRNG, NRNG, r1, r0)] = jseg; // negative segments are on top diagonals
        }
    }

  // create a Michelogram map from rings to sino number in span-11 (1..837)
  std::vector<int> Msn(NRNG * NRNG, -1);
  // number of span-1 sinos per sino in span-11
  std::vector<int> Mnos(NRNG * NRNG, -1);
  std::vector<int> seg = { 127, 115, 115, 93, 93, 71, 71, 49, 49, 27, 27 };
  std::vector<int> msk(NRNG * NRNG, 0);
  std::vector<int> Mtmp(NRNG * NRNG);
  int i = 0;
  for (unsigned iseg = 0; iseg < seg.size(); ++iseg)
    {
      // msk = (Mseg==iseg)
      for (unsigned a = 0; a < unsigned(NRNG * NRNG); ++a)
        msk[a] = Mseg[a] == int(iseg) ? 1 : 0;
      // Mtmp = np.copy(Mssrb)
      // Mtmp[~msk] = -1
      for (unsigned a = 0; a < unsigned(NRNG * NRNG); ++a)
        Mtmp[a] = msk[a] ? Mssrb[a] : -1;

      // uq = np.unique(Mtmp[msk])
      std::vector<int> uq;
      for (unsigned a = 0; a < unsigned(NRNG * NRNG); ++a)
        if (msk[a] && std::find(uq.begin(), uq.end(), Mtmp[a]) == uq.end())
          uq.push_back(Mtmp[a]);
      // for u in range(0,len(uq)):
      for (unsigned u = 0; u < uq.size(); ++u)
        {
          // Msn [ Mtmp==uq[u] ] = i
          for (unsigned a = 0; a < unsigned(NRNG * NRNG); ++a)
            if (Mtmp[a] == uq[u])
              Msn[a] = i;
          // Mnos[ Mtmp==uq[u] ] = np.sum(Mtmp==uq[u])
          int sum = 0;
          for (unsigned a = 0; a < unsigned(NRNG * NRNG); ++a)
            if (Mtmp[a] == uq[u])
              ++sum;
          for (unsigned a = 0; a < unsigned(NRNG * NRNG); ++a)
            if (Mtmp[a] == uq[u])
              Mnos[a] = sum;
          ++i;
        }
    }

  //====full LUT
  short* sn1_rno = create_heap_array<short>(NSN1_c * 2, 0);
  short* sn1_ssrb = create_heap_array<short>(NSN1_c, 0);
  short* sn1_sn11 = create_heap_array<short>(NSN1_c, 0);
  char* sn1_sn11no = create_heap_array<char>(NSN1_c, 0);
  int sni = 0; // full linear index, up to 4084
  // michelogram of sino numbers for spn-1
  std::vector<short> Msn1(NRNG * NRNG, -1);
  for (unsigned ro = 0; ro < unsigned(NRNG); ++ro)
    {
      unsigned oblique = ro == 0 ? 1 : 2;
      // for m in range(oblique):
      for (unsigned m = 0; m < oblique; ++m)
        {
          // strt = NRNG*(ro+Cnt['RNG_STRT']) + Cnt['RNG_STRT']
          int strt = NRNG * (ro + cnt.RNG_STRT) + cnt.RNG_STRT;
          int stop = (cnt.RNG_STRT + NRNG_c) * NRNG;
          int step = NRNG + 1;

          // goes along a diagonal started in the first row at r1
          // for li in range(strt, stop, step):
          for (int li = strt; li < stop; li += step)
            {
              int r1, r0;
              // linear indecies of michelogram --> subscript indecies for positive and negative RDs
              if (m == 0)
                {
                  r1 = floor(float(li) / float(NRNG));
                  r0 = li - r1 * NRNG;
                }
              // for positive now (? or vice versa)
              else
                {
                  r0 = floor(float(li) / float(NRNG));
                  r1 = li - r0 * NRNG;
                }
              // avoid case when RD>MRD
              if (Msn[to_1d_idx(NRNG, NRNG, r1, r0)] < 0)
                continue;

              sn1_rno[to_1d_idx(NSN1_c, 2, sni, 0)] = r0;
              sn1_rno[to_1d_idx(NSN1_c, 2, sni, 1)] = r1;

              sn1_ssrb[sni] = Mssrb[to_1d_idx(NRNG, NRNG, r1, r0)];
              sn1_sn11[sni] = Msn[to_1d_idx(NRNG, NRNG, r0, r1)];

              sn1_sn11no[sni] = Mnos[to_1d_idx(NRNG, NRNG, r0, r1)];

              Msn1[to_1d_idx(NRNG, NRNG, r0, r1)] = sni;
              //--
              sni += 1;
            }
        }
    }

  // span-11 sino to SSRB
  // sn11_ssrb = np.zeros(Cnt['NSN11'], dtype=np.int32);
  std::vector<int> sn11_ssrb(cnt.NSN11, -1);
  // sn1_ssrno = np.zeros(Cnt['NSEG0'], dtype=np.int8)
  std::vector<char> sn1_ssrno(cnt.NSEG0, 0);
  // for i in range(NSN1_c):
  for (unsigned i = 0; i < unsigned(NSN1_c); ++i)
    {
      sn11_ssrb[sn1_sn11[i]] = sn1_ssrb[i];
      sn1_ssrno[sn1_ssrb[i]] += 1;
    }

  // sn11_ssrno = np.zeros(Cnt['NSEG0'], dtype=np.int8)
  std::vector<char> sn11_ssrno(cnt.NSEG0, 0);
  // for i in range(Cnt['NSN11']):
  for (unsigned i = 0; i < unsigned(cnt.NSN11); ++i)
    // if sn11_ssrb[i]>0: sn11_ssrno[sn11_ssrb[i]] += 1
    if (sn11_ssrb[i] > 0)
      sn11_ssrno[sn11_ssrb[i]] += 1;

  // sn11_ssrb = sn11_ssrb[sn11_ssrb>=0]
  for (unsigned i = 0; i < unsigned(cnt.NSN11); ++i)
    if (sn11_ssrb[i] < 0)
      sn11_ssrb[i] = 0;

  // ---------------------------------------------------------------------
  // linear index (along diagonals of Michelogram) to rings
  // the number of Michelogram elements considered in projection calculations
  int NLI2R_c = int(float(NRNG_c * NRNG_c) / 2.f + float(NRNG_c) / 2.f);

  // if the whole scanner is used then account for the MRD and subtract 6 ring permutations
  if (NRNG_c == NRNG)
    NLI2R_c -= 6;

  int* li2r = create_heap_array<int>(NLI2R_c * 2);
  // the same as above but to sinos in span-11
  int* li2sn = create_heap_array<int>(NLI2R_c * 2);
  std::vector<short> li2sn1(NLI2R_c * 2);
  li2rng = std::vector<float>(NLI2R_c * 2);
  // ...to number of sinos (nos)
  int* li2nos = create_heap_array<int>(NLI2R_c);

  int dli = 0;
  for (unsigned ro = 0; ro < unsigned(NRNG_c); ++ro)
    {
      // selects the sub-Michelogram of the whole Michelogram
      unsigned strt = NRNG * (ro + cnt.RNG_STRT) + cnt.RNG_STRT;
      unsigned stop = (cnt.RNG_STRT + NRNG_c) * NRNG;
      unsigned step = NRNG + 1;

      // goes along a diagonal started in the first row at r2o
      for (unsigned li = strt; li < stop; li += step)
        {
          // from the linear indexes of Michelogram get the subscript indexes
          unsigned r1 = floor(float(li) / float(NRNG));
          unsigned r0 = li - r1 * NRNG;
          if (Msn[to_1d_idx(NRNG, NRNG, r1, r0)] < 0)
            continue;

          li2r[to_1d_idx(NLI2R_c, 2, dli, 0)] = r0;
          li2r[to_1d_idx(NLI2R_c, 2, dli, 1)] = r1;
          //--//rng[to_1d_idx(NRNG,2,i,1)] = z;
          li2rng[to_1d_idx(NLI2R_c, 2, dli, 0)] = rng[to_1d_idx(NRNG, 2, r0, 0)];
          li2rng[to_1d_idx(NLI2R_c, 2, dli, 1)] = rng[to_1d_idx(NRNG, 2, r1, 0)];
          //--
          li2sn[to_1d_idx(NLI2R_c, 2, dli, 0)] = Msn[to_1d_idx(NRNG, NRNG, r0, r1)];
          li2sn[to_1d_idx(NLI2R_c, 2, dli, 1)] = Msn[to_1d_idx(NRNG, NRNG, r1, r0)];

          li2sn1[to_1d_idx(NLI2R_c, 2, dli, 0)] = Msn1[to_1d_idx(NRNG, NRNG, r0, r1)];
          li2sn1[to_1d_idx(NLI2R_c, 2, dli, 1)] = Msn1[to_1d_idx(NRNG, NRNG, r1, r0)];

          li2nos[dli] = Mnos[to_1d_idx(NRNG, NRNG, r1, r0)];

          ++dli;
        }
    }

  // Need some results in a different data type
  li2sn_s = std::vector<short>(NLI2R_c * 2);
  for (unsigned i = 0; i < unsigned(NLI2R_c * 2); ++i)
    li2sn_s[i] = short(li2sn[i]);
  li2nos_c = std::vector<char>(NLI2R_c);
  for (unsigned i = 0; i < unsigned(NLI2R_c); ++i)
    li2nos_c[i] = char(li2nos[i]);

  // Fill in struct
  axlut_sptr = shared_ptr<axialLUT>(new axialLUT, delete_axialLUT);
  axlut_sptr->li2rno = li2r;           // int   linear indx to ring indx
  axlut_sptr->li2sn = li2sn;           // int   linear michelogram index (along diagonals) to sino index
  axlut_sptr->li2nos = li2nos;         // int   linear indx to no of sinos in span-11
  axlut_sptr->sn1_rno = sn1_rno;       // short
  axlut_sptr->sn1_sn11 = sn1_sn11;     // short
  axlut_sptr->sn1_ssrb = sn1_ssrb;     // short
  axlut_sptr->sn1_sn11no = sn1_sn11no; // char
  // array sizes
  axlut_sptr->Nli2rno[0] = NLI2R_c;
  axlut_sptr->Nli2rno[1] = 2;
  axlut_sptr->Nli2sn[0] = NLI2R_c;
  axlut_sptr->Nli2sn[1] = 2;
  axlut_sptr->Nli2nos = NLI2R_c;
}

static void
get_txLUT_sptr(shared_ptr<txLUTs>& txlut_sptr, std::vector<float>& crs, std::vector<short>& s2c, Cnst& cnt)
{
  txlut_sptr = shared_ptr<txLUTs>(new txLUTs, delete_txLUT);
  *txlut_sptr = get_txlut(cnt);

  s2c = std::vector<short>(txlut_sptr->naw * 2);
  for (unsigned i = 0; i < unsigned(txlut_sptr->naw); ++i)
    {
      s2c[2 * i] = txlut_sptr->s2c[i].c0;
      s2c[2 * i + 1] = txlut_sptr->s2c[i].c1;
    }
  // from mmraux.py
  const float bw = 3.209f; // block width
  // const float dg = 0.474f; // block gap [cm]
  const int NTBLK = 56;
  const float alpha = 2 * M_PI / float(NTBLK); // 2*pi/NTBLK
  crs = std::vector<float>(4 * cnt.NCRS);
  float phi = 0.5f * M_PI - alpha / 2.f - 0.001f;
  for (int bi = 0; bi < NTBLK; ++bi)
    {
      //-tangent point (ring against detector block)
      // ye = RE*np.sin(phi)
      // xe = RE*np.cos(phi)
      float y = cnt.RE * sin(phi);
      float x = cnt.RE * cos(phi);
      //-vector for the face of crystals
      float pv[2] = { -y, x };
      float pv_ = pow(pv[0] * pv[0] + pv[1] * pv[1], 0.5f);
      pv[0] /= pv_;
      pv[1] /= pv_;
      // update phi for next block
      phi -= alpha;
      //-end block points
      float xcp = x + (bw / 2) * pv[0];
      float ycp = y + (bw / 2) * pv[1];
      for (unsigned n = 1; n < 9; ++n)
        {
          int c = bi * 9 + n - 1;
          crs[to_1d_idx(4, cnt.NCRS, 0, c)] = xcp;
          crs[to_1d_idx(4, cnt.NCRS, 1, c)] = ycp;
          float xc = x + (bw / 2 - float(n) * bw / 8) * pv[0];
          float yc = y + (bw / 2 - float(n) * bw / 8) * pv[1];
          crs[to_1d_idx(4, cnt.NCRS, 2, c)] = xc;
          crs[to_1d_idx(4, cnt.NCRS, 3, c)] = yc;
          xcp = xc;
          ycp = yc;
        }
    }
}

void
NiftyPETHelper::set_up()
{
  if (_span < 0)
    throw std::runtime_error("NiftyPETHelper::set_up() "
                             "sinogram span not set.");

  if (_att < 0)
    throw std::runtime_error("NiftyPETHelper::set_up() "
                             "emission or transmission mode (att) not set.");

  if (_scanner_type == Scanner::Unknown_scanner)
    throw std::runtime_error("NiftyPETHelper::set_up() "
                             "scanner type not set.");

  // Get consts
  _cnt_sptr = get_cnst(_scanner_type, _verbose, _devid, _span);

  // Get txLUT
  get_txLUT_sptr(_txlut_sptr, _crs, _s2c, *_cnt_sptr);

  // Get axLUT
  get_axLUT_sptr(_axlut_sptr, _li2rng, _li2sn, _li2nos, *_cnt_sptr);

  switch (_cnt_sptr->SPN)
    {
    case 11:
      _nsinos = _cnt_sptr->NSN11;
      break;
    case 1:
      _nsinos = _cnt_sptr->NSEG0;
      break;
    default:
      throw std::runtime_error("Unsupported span");
    }

  // isub
  _isub = std::vector<int>(unsigned(AW));
  for (unsigned i = 0; i < unsigned(AW); i++)
    _isub[i] = int(i);

  _already_set_up = true;
}

void
NiftyPETHelper::check_set_up() const
{
  if (!_already_set_up)
    throw std::runtime_error("NiftyPETHelper::check_set_up() "
                             "Make sure filenames have been set and set_up has been run.");
}

std::vector<float>
NiftyPETHelper::create_niftyPET_image()
{
  return std::vector<float>(SZ_IMZ * SZ_IMX * SZ_IMY, 0);
}

shared_ptr<VoxelsOnCartesianGrid<float>>
NiftyPETHelper::create_stir_im()
{
  int nz(SZ_IMZ), nx(SZ_IMX), ny(SZ_IMY);
  float sz(SZ_VOXZ * 10.f), sx(SZ_VOXY * 10.f), sy(SZ_VOXY * 10.f);
  shared_ptr<VoxelsOnCartesianGrid<float>> out_im_stir_sptr = MAKE_SHARED<VoxelsOnCartesianGrid<float>>(
      IndexRange3D(0, nz - 1, -(ny / 2), -(ny / 2) + ny - 1, -(nx / 2), -(nx / 2) + nx - 1),
      CartesianCoordinate3D<float>(0.f, 0.f, 0.f),
      CartesianCoordinate3D<float>(sz, sy, sx));
  return out_im_stir_sptr;
}

std::vector<float>
NiftyPETHelper::create_niftyPET_sinogram_no_gaps() const
{
  check_set_up();
  return std::vector<float>(_isub.size() * static_cast<unsigned long>(_nsinos), 0);
}

std::vector<float>
NiftyPETHelper::create_niftyPET_sinogram_with_gaps() const
{
  return std::vector<float>(NSBINS * NSANGLES * unsigned(_nsinos), 0);
}

void
get_stir_indices_and_dims(int stir_dim[3],
                          Coordinate3D<int>& min_indices,
                          Coordinate3D<int>& max_indices,
                          const DiscretisedDensity<3, float>& stir)
{
  if (!stir.get_regular_range(min_indices, max_indices))
    throw std::runtime_error("NiftyPETHelper::set_input - "
                             "expected image to have regular range.");
  for (int i = 0; i < 3; ++i)
    stir_dim[i] = max_indices[i + 1] - min_indices[i + 1] + 1;
}

unsigned
convert_NiftyPET_im_3d_to_1d_idx(const unsigned x, const unsigned y, const unsigned z)
{
  return z * SZ_IMX * SZ_IMY + y * SZ_IMX + x;
}

unsigned
NiftyPETHelper::convert_NiftyPET_proj_3d_to_1d_idx(const unsigned ang, const unsigned bins, const unsigned sino) const
{
  return sino * NSANGLES * NSBINS + ang * NSBINS + bins;
}

void
NiftyPETHelper::permute(std::vector<float>& output_array,
                        const std::vector<float>& orig_array,
                        const unsigned output_dims[3],
                        const unsigned permute_order[3]) const
{
#ifndef NDEBUG
  // Check that in the permute order, each number is between 0 and 2 (can't be <0 because it's unsigned)
  for (unsigned i = 0; i < 3; ++i)
    if (permute_order[i] > 2)
      throw std::runtime_error("Permute order values should be between 0 and 2.");
  // Check that each number is unique
  for (unsigned i = 0; i < 3; ++i)
    for (unsigned j = i + 1; j < 3; ++j)
      if (permute_order[i] == permute_order[j])
        throw std::runtime_error("Permute order values should be unique.");
  // Check that size of output_dims==arr.size()
  assert(orig_array.size() == output_dims[0] * output_dims[1] * output_dims[2]);
  // Check that output array is same size as input array
  assert(orig_array.size() == output_array.size());
#endif

  // Calculate old dimensions
  unsigned old_dims[3];
  for (unsigned i = 0; i < 3; ++i)
    old_dims[permute_order[i]] = output_dims[i];

  // Loop over all elements
  for (unsigned old_1d_idx = 0; old_1d_idx < orig_array.size(); ++old_1d_idx)
    {

      // From the 1d index, generate the old 3d index
      unsigned old_3d_idx[3]
          = { old_1d_idx / (old_dims[2] * old_dims[1]), (old_1d_idx / old_dims[2]) % old_dims[1], old_1d_idx % old_dims[2] };

      // Get the corresponding new 3d index
      unsigned new_3d_idx[3];
      for (unsigned i = 0; i < 3; ++i)
        new_3d_idx[i] = old_3d_idx[permute_order[i]];

      // Get the new 1d index from the new 3d index
      const unsigned new_1d_idx
          = new_3d_idx[0] * output_dims[2] * output_dims[1] + new_3d_idx[1] * output_dims[2] + new_3d_idx[2];

      // Fill the data
      output_array[new_1d_idx] = orig_array[old_1d_idx];
    }
}

void
NiftyPETHelper::remove_gaps(std::vector<float>& sino_no_gaps, const std::vector<float>& sino_w_gaps) const
{
  check_set_up();
  assert(!sino_no_gaps.empty());

  if (_verbose)
    getMemUse();

  ::remove_gaps(sino_no_gaps.data(), const_cast<float*>(sino_w_gaps.data()), _nsinos, _txlut_sptr->aw2ali, *_cnt_sptr);
}

void
NiftyPETHelper::put_gaps(std::vector<float>& sino_w_gaps, const std::vector<float>& sino_no_gaps) const
{
  check_set_up();
  assert(!sino_w_gaps.empty());

  std::vector<float> unpermuted_sino_w_gaps = this->create_niftyPET_sinogram_with_gaps();

  if (_verbose)
    getMemUse();

  ::put_gaps(unpermuted_sino_w_gaps.data(), const_cast<float*>(sino_no_gaps.data()), _txlut_sptr->aw2ali, *_cnt_sptr);

  // Permute the data (as this is done on the NiftyPET python side after put gaps
  unsigned output_dims[3] = { 837, 252, 344 };
  unsigned permute_order[3] = { 2, 0, 1 };
  this->permute(sino_w_gaps, unpermuted_sino_w_gaps, output_dims, permute_order);
}

void
NiftyPETHelper::back_project(std::vector<float>& image, const std::vector<float>& sino_no_gaps) const
{
  check_set_up();
  assert(!image.empty());

  std::vector<float> unpermuted_image = this->create_niftyPET_image();

  if (_verbose)
    getMemUse();

  gpu_bprj(unpermuted_image.data(),
           const_cast<float*>(sino_no_gaps.data()),
           const_cast<float*>(_li2rng.data()),
           const_cast<short*>(_li2sn.data()),
           const_cast<char*>(_li2nos.data()),
           const_cast<short*>(_s2c.data()),
           _txlut_sptr->aw2ali,
           const_cast<float*>(_crs.data()),
           const_cast<int*>(_isub.data()),
           int(_isub.size()),
           AW,
           4, // n0crs
           nCRS,
           *_cnt_sptr);

  // Permute the data (as this is done on the NiftyPET python side after back projection
  unsigned output_dims[3] = { 127, 320, 320 };
  unsigned permute_order[3] = { 2, 0, 1 };
  this->permute(image, unpermuted_image, output_dims, permute_order);
}

void
NiftyPETHelper::forward_project(std::vector<float>& sino_no_gaps, const std::vector<float>& image) const
{
  check_set_up();
  assert(!sino_no_gaps.empty());

  // Permute the data (as this is done on the NiftyPET python side before forward projection
  unsigned output_dims[3] = { 320, 320, 127 };
  unsigned permute_order[3] = { 1, 2, 0 };
  std::vector<float> permuted_image = this->create_niftyPET_image();
  this->permute(permuted_image, image, output_dims, permute_order);

  if (_verbose)
    getMemUse();

  gpu_fprj(sino_no_gaps.data(),
           permuted_image.data(),
           const_cast<float*>(_li2rng.data()),
           const_cast<short*>(_li2sn.data()),
           const_cast<char*>(_li2nos.data()),
           const_cast<short*>(_s2c.data()),
           _txlut_sptr->aw2ali,
           const_cast<float*>(_crs.data()),
           const_cast<int*>(_isub.data()),
           int(_isub.size()),
           AW,
           4, // n0crs
           nCRS,
           *_cnt_sptr,
           _att);
}

shared_ptr<ProjData>
NiftyPETHelper::create_stir_sino()
{
  const int span = 11;
  const int max_ring_diff = 60;
  const int view_mash_factor = 1;
  shared_ptr<ExamInfo> ei_sptr = MAKE_SHARED<ExamInfo>();
  ei_sptr->imaging_modality = ImagingModality::PT;
  shared_ptr<Scanner> scanner_sptr(Scanner::get_scanner_from_name("mMR"));
  int num_views = scanner_sptr->get_num_detectors_per_ring() / 2 / view_mash_factor;
  int num_tang_pos = scanner_sptr->get_max_num_non_arccorrected_bins();
  shared_ptr<ProjDataInfo> pdi_sptr
      = ProjDataInfo::construct_proj_data_info(scanner_sptr, span, max_ring_diff, num_views, num_tang_pos, false);
  shared_ptr<ProjDataInMemory> pd_sptr = MAKE_SHARED<ProjDataInMemory>(ei_sptr, pdi_sptr);
  return pd_sptr;
}

template <class dataType>
static dataType*
read_from_binary_file(std::ifstream& file, const unsigned long num_elements)
{
  // Get current position, get size to end and go back to current position
  const unsigned long current_pos = file.tellg();
  file.seekg(std::ios::cur, std::ios::end);
  const unsigned long remaining_elements = file.tellg() / sizeof(dataType);
  file.seekg(current_pos, std::ios::beg);

  if (remaining_elements < num_elements)
    throw std::runtime_error("File smaller than requested.");

  dataType* contents = create_heap_array<dataType>(num_elements);
  file.read(reinterpret_cast<char*>(contents), num_elements * sizeof(dataType));
  return contents;
}

/// Read numpy file. No error checking here (assume not fortran order etc.)
/// Use std::cout << header if debugging.
static float*
read_numpy_axf1(const unsigned long num_elements)
{
  const char* NP_SOURCE = std::getenv("NP_SOURCE");
  if (!NP_SOURCE)
    throw std::runtime_error("NP_SOURCE not defined, cannot find data");

  std::string numpy_filename = std::string(NP_SOURCE) + "/niftypet/auxdata/AxialFactorForSpan1.npy";
  // Skip over the header (first newline)
  std::ifstream numpy_file(numpy_filename, std::ios::in | std::ios::binary);
  if (!numpy_file.is_open())
    throw std::runtime_error("Failed to open numpy file: " + numpy_filename);

  std::string header;
  std::getline(numpy_file, header);
  // Read
  float* axf1 = read_from_binary_file<float>(numpy_file, num_elements);

  // Close file
  numpy_file.close();

  return axf1;
}

// Taken from mmrnorm.py
static NormCmp
get_norm_helper_struct(const std::string& norm_binary_file, const Cnst& cnt)
{
  // Open the norm binary file
  std::ifstream norm_file(norm_binary_file, std::ios::in | std::ios::binary);
  if (!norm_file.is_open())
    throw std::runtime_error("Failed to open norm binary: " + norm_binary_file);

  NormCmp normc;

  // Dimensions of arrays
  normc.ngeo[0] = cnt.NSEG0;
  normc.ngeo[1] = cnt.W;
  normc.ncinf[0] = cnt.W;
  normc.ncinf[1] = 9;
  normc.nceff[0] = cnt.NRNG;
  normc.nceff[1] = cnt.NCRS;
  normc.naxe = cnt.NSN11;
  normc.nrdt = cnt.NRNG;
  normc.ncdt = 9;

  // geo
  normc.geo = read_from_binary_file<float>(norm_file, normc.ngeo[0] * normc.ngeo[1]);
  // crystal interference
  normc.cinf = read_from_binary_file<float>(norm_file, normc.ncinf[0] * normc.ncinf[1]);
  // crystal efficiencies
  normc.ceff = read_from_binary_file<float>(norm_file, normc.nceff[0] * normc.nceff[1]);
  // axial effects
  normc.axe1 = read_from_binary_file<float>(norm_file, normc.naxe);
  // paralyzing ring DT parameters
  normc.dtp = read_from_binary_file<float>(norm_file, normc.nrdt);
  // non-paralyzing ring DT parameters
  normc.dtnp = read_from_binary_file<float>(norm_file, normc.nrdt);
  // TX crystal DT parameter
  normc.dtc = read_from_binary_file<float>(norm_file, normc.ncdt);
  // additional axial effects
  normc.axe2 = read_from_binary_file<float>(norm_file, normc.naxe);

  // Close file
  norm_file.close();

  // One of the pieces of data is stored as a numpy file. Read it.
  normc.axf1 = read_numpy_axf1(NSINOS);

  return normc;
}

/// Get bucket singles (from mmrhist.py)
std::vector<int>
get_buckets(unsigned int* bck, const unsigned B, const unsigned nitag)
{
  // number of single rates reported for the given second
  // nsr = (hstout['bck'][1,:,:]>>30)
  std::vector<unsigned int> nsr(nitag * B);
  for (unsigned i = 0; i < nitag; ++i)
    for (unsigned j = 0; j < B; ++j)
      nsr[to_1d_idx(nitag, B, i, j)] = bck[nitag * B + to_1d_idx(nitag, B, i, j)] >> 30;

  // average in a second period
  // hstout['bck'][0,nsr>0] /= nsr[nsr>0]
  for (unsigned i = 0; i < nitag; ++i)
    for (unsigned j = 0; j < B; ++j)
      if (nsr[to_1d_idx(nitag, B, i, j)] > 0)
        bck[nitag * B + to_1d_idx(nitag, B, i, j)] /= nsr[to_1d_idx(nitag, B, i, j)];

  // time indices when single rates given
  // tmsk = np.sum(nsr,axis=1)>0
  std::vector<bool> tmsk(nitag, false);
  for (unsigned i = 0; i < nitag; ++i)
    for (unsigned j = 0; j < B; ++j)
      if (nsr[to_1d_idx(nitag, B, i, j)] > 0)
        {
          tmsk[i] = true;
          break;
        }

  // single_rate = np.copy(hstout['bck'][0,tmsk,:])
  std::vector<unsigned int> single_rate;
  for (unsigned i = 0; i < nitag; ++i)
    if (tmsk[i])
      for (unsigned j = 0; j < B; ++j)
        single_rate.push_back(bck[to_1d_idx(nitag, B, i, j)]);
  unsigned sr_dim0 = single_rate.size() / B;

  // get the average bucket singles:
  // buckets = np.int32( np.sum(single_rate,axis=0)/single_rate.shape[0] )
  std::vector<int> buckets(B, 0);
  for (unsigned i = 0; i < sr_dim0; ++i)
    for (unsigned j = 0; j < B; ++j)
      buckets[j] += int(single_rate[to_1d_idx(sr_dim0, B, i, j)]);
  for (unsigned i = 0; i < B; ++i)
    buckets[i] /= sr_dim0;

  return buckets;
}

void
NiftyPETHelper::lm_to_proj_data(shared_ptr<ProjData>& prompts_sptr,
                                shared_ptr<ProjData>& delayeds_sptr,
                                shared_ptr<ProjData>& randoms_sptr,
                                shared_ptr<ProjData>& norm_sptr,
                                const int tstart,
                                const int tstop,
                                const std::string& lm_binary_file,
                                const std::string& norm_binary_file) const
{
  check_set_up();

  // Get LM file as absolute path
  std::string lm_abs = lm_binary_file;
  if (!FilePath::is_absolute(lm_binary_file))
    {
      FilePath fp_lm_binary(lm_binary_file);
      fp_lm_binary.prepend_directory_name(FilePath::get_current_working_directory());
      lm_abs = fp_lm_binary.get_as_string();
    }

  // Get listmode info
  getLMinfo(const_cast<char*>(lm_abs.c_str()), *_cnt_sptr);
  free(lmprop.atag);
  free(lmprop.btag);
  free(lmprop.ele4chnk);
  free(lmprop.ele4thrd);
  free(lmprop.t2dfrm);

  // preallocate all the output arrays - in def.h VTIME=2 (), MXNITAG=5400 (max time 1h30)
  const int nitag = lmprop.nitag;
  const int pow_2_MXNITAG = pow(2, VTIME);
  int tn;
  if (nitag > MXNITAG)
    tn = MXNITAG / pow_2_MXNITAG;
  else
    tn = (nitag + pow_2_MXNITAG - 1) / pow_2_MXNITAG;

  unsigned short frames(0);
  int nfrm(1);

  // structure of output data
  // var   | type               | python var | description                      | shape
  // ------+--------------------|------------+----------------------------------+-----------------------------------------------------------------
  // nitag | int                |            | gets set inside lmproc           |
  // sne   | int                |            | gets set inside lmproc           |
  // snv   | unsigned int *     | pvs        | sino views                       | [ tn,           Cnt['NSEG0'],    Cnt['NSBINS'] ]
  // hcp   | unsigned int *     | phc        | head curve prompts               | [ nitag ] hcd   | unsigned int *     | dhc |
  // head curve delayeds              | [ nitag                                                         ] fan   | unsigned int *
  // | fan        | fansums                          | [ nfrm,         Cnt['NRNG'],     Cnt['NCRS']                    ] bck   |
  // unsigned int *     | bck        | buckets (singles)                | [ 2,            nitag,           Cnt['NBCKT'] ] mss   |
  // float *            | mss        | centre of mass (axially)         | [ nitag ] ssr   | unsigned int *     | ssr        | | [
  // Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']                  ] psn   | void *             | psino      | if nfrm==1,
  // unsigned int*        | [ nfrm,          nsinos,         Cnt['NSANGLES'], Cnt['NSBINS'] ] dsn   | void *             | dsino
  // | if nfrm==1, unsigned int*        | [ nfrm,          nsinos,         Cnt['NSANGLES'], Cnt['NSBINS'] ] psm   | unsigned long
  // long |            | gets set inside lmproc           | dsm   | unsigned long long |            | gets set inside lmproc | tot
  // | unsigned int       |            | gets set inside lmproc           |
  const unsigned int num_sino_elements = _nsinos * _cnt_sptr->A * _cnt_sptr->W;
  hstout dicout;
  dicout.snv = create_heap_array<unsigned int>(tn * _cnt_sptr->NSEG0 * _cnt_sptr->W);
  dicout.hcp = create_heap_array<unsigned int>(nitag);
  dicout.hcd = create_heap_array<unsigned int>(nitag);
  dicout.fan = create_heap_array<unsigned int>(nfrm * _cnt_sptr->NRNG * _cnt_sptr->NCRS);
  dicout.bck = create_heap_array<unsigned int>(2 * nitag * _cnt_sptr->B);
  dicout.mss = create_heap_array<float>(nitag);
  dicout.ssr = create_heap_array<unsigned int>(_cnt_sptr->NSEG0 * _cnt_sptr->A * _cnt_sptr->W);
  if (nfrm == 1)
    {
      dicout.psn = create_heap_array<unsigned short>(nfrm * num_sino_elements);
      dicout.dsn = create_heap_array<unsigned short>(nfrm * num_sino_elements);
    }
  else
    throw std::runtime_error("NiftyPETHelper::lm_to_proj_data: If nfrm>1, "
                             "dicout.psn and dicout.dsn should be unsigned char*. Not "
                             "tested, but should be pretty easy.");

  lmproc(dicout,                            // hstout (struct): output
         const_cast<char*>(lm_abs.c_str()), // char *: binary filename (.s, .bf)
         &frames,                           // unsigned short *: think for one frame, frames = 0
         nfrm,                              // int: num frames
         tstart,                            // int
         tstop,                             // int
         _txlut_sptr->s2cF,                 // *LORcc (struct)
         *_axlut_sptr,                      // axialLUT (struct)
         *_cnt_sptr);                       // Cnst (struct)

  // Convert prompts and delayeds to STIR sinogram
  const unsigned short* psn_int = (const unsigned short*)dicout.psn;
  const unsigned short* dsn_int = (const unsigned short*)dicout.dsn;
  std::vector<float> np_prompts = create_niftyPET_sinogram_with_gaps();
  std::vector<float> np_delayeds = create_niftyPET_sinogram_with_gaps();
  for (unsigned i = 0; i < num_sino_elements; ++i)
    {
      np_prompts[i] = float(psn_int[i]);
      np_delayeds[i] = float(dsn_int[i]);
    }
  prompts_sptr = create_stir_sino();
  delayeds_sptr = create_stir_sino();
  convert_proj_data_niftyPET_to_stir(*prompts_sptr, np_prompts);
  convert_proj_data_niftyPET_to_stir(*delayeds_sptr, np_delayeds);

  // estimated crystal map of singles
  // cmap = np.zeros((Cnt['NCRS'], Cnt['NRNG']), dtype=np.float32)
  std::vector<float> cmap(_cnt_sptr->NCRS * _cnt_sptr->NRNG, 0);

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

  // If norm binary has been supplied, generate the norm sinogram
  if (!norm_binary_file.empty())
    {

      // Get helper
      NormCmp normc = get_norm_helper_struct(norm_binary_file, *_cnt_sptr);

      // Get bucket singles
      std::vector<int> buckets = get_buckets(dicout.bck, unsigned(_cnt_sptr->B), unsigned(nitag));

      std::vector<float> np_norm_no_gaps = this->create_niftyPET_sinogram_no_gaps();

      // Do the conversion
      norm_from_components(np_norm_no_gaps.data(), normc, *_axlut_sptr, _txlut_sptr->aw2ali, buckets.data(), *_cnt_sptr);

      // Add gaps
      std::vector<float> np_norm_w_gaps = this->create_niftyPET_sinogram_with_gaps();
      put_gaps(np_norm_w_gaps, np_norm_no_gaps);

      // Convert to STIR sinogram
      norm_sptr = create_stir_sino();
      convert_proj_data_niftyPET_to_stir(*norm_sptr, np_norm_w_gaps);

      // Clear up
      delete[] normc.geo;
      delete[] normc.cinf;
      delete[] normc.ceff;
      delete[] normc.axe1;
      delete[] normc.dtp;
      delete[] normc.dtnp;
      delete[] normc.dtc;
      delete[] normc.axe2;
      delete[] normc.axf1;
    }

  // Clear up
  delete[] dicout.snv;
  delete[] dicout.hcp;
  delete[] dicout.hcd;
  delete[] dicout.fan;
  delete[] dicout.bck;
  delete[] dicout.mss;
  delete[] dicout.ssr;
  if (nfrm == 1)
    {
      delete[](unsigned short*) dicout.psn;
      delete[](unsigned short*) dicout.dsn;
    }
  else
    throw std::runtime_error("NiftyPETHelper::lm_to_proj_data: If nfrm>1, "
                             "need to cast before deleting as is stored as void*.");
}

void
check_im_sizes(const int stir_dim[3], const int np_dim[3])
{
  for (int i = 0; i < 3; ++i)
    if (stir_dim[i] != np_dim[i])
      throw std::runtime_error(format("NiftyPETHelper::check_im_sizes() - "
                                      "STIR image ({}, {}, {}) should be == ({},{},{}).",
                                      stir_dim[0],
                                      stir_dim[1],
                                      stir_dim[2],
                                      np_dim[0],
                                      np_dim[1],
                                      np_dim[2]));
}

void
check_voxel_spacing(const DiscretisedDensity<3, float>& stir)
{
  // Requires image to be a VoxelsOnCartesianGrid
  const VoxelsOnCartesianGrid<float>& stir_vocg = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(stir);
  const BasicCoordinate<3, float> stir_spacing = stir_vocg.get_grid_spacing();

  // Get NiftyPET image spacing (need to *10 for mm)
  float np_spacing[3] = { 10.f * SZ_VOXZ, 10.f * SZ_VOXY, 10.f * SZ_VOXY };

  for (unsigned i = 0; i < 3; ++i)
    if (std::abs(stir_spacing[int(i) + 1] - np_spacing[i]) > 1e-4f)
      throw std::runtime_error(format("NiftyPETHelper::check_voxel_spacing() - "
                                      "STIR image ({}, {}, {}) should be == ({},{},{}).",
                                      stir_spacing[1],
                                      stir_spacing[2],
                                      stir_spacing[3],
                                      np_spacing[0],
                                      np_spacing[1],
                                      np_spacing[2]));
}

void
NiftyPETHelper::convert_image_stir_to_niftyPET(std::vector<float>& np_vec, const DiscretisedDensity<3, float>& stir)
{
  // Get the dimensions of the input image
  Coordinate3D<int> min_indices;
  Coordinate3D<int> max_indices;
  int stir_dim[3];
  get_stir_indices_and_dims(stir_dim, min_indices, max_indices, stir);

  // NiftyPET requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
  // which at the time of writing was (127,320,320).
  const int np_dim[3] = { SZ_IMZ, SZ_IMX, SZ_IMY };
  check_im_sizes(stir_dim, np_dim);
  check_voxel_spacing(stir);

  // Copy data from STIR to NiftyPET image
  unsigned np_z, np_y, np_x, np_1d;
  for (int z = min_indices[1]; z <= max_indices[1]; z++)
    {
      for (int y = min_indices[2]; y <= max_indices[2]; y++)
        {
          for (int x = min_indices[3]; x <= max_indices[3]; x++)
            {
              // Convert the stir 3d index to a NiftyPET 1d index
              np_z = unsigned(z - min_indices[1]);
              np_y = unsigned(y - min_indices[2]);
              np_x = unsigned(x - min_indices[3]);
              np_1d = convert_NiftyPET_im_3d_to_1d_idx(np_x, np_y, np_z);
              np_vec[np_1d] = stir[z][y][x];
            }
        }
    }
}

void
NiftyPETHelper::convert_image_niftyPET_to_stir(DiscretisedDensity<3, float>& stir, const std::vector<float>& np_vec)
{
  // Get the dimensions of the input image
  Coordinate3D<int> min_indices;
  Coordinate3D<int> max_indices;
  int stir_dim[3];
  get_stir_indices_and_dims(stir_dim, min_indices, max_indices, stir);

  // NiftyPET requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
  // which at the time of writing was (127,320,320).
  const int np_dim[3] = { SZ_IMZ, SZ_IMX, SZ_IMY };
  check_im_sizes(stir_dim, np_dim);
  check_voxel_spacing(stir);

  // Copy data from NiftyPET to STIR image
  unsigned np_z, np_y, np_x, np_1d;
  for (int z = min_indices[1]; z <= max_indices[1]; z++)
    {
      for (int y = min_indices[2]; y <= max_indices[2]; y++)
        {
          for (int x = min_indices[3]; x <= max_indices[3]; x++)
            {
              // Convert the stir 3d index to a NiftyPET 1d index
              np_z = unsigned(z - min_indices[1]);
              np_y = unsigned(y - min_indices[2]);
              np_x = unsigned(x - min_indices[3]);
              np_1d = convert_NiftyPET_im_3d_to_1d_idx(np_x, np_y, np_z);
              stir[z][y][x] = np_vec[np_1d];
            }
        }
    }
}

void
get_vals_for_proj_data_conversion(std::vector<int>& sizes,
                                  std::vector<int>& segment_sequence,
                                  int& num_sinograms,
                                  int& min_view,
                                  int& max_view,
                                  int& min_tang_pos,
                                  int& max_tang_pos,
                                  const ProjDataInfo& proj_data_info,
                                  const std::vector<float>& np_vec)
{
  const ProjDataInfoCylindricalNoArcCorr* info_sptr = dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(&proj_data_info);
  if (is_null_ptr(info_sptr))
    error("NiftyPETHelper: only works with cylindrical projection data without arc-correction");

  segment_sequence = ecat::find_segment_sequence(proj_data_info);
  sizes.resize(segment_sequence.size());
  for (std::size_t s = 0U; s < segment_sequence.size(); ++s)
    sizes[s] = proj_data_info.get_num_axial_poss(segment_sequence[s]);

  // Get dimensions of STIR sinogram
  min_view = proj_data_info.get_min_view_num();
  max_view = proj_data_info.get_max_view_num();
  min_tang_pos = proj_data_info.get_min_tangential_pos_num();
  max_tang_pos = proj_data_info.get_max_tangential_pos_num();

  num_sinograms = proj_data_info.get_num_axial_poss(0);
  for (int s = 1; s <= proj_data_info.get_max_segment_num(); ++s)
    num_sinograms += 2 * proj_data_info.get_num_axial_poss(s);

  int num_proj_data_elems = num_sinograms * (1 + max_view - min_view) * (1 + max_tang_pos - min_tang_pos);

  // Make sure they're the same size
  if (np_vec.size() != unsigned(num_proj_data_elems))
    error(format("NiftyPETHelper::get_vals_for_proj_data_conversion "
                 "NiftyPET and STIR sinograms are different sizes ({} for STIR versus {} for NP",
                 num_proj_data_elems,
                 np_vec.size()));
}

void
get_stir_segment_and_axial_pos_from_NiftyPET_sino(
    int& segment, int& axial_pos, const unsigned np_sino, const std::vector<int>& sizes, const std::vector<int>& segment_sequence)
{
  int z = int(np_sino);
  for (unsigned i = 0; i < segment_sequence.size(); ++i)
    {
      if (z < sizes[i])
        {
          axial_pos = z;
          segment = segment_sequence[i];
          return;
        }
      else
        {
          z -= sizes[i];
        }
    }
}

void
get_NiftyPET_sino_from_stir_segment_and_axial_pos(unsigned& np_sino,
                                                  const int segment,
                                                  const int axial_pos,
                                                  const std::vector<int>& sizes,
                                                  const std::vector<int>& segment_sequence)
{
  np_sino = 0U;
  for (unsigned i = 0; i < segment_sequence.size(); ++i)
    {
      if (segment == segment_sequence[i])
        {
          np_sino += axial_pos;
          return;
        }
      else
        {
          np_sino += sizes[i];
        }
    }
  throw std::runtime_error(
      "NiftyPETHelper::get_NiftyPET_sino_from_stir_segment_and_axial_pos(): Failed to find NiftyPET sinogram.");
}

void
NiftyPETHelper::convert_viewgram_stir_to_niftyPET(std::vector<float>& np_vec, const Viewgram<float>& viewgram) const
{
  // Get the values (and LUT) to be able to switch between STIR and NiftyPET projDatas
  std::vector<int> sizes, segment_sequence;
  int num_sinograms, min_view, max_view, min_tang_pos, max_tang_pos;
  get_vals_for_proj_data_conversion(sizes,
                                    segment_sequence,
                                    num_sinograms,
                                    min_view,
                                    max_view,
                                    min_tang_pos,
                                    max_tang_pos,
                                    *viewgram.get_proj_data_info_sptr(),
                                    np_vec);

  const int segment = viewgram.get_segment_num();
  const int view = viewgram.get_view_num();

  // Loop over the STIR view and tangential position
  for (int ax_pos = viewgram.get_min_axial_pos_num(); ax_pos <= viewgram.get_max_axial_pos_num(); ++ax_pos)
    {

      unsigned np_sino;

      // Convert the NiftyPET sinogram to STIR's segment and axial position
      get_NiftyPET_sino_from_stir_segment_and_axial_pos(np_sino, segment, ax_pos, sizes, segment_sequence);

      for (int tang_pos = min_tang_pos; tang_pos <= max_tang_pos; ++tang_pos)
        {

          unsigned np_ang = unsigned(view - min_view);
          unsigned np_bin = unsigned(tang_pos - min_tang_pos);
          unsigned np_1d = convert_NiftyPET_proj_3d_to_1d_idx(np_ang, np_bin, np_sino);
          np_vec.at(np_1d) = viewgram.at(ax_pos).at(tang_pos);
        }
    }
}

void
NiftyPETHelper::convert_proj_data_stir_to_niftyPET(std::vector<float>& np_vec, const ProjData& stir) const
{
  const int min_view = stir.get_min_view_num();
  const int max_view = stir.get_max_view_num();
  const int min_segment = stir.get_min_segment_num();
  const int max_segment = stir.get_max_segment_num();

  for (int view = min_view; view <= max_view; ++view)
    {
      for (int segment = min_segment; segment <= max_segment; ++segment)
        {
          convert_viewgram_stir_to_niftyPET(np_vec, stir.get_viewgram(view, segment));
        }
    }
}

void
NiftyPETHelper::convert_proj_data_niftyPET_to_stir(ProjData& stir, const std::vector<float>& np_vec) const
{
  // Get the values (and LUT) to be able to switch between STIR and NiftyPET projDatas
  std::vector<int> sizes, segment_sequence;
  int num_sinograms, min_view, max_view, min_tang_pos, max_tang_pos;
  get_vals_for_proj_data_conversion(sizes,
                                    segment_sequence,
                                    num_sinograms,
                                    min_view,
                                    max_view,
                                    min_tang_pos,
                                    max_tang_pos,
                                    *stir.get_proj_data_info_sptr(),
                                    np_vec);

  int segment, axial_pos;
  // Loop over all NiftyPET sinograms
  for (unsigned np_sino = 0; np_sino < unsigned(num_sinograms); ++np_sino)
    {

      // Convert the NiftyPET sinogram to STIR's segment and axial position
      get_stir_segment_and_axial_pos_from_NiftyPET_sino(segment, axial_pos, np_sino, sizes, segment_sequence);

      // Get the corresponding STIR sinogram
      Sinogram<float> sino = stir.get_empty_sinogram(axial_pos, segment);

      // Loop over the STIR view and tangential position
      for (int view = min_view; view <= max_view; ++view)
        {
          for (int tang_pos = min_tang_pos; tang_pos <= max_tang_pos; ++tang_pos)
            {

              unsigned np_ang = unsigned(view - min_view);
              unsigned np_bin = unsigned(tang_pos - min_tang_pos);
              unsigned np_1d = convert_NiftyPET_proj_3d_to_1d_idx(np_ang, np_bin, np_sino);
              sino.at(view).at(tang_pos) = np_vec.at(np_1d);
            }
        }
      stir.set_sinogram(sino);
    }
}

END_NAMESPACE_STIR
