/*
    Copyright (C) 2022, Matthew Strugari
    Copyright (C) 2014, Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. All rights reserved.
    Copyright (C) 2014, 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

    \author Carles Falcon
    \author Matthew Strugari
*/

// system libraries
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>

// user defined libraries
#include "stir/recon_buildblock/PinholeSPECTUB_Tools.h"
#include "stir/info.h"
#include "stir/error.h"

using std::min;
using std::max;
using std::string;
using std::nothrow;
using std::ifstream;
using std::endl;
using std::vector;
using std::floor;
using std::exit;

namespace SPECTUB_mph
{

#define EPSILON 1e-12

#define maxim(a, b) ((a) >= (b) ? (a) : (b))
#define minim(a, b) ((a) <= (b) ? (a) : (b))
#define abs(a) ((a) >= 0 ? (a) : (-a))
#define SIGN(a) (a < -EPSILON ? -1 : (a > EPSILON ? 1 : 0))

constexpr float dg2rd = static_cast<float>(_PI / 180.);

#define DELIMITER1 '#' // delimiter character in input parameter text file
#define DELIMITER2 '%' // delimiter character in input parameter text file

static std::string wm_SPECT_read_value_1d(std::ifstream* stream1, char DELIMITER);

static void wm_SPECT_read_hvalues_mph(std::ifstream* stream1, char DELIMITER, int* nh, bool do_cyl, wmh_mph_type& wmh);

//=============================================================================
//=== wm_alloc =============================================================
//=============================================================================

void
wm_alloc(const int* Nitems, wm_da_type& wm, const wmh_mph_type& wmh)
{

  //... double array wm.val and wm.col .....................................................

  // if ( ( wm.val = new (nothrow) float * [ wmh.prj.NbOS ] ) == NULL ) error_wmtools_SPECT_mph( 200, wmh.prj.NbOS, "wm.val[]" );
  // if ( ( wm.col = new (nothrow) int   * [ wmh.prj.NbOS ] ) == NULL ) error_wmtools_SPECT_mph( 200, wmh.prj.NbOS, "wm.col[]" );

  //... array wm.ne .........................................................................

  // if ( ( wm.ne = new (nothrow) int [ wmh.prj.NbOS + 1 ] ) == 0 ) error_wmtools_SPECT_mph(200, wmh.prj.NbOS + 1, "wm.ne[]");

  //... memory allocation for wm double arrays ...................................

  for (int i = 0; i < wmh.prj.NbOS; i++)
    {

      if ((wm.val[i] = new (nothrow) float[Nitems[i]]) == NULL)
        error_wmtools_SPECT_mph(200, Nitems[i], "wm.val[][]");
      if ((wm.col[i] = new (nothrow) int[Nitems[i]]) == NULL)
        error_wmtools_SPECT_mph(200, Nitems[i], "wm.col[][]");
    }

  //... to initialize wm to zero ......................

  for (int i = 0; i < wmh.prj.NbOS; i++)
    {

      wm.ne[i] = 0;

      for (int j = 0; j < Nitems[i]; j++)
        {

          wm.val[i][j] = (float)0.;
          wm.col[i][j] = 0;
        }
    }
  wm.ne[wmh.prj.NbOS] = 0;
}

//=============================================================================
//=== precalculated functions ===============================================
//==============================================================================

void
fill_pcf(const wmh_mph_type& wmh, pcf_type& pcf)
{

  //... distribution function for a round shape hole .................

  if (wmh.do_round_cumsum)
    {

      float lngcmd2 = (float)0.5;

      float d1, d2_2, d2;

      pcf.round.res = wmh.highres;
      int dimd2 = (int)floorf(lngcmd2 / pcf.round.res) + 2; // add 2 to have at least one column of zeros as margin
      pcf.round.dim = dimd2 * 2;            // length of the density function (in resolution elements). even number
      lngcmd2 += (float)2. * pcf.round.res; // center of the function

      pcf.round.val = new float*[pcf.round.dim]; // density function allocation

      for (int j = 0; j < pcf.round.dim; j++)
        {

          pcf.round.val[j] = new float[pcf.round.dim];

          d2 = (float)j * pcf.round.res - lngcmd2;
          d2_2 = d2 * d2;

          for (int i = 0; i < pcf.round.dim; i++)
            {

              d1 = (float)i * pcf.round.res - lngcmd2;

              if (sqrtf(d2_2 + d1 * d1) <= (float)0.5)
                pcf.round.val[j][i] = (float)1.;
              else
                pcf.round.val[j][i] = (float)0.;
            }
        }

      calc_cumsum(&pcf.round);

      // cout << "\n\tLength of pcf.round density function: " << pcf.round.dim << endl;
    }

  //... distribution function for a square shape hole ...................

  if (wmh.do_square_cumsum)
    {

      float lngcmd2 = (float)0.5;

      float d1, d2;

      pcf.square.res = wmh.highres;

      int dimd2 = (int)floorf(lngcmd2 / pcf.square.res) + 2; // add 2 to have at least one column of zeros as margin
      pcf.square.dim = dimd2 * 2;            // length of the density function (in resolution elements). even number
      lngcmd2 += (float)2. * pcf.square.res; // center of the function

      pcf.square.val = new float*[pcf.square.dim]; // density function allocation

      for (int j = 0; j < pcf.square.dim; j++)
        {

          pcf.square.val[j] = new float[pcf.square.dim];

          d2 = lngcmd2 - (float)j * pcf.square.res;

          for (int i = 0; i < pcf.square.dim; i++)
            {

              if (fabs(d2) > (float)0.5)
                pcf.square.val[j][i] = (float)0.;
              else
                {
                  d1 = lngcmd2 - (float)i * pcf.square.res;
                  if (fabs(d1) > (float)0.5)
                    pcf.square.val[j][i] = (float)0.;
                  else
                    pcf.square.val[j][i] = (float)1.;
                }
            }
        }

      calc_cumsum(&pcf.square);

      // cout << "\n\tLength of pcf.square density function: " << pcf.square.dim << endl;
    }

  if (wmh.do_depth)
    {

      pcf.cr_att.dim = (int)floorf(wmh.prj.max_dcr / wmh.highres);

      pcf.cr_att.i_max = pcf.cr_att.dim - 1;

      pcf.cr_att.val = new float[pcf.cr_att.dim];

      float stp = wmh.highres * wmh.prj.crattcoef;

      for (int i = 0; i < pcf.cr_att.dim; i++)
        pcf.cr_att.val[i] = expf(-(float)i * stp);

      // cout << "\n\tLength of exponential to correct for crystal attenuation when do_depth: " << pcf.cr_att.dim << endl;
    }
}

//==========================================================================
//=== calc_round_cumsum ===================================================
//==========================================================================

void
calc_cumsum(discrf2d_type* f)
{

  //... cumulative sum by columns ...........................

  for (int j = 0; j < f->dim; j++)
    {

      for (int i = 1; i < f->dim; i++)
        {

          f->val[j][i] = f->val[j][i] + f->val[j][i - 1];
        }
    }

  //... cumulative sum by rows ...............................

  for (int j = 1; j < f->dim; j++)
    {

      for (int i = 0; i < f->dim; i++)
        {

          f->val[j][i] = f->val[j][i] + f->val[j - 1][i];
        }
    }

  //... normalization to one .................................

  float vmax = f->val[f->dim - 1][f->dim - 1];

  for (int j = 0; j < f->dim; j++)
    {

      for (int i = 0; i < f->dim; i++)
        {

          f->val[j][i] /= vmax;
        }
    }

  f->i_max = f->j_max = f->dim - 1;
}

void
read_prj_params_mph(wmh_mph_type& wmh)
{
  string token;
  detel_type d;
  std::stringstream info_stream;

  char DELIMITER = ':';

  ifstream stream1;
  stream1.open(wmh.detector_fn.c_str());
  if (!stream1)
    error_wmtools_SPECT_mph(122, 0, wmh.detector_fn);

  token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
  int Nring = std::stoi(token);

  if (Nring <= 0)
    error_wmtools_SPECT_mph(222, Nring, "Nring");
  /*
  token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
  float FOVcmx = std::stof(token);

  token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
  float FOVcmz = std::stof(token);

  token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
  wmh.prj.Nbin = std::stoi(token);

  token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
  wmh.prj.Nsli = std::stoi(token);
  */
  token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
  wmh.prj.sgm_i = std::stof(token);

  token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
  wmh.prj.crth = std::stof(token);

  token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
  wmh.prj.crattcoef = std::stof(token);

  //... check for parameters ...........................................

  if (wmh.prj.Nbin <= 0)
    error_wmtools_SPECT_mph(190, wmh.prj.Nbin, "Nbin < 1");
  if (wmh.prj.Nsli <= 0)
    error_wmtools_SPECT_mph(190, wmh.prj.Nsli, "Nsli < 1");

  if (wmh.prj.szcm <= 0.)
    error_wmtools_SPECT_mph(190, wmh.prj.szcm, "szcm non positive");
  if (wmh.prj.thcm <= 0.)
    error_wmtools_SPECT_mph(190, wmh.prj.thcm, "thcm non positive");

  if (wmh.prj.rad <= 0.)
    error_wmtools_SPECT_mph(190, wmh.prj.rad, "Drad non positive");
  if (wmh.prj.sgm_i < 0.)
    error_wmtools_SPECT_mph(190, wmh.prj.sgm_i, "PSF int: sigma non positive");

  //... derived variables .......................

  wmh.prj.radc = wmh.prj.rad + wmh.prj.crth;
  wmh.prj.crth_2 = wmh.prj.crth * wmh.prj.crth;

  if (!wmh.do_depth)
    wmh.prj.rad += wmh.prj.crth / (float)2.; // setting detection plane at half of the crystal thickness

  //... print out values (to comment or remove)..............................

  info_stream << "Projection parameters" << endl;
  info_stream << "Number of rings: " << Nring << endl;
  info_stream << "Radius (cm): " << wmh.prj.rad << endl;
  info_stream << "FOVcmx (cm): " << wmh.prj.FOVxcmd2 * 2. << endl;
  info_stream << "FOVcmz (cm): " << wmh.prj.FOVzcmd2 * 2. << endl;
  info_stream << "Number of bins: " << wmh.prj.Nbin << endl;
  info_stream << "Number of slices: " << wmh.prj.Nsli << endl;
  info_stream << "Bin size (cm): " << wmh.prj.szcm << endl;
  info_stream << "Slice thickness (cm): " << wmh.prj.thcm << endl;
  info_stream << "Intrinsic PSF sigma (cm): " << wmh.prj.sgm_i << endl;
  info_stream << "Crystal thickness (cm): " << wmh.prj.crth << endl;

  //... for each ring ..............................

  wmh.prj.Ndt = 0;

  for (int i = 0; i < Nring; i++)
    {

      token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
      int Nang = std::stoi(token);
      if (Nang <= 0)
        error_wmtools_SPECT_mph(190, Nang, "Nang < 1");

      token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
      float ang0 = std::stof(token);

      token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
      float incr = std::stof(token);

      token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
      d.z0 = std::stof(token);

      d.nh = 0;

      for (int j = 0; j < Nang; j++)
        {

          //... angles and ratios ................................................

          d.theta = (ang0 + (float)j * incr) * dg2rd; // projection angle in radians
          d.costh = std::cos(d.theta);                    // cosinus of the angle
          d.sinth = std::sin(d.theta);                    // sinus of the angle

          //... cartesian coordinates of the center of the detector element .............

          d.x0 = wmh.prj.rad * d.costh;
          d.y0 = wmh.prj.rad * d.sinth;

          //... coordinates of the first bin of each projection and increments for consecutive bins ....

          if (wmh.do_att)
            {

              d.incx = wmh.prj.szcm * d.costh;
              d.incy = wmh.prj.szcm * d.sinth;
              d.incz = wmh.prj.thcm;

              d.xbin0 = -wmh.prj.rad * d.sinth - (wmh.prj.FOVxcmd2 + wmh.prj.szcm * (float)0.5) * d.costh;
              d.ybin0 = wmh.prj.rad * d.costh - (wmh.prj.FOVxcmd2 + wmh.prj.szcm * (float)0.5) * d.sinth;
              d.zbin0 = d.z0 - wmh.prj.FOVzcmd2 + wmh.prj.thcmd2;
            }
          wmh.detel.push_back(d);
        }

      //... update of wmh cumulative values .....................................

      wmh.prj.Ndt += Nang;

      //... print out values (to comment or remove)..............................

      info_stream << "\nDetector ring: " << i << endl;
      info_stream << "Number of angles: " << Nang << endl;
      info_stream << "ang0: " << ang0 << endl;
      info_stream << "incr: " << incr << endl;
      info_stream << "z0: " << d.z0 << endl;
      info_stream << "Number of holes per detel: " << d.nh << endl;
    }

  //... fill detel .....................................................

  stream1.close();

  wmh.prj.Nbt = wmh.prj.Nbd * wmh.prj.Ndt;

  info_stream << "\nTotal number of detels: " << wmh.prj.Ndt << endl;
  info_stream << "Total number of bins: " << wmh.prj.Nbt << endl;

  stir::info(info_stream.str());

  return;
}

///=============================================================================
//=== read collimator params mph ===============================================
//==============================================================================

void
read_coll_params_mph(wmh_mph_type& wmh)
{
  string token;
  vector<string> param;
  std::stringstream info_stream;

  char DELIMITER = ':';

  ifstream stream1;
  stream1.open(wmh.collim_fn.c_str());

  if (!stream1)
    error_wmtools_SPECT_mph(122, 0, wmh.collim_fn);

  wmh.collim.model = wm_SPECT_read_value_1d(&stream1, DELIMITER);

  token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
  wmh.collim.rad = std::stof(token);

  token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
  wmh.collim.L = std::stof(token);
  wmh.collim.Ld2 = wmh.collim.L / (float)2.;

  token = wm_SPECT_read_value_1d(&stream1, DELIMITER);
  wmh.collim.Nht = std::stoi(token);

  //    wmh.collim.holes = new hole_type [ wmh.collim.Nht ];

  int nh = 0;
  if (wmh.collim.model == "cyl")
    wm_SPECT_read_hvalues_mph(&stream1, DELIMITER, &nh, true, wmh);
  else
    {
      if (wmh.collim.model == "pol")
        wm_SPECT_read_hvalues_mph(&stream1, DELIMITER, &nh, false, wmh);
      else
        error_wmtools_SPECT_mph(334, 0, wmh.collim.model);
    }

  if (nh != wmh.collim.Nht)
    error_wmtools_SPECT_mph(150, nh, "");

  //... check for parameters ...........................................

  if (wmh.collim.rad <= 0.)
    error_wmtools_SPECT_mph(190, wmh.collim.rad, "Collimator radius non positive");

  if (wmh.collim.Nht <= 0)
    error_wmtools_SPECT_mph(190, wmh.collim.Nht, "Number of Holes < 1");

  //... print out values (to comment or remove)..............................

  stream1.close();

  info_stream << "Collimator parameters" << endl;
  info_stream << "\nCollimator model: " << wmh.collim.model << endl;
  info_stream << "Collimator rad: " << wmh.collim.rad << endl;
  info_stream << "Number of holes: " << wmh.collim.Nht << endl;

  stir::info(info_stream.str());

  return;
}

//=====================================================================
//======== wm_SPECT_read_hvalues_mph ==============================
//=====================================================================

void
wm_SPECT_read_hvalues_mph(ifstream* stream1, char DELIMITER, int* nh, bool do_cyl, wmh_mph_type& wmh)
{

  size_t pos1, pos2, pos3;
  string line, token;
  hole_type h;

  float max_hsxcm = (float)0.;
  float max_hszcm = (float)0.;

  float max_aix = (float)0.;
  float max_aiz = (float)0.;

  *nh = 0;

  while (getline(*stream1, line))
    {

      pos1 = line.find(DELIMITER);

      if (pos1 == string::npos)
        continue;

      //... detel index ...................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos || pos3 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "idet");
      token = line.substr(pos2, pos3 - pos2);
      int idet = std::stoi(token) - 1;
      wmh.detel[idet].who.push_back(*nh);
      wmh.detel[idet].nh++;
      pos1 = pos3;

      //... second parameter ...........................

      if (do_cyl)
        {

          //... angle ...........................

          pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
          pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
          if (pos2 == string::npos || pos3 == string::npos)
            error_wmtools_SPECT_mph(333, *nh, "angle(deg)");
          token = line.substr(pos2, pos3 - pos2);
          h.acy = std::stof(token) * dg2rd;
          pos1 = pos3;
        }
      else
        {

          //... x position ...........................

          pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
          pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
          if (pos2 == string::npos || pos3 == string::npos)
            error_wmtools_SPECT_mph(333, *nh, "x(cm)");
          token = line.substr(pos2, pos3 - pos2);
          h.x1 = std::stof(token);
          pos1 = pos3;
        }

      //... y position (along collimator wall. 0 for centrer of collimator wall) ...................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos || pos3 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "y(cm)");
      token = line.substr(pos2, pos3 - pos2);
      float yd = std::stof(token);
      pos1 = pos3;

      //... z position ...................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos || pos3 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "z(cm)");
      token = line.substr(pos2, pos3 - pos2);
      h.z1 = std::stof(token);
      pos1 = pos3;

      //... shape .............................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos || pos3 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "shape");
      token = line.substr(pos2, pos3 - pos2);
      if (token.compare("rect") != 0 && token.compare("round") != 0)
        error_wmtools_SPECT_mph(444, *nh, "");
      h.shape = token.c_str();
      pos1 = pos3;

      if (token.compare("rect") == 0)
        {
          wmh.do_square_cumsum = true;
          h.do_round = false;
        }
      else
        {
          wmh.do_round_cumsum = true;
          h.do_round = true;
        }

      //... dimension x cm .......................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "dxcm");
      token = line.substr(pos2, pos3 - pos2);
      h.dxcm = std::stof(token);
      if (h.dxcm > max_hsxcm)
        max_hsxcm = h.dxcm;
      pos1 = pos3;

      //... dimension z cm .......................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "dzcm");
      token = line.substr(pos2, pos3 - pos2);
      h.dzcm = std::stof(token);
      if (h.dzcm > max_hszcm)
        max_hszcm = h.dzcm;
      pos1 = pos3;

      //... hole axial angle x .......................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "ahx");
      token = line.substr(pos2, pos3 - pos2);
      h.ahx = std::stof(token) * dg2rd;
      pos1 = pos3;

      //... hole axial angle z .......................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "ahz");
      token = line.substr(pos2, pos3 - pos2);
      h.ahz = std::stof(token) * dg2rd;
      pos1 = pos3;

      //... x acceptance angle ........................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "aa_x");
      token = line.substr(pos2, pos3 - pos2);
      h.aa_x = std::stof(token) * dg2rd;
      pos1 = pos3;

      //... z acceptance angle ........................

      pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
      pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
      if (pos2 == string::npos)
        error_wmtools_SPECT_mph(333, *nh, "aa_z");
      token = line.substr(pos2, pos3 - pos2);
      h.aa_z = std::stof(token) * dg2rd;

      //... derived variables ........................................

      if (do_cyl)
        {
          h.acyR = h.acy - wmh.detel[idet].theta;
          h.x1 = (wmh.collim.rad + yd) * std::sin(h.acyR);
          h.y1 = (wmh.collim.rad + yd) * std::cos(h.acyR);
        }
      else
        {
          h.y1 = wmh.collim.rad + yd;
        }

      h.z1 = wmh.detel[idet].z0 + h.z1;

      //... edge slope x,z minor and major .......................

      h.Egx = h.ahx + h.aa_x;
      h.egx = h.ahx - h.aa_x;
      h.Egz = h.ahz + h.aa_z;
      h.egz = h.ahz - h.aa_z;

      //... angles max and min ..........................................

      h.ax_M = h.Egx;
      h.ax_m = h.egx;
      h.az_M = h.Egz;
      h.az_m = h.egz;

      //... incidence angle maximum, for PSF allocation when correction for depth...............

      if (fabs(h.ax_m) > max_aix)
        max_aix = h.ax_m;
      if (fabs(h.ax_M) > max_aix)
        max_aix = h.ax_M;

      if (fabs(h.az_m) > max_aiz)
        max_aiz = h.az_m;
      if (fabs(h.az_M) > max_aiz)
        max_aiz = h.az_M;

      wmh.collim.holes.push_back(h);

      *nh = *nh + 1;
    }

  //... maximum hole dimensions and incidence angles .................

  wmh.max_hsxcm = max_hsxcm;
  wmh.max_hszcm = max_hszcm;

  wmh.tmax_aix = std::tan(max_aix);
  wmh.tmax_aiz = std::tan(max_aiz);

  wmh.prj.max_dcr = (float)1.2 * wmh.prj.crth / cosf(max(max_aix, max_aiz));
}

//=============================================================================
//=== generate_msk_mph ========================================================
//=============================================================================

void
generate_msk_mph(bool* msk_3d, const float* attmap, const wmh_mph_type& wmh)
{

  //    bool do_save_resulting_msk = true;

  //... to create mask from attenuation map ..................

  if (wmh.do_msk_att)
    {
      for (int i = 0; i < wmh.vol.Nvox; i++)
        {
          msk_3d[i] = (attmap[i] > EPSILON);
        }
    }
  else
    {
      //... to read a mask from a (int) file ....................

      if (wmh.do_msk_file)
        stir::error("Mask incorrectly read from file."); // read_msk_file_mph( msk_3d );  // STIR implementation never calls this
                                                         // to avoid using read_msk_file_mph

      else
        {

          //... to create a cylindrical mask......................

          float xi2, yi2;

          float Rmax2 = wmh.ro * wmh.ro; // Maximum allowed radius (distance from volume centre)

          for (int j = 0, ip = 0; j < wmh.vol.Dimy; j++)
            {

              yi2 = ((float)j + (float)0.5) * wmh.vol.szcm - wmh.vol.FOVcmyd2;
              yi2 *= yi2;

              for (int i = 0; i < wmh.vol.Dimx; i++, ip++)
                {

                  xi2 = ((float)i + (float)0.5) * wmh.vol.szcm - wmh.vol.FOVxcmd2;
                  xi2 *= xi2;

                  if ((xi2 + yi2) > Rmax2)
                    {

                      for (int k = 0; k < wmh.vol.Dimz; k++)
                        msk_3d[ip + k * wmh.vol.Npix] = false;
                    }
                  else
                    {
                      for (int k = 0; k < wmh.vol.Dimz; k++)
                        msk_3d[ip + k * wmh.vol.Npix] = true;
                    }
                }
            }
        }
    }
}

//=====================================================================
//======== wm_SPECT_read_value_1d =====================================
//=====================================================================

string
wm_SPECT_read_value_1d(ifstream* stream1, char DELIMITER)
{

  size_t pos1, pos2, pos3;
  string line;

  int k = 0;

  while (!stream1->eof())
    {
      getline(*stream1, line);

      pos1 = line.find(DELIMITER);

      if (pos1 != string::npos)
        {
          k++;
          break;
        }
    }

  if (k == 0)
    {
      // error_wmtools_SPECT_mph(888, 0, "");
      stir::error("Error wm_SPECT: missing parameter in collimator file");
    }

  pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1);
  pos3 = line.find_first_of(" \t\f\v\n\r", pos2);

  return (line.substr(pos2, pos3 - pos2));
}

//=============================================================================
//== error_wmtools_SPECT_mph ======================================================
//=============================================================================

void
error_wmtools_SPECT_mph(int nerr, int ip, string txt)
{
  using stir::error;
  switch (nerr)
    {

    case 55:
      printf("\n\nError %d weight3d: Dowmsampling. Incongruency factor-dim: %d \n", nerr, ip);
      break;
    case 56:
      printf("\n\nError %d weight3d: Downsampling. Resulting dim bigger than max %d \n", nerr, ip);
      break;
    case 77:
      printf("\n\nError %d weight3d: Convolution. psf_out is not big enough %d. Verify geometry. \n", nerr, ip);
      break;
    case 78:
      printf("\n\nError %d weight3d: Geometric PSF. psf_out is not big enough %d. Verify geometry. \n", nerr, ip);
      break;

      //... error: value of argv[]..........................

    case 122:
      printf("\n\nError wm_SPECT: File with variable parameters: %s not found.\n", txt.c_str());
      break;
    case 124:
      printf("\n\nError wm_SPECT: Cannot open attenuation map: %s for reading..\n", txt.c_str());
      break;
    case 126:
      printf("\n\nError wm_SPECT: Cannot open file mask: %s for reading\n", txt.c_str());
      break;
    case 150:
      printf("\n\nError wm_SPECT: List of hole parameters has different length (%d) than number of holes.\n", ip);
      break;
    case 190:
      printf("\n\nError wm_SPECT: Wrong value in detector parameter: %s \n", txt.c_str());
      break;
    case 200:
      printf("\n\nError wm_SPECT: Cannot allocate %d element of the variable: %s\n", ip, txt.c_str());
      break;
    case 222:
      printf("\n\nError wm_SPECT: Wrong number of rings: %d\n", ip);
      break;
    case 333:
      printf("\n\nError wm_SPECT: Missing parameter in hole %d definition: %s\n", ip, txt.c_str());
      break;
    case 334:
      printf("\n\nError wm_SPECT: %s unknown collimator model. Options: cyl/pol.\n", txt.c_str());
      break;
    case 444:
      printf("\n\nError wm_SPECT: Hole %d: Wrong hole shape. Hole shape should be either rect or round.\n", ip);
      break;
    case 888:
      error("\n\nError wm_SPECT: Missing parameter in collimator file.\n");
      break;
    default:
      printf("\n\nError wmtools_SPECT: %d unknown error number on error_wmtools_SPECT().", nerr);
    }

  exit(0);
}

} // namespace SPECTUB_mph
