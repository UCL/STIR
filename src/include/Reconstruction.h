// $Id$: $Date$

#ifndef __RECONSTRUCTION_H__
#define __RECONSTRUCTION_H__

#include "pet_common.h"
#include <string>

#include "sinodata.h"
#include "imagedata.h"
#include "Filter.h"

//   reconstruction methods are also defined as classes.
//   use as follows:
//
//  	PETPROMISReconstruction recon(/* appropriate parameters*/);
//	PETSinogramOfVolume s(/* appropriate parameters*/);
//	PETImageVolume v(/* appropriate parameters*/);
//		
//	recon.reconstruct(s,v);


// KT 28/06/98 disabled for now
#if 0
// CL 1/12 These two fonctions should be declared somewherre else
PETSinogramOfVolume CorrectForAttenuation(const PETSinogramOfVolume &sino, const PETSinogramOfVolume &atten)       
{
// Kernel of Attenuation correction
	 PETerror("TODO"); Abort();
	 return sino;
}

PETSegmentBySinogram CorrectForAttenuation(const PETSegmentBySinogram &sino, const PETSegmentBySinogram &atten)
{
// Kernel of Attenuation correction
	PETerror("TODO"); Abort();
	return sino;
}
#endif

class PETReconstruction
{
public:
  // KT 02/06/98 made pure virtual
  virtual string method_info() = 0;

  virtual string parameter_info()
    { return ""; }
  virtual void reconstruct(const PETSinogramOfVolume&, PETImageOfVolume&)  = 0;
// KT 28/06/98 disabled for now
#if 0
  virtual void  reconstruct(const PETSinogramOfVolume &s, const PETSinogramOfVolume &a, PETImageOfVolume &v) 
    { 
	reconstruct(CorrectForAttenuation(s, a), v); 
    }
#endif
};

class PETAnalyticReconstruction: public PETReconstruction
{
public:
  int delta_min;
  int delta_max;
  const Filter1D<float>& filter;
  virtual string parameter_info();
  PETAnalyticReconstruction(int min, int max, const Filter1D<float>& f);
};

inline PETAnalyticReconstruction::PETAnalyticReconstruction
     (int min, int max, const Filter1D<float>& f) :  delta_min(min), delta_max(max), filter(f)
{};

inline string PETAnalyticReconstruction::parameter_info()
{
  // KT 02/06/98 stringstream doesn't understand this
  /*char str[100];

  ostrstream s(str, 100);
  s << "delta_min " << delta_min << " delta_max" << delta_max;
  return str;
  */
  ostrstream s;
  // KT 28/07/98 added 'ends' to make sure the string is null terminated
  s << "delta_min " << delta_min << ", delta_max " << delta_max << ends;
  return s.str();
}

class PETIterativeReconstruction: public PETReconstruction
{
public:
  int max_iterations;
PETIterativeReconstruction(int max);
};

inline PETIterativeReconstruction::PETIterativeReconstruction
     (int max) :max_iterations(max)
{    
}

class PETPROMISReconstruction : public PETAnalyticReconstruction
{
private:
  const int PadS;
  const int PadZ;
  const int disp;
  const int save;
  const bool process_by_view;
  
public:
  // KT 02/06/98 changed from method() to method_info()
  string method_info()
    { return("PROMIS"); }
  PETPROMISReconstruction
    (int min, int max, Filter1D<float> f,
     const int PadS = 0, const int PadZ = 1,
     const bool process_by_view = true, const int disp = 0, const int save = 0);
  void reconstruct(const PETSinogramOfVolume &s, PETImageOfVolume &v);
};

inline PETPROMISReconstruction::PETPROMISReconstruction
    (int min, int max, Filter1D<float> f,
     const int PadS, const int PadZ,
     const bool process_by_view, const int disp, const int save)
   : PETAnalyticReconstruction(min, max, f),
     PadS(PadS), PadZ(PadZ), disp(disp), save(save), process_by_view(process_by_view)
{
}


/******************* 2D reconstructions ************/
class PETReconstruction2D
{
public:
  // KT 02/06/98 made pure virtual
  virtual string method_info() = 0;
  virtual string parameter_info()
    { return ""; }

  //KT 28/07/98 new
  virtual void reconstruct(const PETSinogram &, PETPlane &) = 0;
  //KT 28/07/98 implement this below
  virtual void reconstruct(const PETSegment& , PETImageOfVolume&);
  // KT 28/06/98 disabled for now
#if 0
  virtual void reconstruct(const PETSegment& s, 
			   const PETSegment& a, PETImageOfVolume& v) 
    { reconstruct(CorrectForAttenuation(s, a), v); }
#endif
};

//KT 28/07/98 implement as in FBP2D
void PETReconstruction2D::reconstruct(const PETSegment& sinos, PETImageOfVolume& image)
{
    assert(sinos.min_ring() == image.get_min_z());
    assert(sinos.max_ring() == image.get_max_z());
    assert((sinos.get_num_bins() ==image.get_x_size()) && (image.get_x_size() == image.get_y_size()));
    assert(sinos.get_average_ring_difference() ==0);

    PETPlane image2D=image.get_plane(0);
    for (int  z = sinos.min_ring(); z <= sinos.max_ring(); z++)
    {
       reconstruct(sinos.get_sinogram(z), image2D);
       image.set_plane(image2D, z);
    }
}

class PETAnalyticReconstruction2D: public PETReconstruction2D
{
  // KT 28/07/98 made protected
protected:
  const Filter1D<float>& filter;
public:
  virtual string parameter_info();
  PETAnalyticReconstruction2D(const Filter1D<float>& f);
};

inline PETAnalyticReconstruction2D::PETAnalyticReconstruction2D
     (const Filter1D<float>& f)
     :  filter(f)
{};

inline string PETAnalyticReconstruction2D::parameter_info()
{ char str[100];

  ostrstream s(str, 100);
  // TODO info on filter
  //s << "delta_min " << delta_min << " delta_max" << delta_max;
  return str;
}

class Reconstruct2DFBP : public PETAnalyticReconstruction2D
{
 public:
  Reconstruct2DFBP(const Filter1D<float>& f): PETAnalyticReconstruction2D(f)
    {}

  virtual void reconstruct(const PETSinogram &sino2D, PETPlane &image2D);
};

// KT 28/07/98 implement here, but TODO FBP2D should be changed
#include "recon_buildblock/FBP2D.h"

void Reconstruct2DFBP::reconstruct(const PETSinogram &sino2D, PETPlane &image2D)
{
	FBP2D(sino2D, filter, image2D);
}
    
#endif
