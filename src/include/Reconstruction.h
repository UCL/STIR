#ifndef __RECONSTRUCTION_H__
#define __RECONSTRUCTION_H__

//#include "PARAPET.h"
// CL 25/11 Change <> to ""

#include "Filter.h"
#include <String.h>
#include <strstream.h>
#include "PROMIS/glgtime.h"
#include "sinodata.h"
#include "imagedata.h"
/*
   reconstruction methods are also defined as classes.
   use as follows:

  	PROMIS recon(1,6);
	PETSinogramOfVolume s;
	PETImageVolume v;
		
	recon.reconstruct(s,v);
 */

//KT 17/11 added const references for efficiency only
extern PETSinogramOfVolume& CorrectForAttenuation(const PETSinogramOfVolume& s, const PETSinogramOfVolume& a);
extern PETSegmentBySinogram& CorrectForAttenuation(const PETSegmentBySinogram& s, const PETSegmentBySinogram& a);
 
class PETReconstruction
{
public:
virtual String method_info()
    { return ""; }
virtual String parameter_info()
    { return ""; }
// CL 24/11 ADd reference
virtual void reconstruct(const PETSinogramOfVolume&, PETImageOfVolume&)  = 0;
virtual void  reconstruct(const PETSinogramOfVolume &s, const PETSinogramOfVolume &a, PETImageOfVolume &v) 
    { 
	reconstruct(CorrectForAttenuation(s, a), v); 
    }
};

class PETAnalyticReconstruction: public PETReconstruction
{
public:
  int delta_min;
  int delta_max;
  Filter filter;
virtual String parameter_info();
PETAnalyticReconstruction(int min, int max, Filter f);
};

inline PETAnalyticReconstruction::PETAnalyticReconstruction
     (int min, int max, Filter f) :  delta_min(min), delta_max(max), filter(f)
{};

inline String PETAnalyticReconstruction::parameter_info()
{ char str[100];

  ostrstream s(str, 100);

  s << "delta_min " << delta_min << " delta_max" << delta_max;
  return str;
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
    /*Do something */
}

class PETPROMISReconstruction : public PETAnalyticReconstruction
{
public:
  String method()
    { return("PROMIS"); }
PETPROMISReconstruction(int min, int max, Filter f);
void reconstruct(const PETSinogramOfVolume &s, PETImageOfVolume &v);
};

inline PETPROMISReconstruction::PETPROMISReconstruction(int min, int max, Filter f)
   : PETAnalyticReconstruction(min, max, f)
{
}


/******************* 2D reconstructions ************/
class PETReconstruction2D
{
public:
virtual String method_info()
    { return ""; }
virtual String parameter_info()
    { return ""; }
  //KT 17/11 added references and const
virtual  void reconstruct(const PETSegmentBySinogram& , PETImageOfVolume&) = 0;
virtual void reconstruct(const PETSegmentBySinogram& s, 
			   const PETSegmentBySinogram& a, PETImageOfVolume& v) 
    { reconstruct(CorrectForAttenuation(s, a), v); }
};

class PETAnalyticReconstruction2D: public PETReconstruction2D
{
  Filter1D<float> *filter;
public:
virtual String parameter_info();
PETAnalyticReconstruction2D(Filter1D<float> *f);
};

inline PETAnalyticReconstruction2D::PETAnalyticReconstruction2D
     (Filter1D<float> *f)
     :  filter(f)
{};

inline String PETAnalyticReconstruction2D::parameter_info()
{ char str[100];

  ostrstream s(str, 100);

  //s << "delta_min " << delta_min << " delta_max" << delta_max;
  return str;
}

class Reconstruct2DFBP : public PETAnalyticReconstruction2D
{
 public:
  Reconstruct2DFBP(Filter1D<float> *f): PETAnalyticReconstruction2D(f)
    {}

  // CL 27/10 Add virtual
  // KT 17/11 added references and const to args

virtual void reconstruct(const PETSegmentBySinogram &segment_0, PETImageOfVolume &direct_image);
};


#endif
