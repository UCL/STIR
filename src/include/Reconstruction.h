#ifndef __RECONSTRUCTION_H__
#define __RECONSTRUCTION_H__

// $Id$ : $Date$

/*
   reconstruction methods are also defined as classes.
   use as follows:

  	PROMIS recon(1,6);
	PETSinogramOfVolume s;
	PETImageVolume v;
		
	recon.reconstruct(s,v);
 */
#include "imagedata.h"
#include "sinodata.h"
#include "filter.h"
#include <strstream.h>

//KT 11/11 glgtime not used here, and it shouldn't be anyway
//#include "glgtime.h"


extern PETSinogramOfVolume& CorrectForAttenuation(PETSinogramOfVolume s, PETSinogramOfVolume a);
extern PETSegmentBySinogram& CorrectForAttenuation(PETSegmentBySinogram& s, PETSegmentBySinogram& a);
 
class PETReconstruction
{
public:
  virtual string method_info()
    { return ""; }
  virtual string parameter_info()
    { return ""; }
  virtual void reconstruct(PETSinogramOfVolume, PETImageOfVolume) 
    = 0;
  virtual void reconstruct(PETSinogramOfVolume s, 
			   PETSinogramOfVolume a, PETImageOfVolume v) 
    { reconstruct(CorrectForAttenuation(s, a), v); }
};

class PETAnalyticReconstruction: public PETReconstruction
{
public:
  int delta_min;
  int delta_max;
  Filter filter;
  virtual string parameter_info();
  PETAnalyticReconstruction(int min, int max, Filter f);
};

inline PETAnalyticReconstruction::PETAnalyticReconstruction
     (int min, int max, Filter f)
     :  delta_min(min), delta_max(max), filter(f)
{};

inline string PETAnalyticReconstruction::parameter_info()
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
     (int max)
     :  max_iterations(max)
{}

class PETPROMISReconstruction : public PETAnalyticReconstruction
{
public:
  string method()
    { return("PROMIS"); }
  PETPROMISReconstruction(int min, int max, Filter f);
  void reconstruct(PETSinogramOfVolume, PETImageOfVolume);
};

inline PETPROMISReconstruction::PETPROMISReconstruction(int min, int max, Filter f)
   : PETAnalyticReconstruction(min, max, f)
{
}


/******************* 2D reconstructions ************/
class PETReconstruction2D
{
public:
  virtual string method_info()
    { return ""; }
  virtual string parameter_info()
    { return ""; }
  virtual void reconstruct(PETSegmentBySinogram, PETImageOfVolume) 
    = 0;
  virtual void reconstruct(PETSegmentBySinogram s, 
			   PETSegmentBySinogram a, PETImageOfVolume v) 
    { reconstruct(CorrectForAttenuation(s, a), v); }
};

class PETAnalyticReconstruction: public PETReconstruction
{
 public:
  int delta_min;
  int delta_max;
  Filter filter;
  virtual string parameter_info();
  PETAnalyticReconstruction(int min, int max, Filter f);
};

inline PETAnalyticReconstruction::PETAnalyticReconstruction
     (int min, int max, Filter f)
       :  delta_min(min), delta_max(max), filter(f)
{};

inline string PETAnalyticReconstruction::parameter_info()
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
     (int max)
       :  max_iterations(max)
{}

class PETPROMISReconstruction : public PETAnalyticReconstruction
{
 public:
  string method()
    { return("PROMIS"); }
  PETPROMISReconstruction(int min, int max, Filter f);
  void reconstruct(PETSinogramOfVolume, PETImageOfVolume);
};

inline PETPROMISReconstruction::PETPROMISReconstruction(int min, int max, Filter f)
  : PETAnalyticReconstruction(min, max, f)
{
}


/******************* 2D reconstructions ************/
class PETReconstruction2D
{
 public:
  virtual string method_info()
    { return ""; }
  virtual string parameter_info()
    { return ""; }
  virtual void reconstruct(PETSegmentBySinogram, PETImageOfVolume) 
    = 0;
  virtual void reconstruct(PETSegmentBySinogram s, 
			      PETSegmentBySinogram a, PETImageOfVolume v) 
    { reconstruct(CorrectForAttenuation(s, a), v); }
};

class PETAnalyticReconstruction2D: public PETReconstruction2D
{
  Filter1D<float> *filter;
 public:
  virtual string parameter_info();
  PETAnalyticReconstruction2D(Filter1D<float> *f);
};

inline PETAnalyticReconstruction2D::PETAnalyticReconstruction2D
     (Filter1D<float> *f)
       :  filter(f)
{};

inline string PETAnalyticReconstruction2D::parameter_info()
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
 
  virtual void reconstruct(PETSegmentBySinogram segment_0, PETImageOfVolume direct_image);
};


#endif
