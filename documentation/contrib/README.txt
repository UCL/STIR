Author:
Kris Thielemans
20 May 2008


This directory contains some documentation on 'external' contributions to STIR
(i.e. not by the original PARAPET partners or Hammersmith Imanet Ltd (HIL)).

Note that up to 2011, the authors have all kindly assigned their copyright to 
HIL, such that HIL can distribute the whole of STIR under a uniform license
condition.

Current contributions listed here that were integrated into STIR (see sub-directories for more info):

- updates to the Shape3D hierarchy 
  by C. Ross Schmidtlein and Assen S. Kirov. 
Added "wedge" capabilities to EllipsoidalCylinder and Box3D.

- parallelisation of OSMAPOSL using MPI and FBP2D using OPEN_MP
  by Tobias Beisel

- code for SimSET support
  by Pablo Aguiar, Nikolaos Dikaios, Charalampos Tsoumpas, Kris Thielemans

- code for (spatial) warping and motion correction (and GIPL IO and raw GATE projection data conversion) (licensed under LGPL)
  by Charalampos Tsoumpas

- code for reading the SAFIR coincidence listmode format and sorting the events
  into a virtual cylindrical scanner for reconstruction (licensed under Apache2).
  by Jannis Fischer
