Author:
Kris Thielemans
20 May 2008


This directory contains some documentation on 'external' contributions to STIR
(i.e. not by the original PARAPET partners or Hammersmith Imanet Ltd (HIL)).
Note that the authors have all kindly assigned their copyright to 
HIL, such that HIL con distribute the whole of STIR under a uniform license
condition.

Current contributions integrated into STIR:

- updates to the Shape3D hierarchy 
  by C. Ross Schmidtlein and Assen S. Kirov. 
Added "wedge" capabilities to EllipsoidalCylinder and Box3D.

- parallelisation of OSMAPOSL using MPI and FBP2D using OPEN_MP
  by Tobias Beisel


CVS Info
--------
This is only relevant for people with access to the CVS repository of STIR.

CVS Import for the external contributions normally goes as follows:

- create new directory
- move contributed files to new directory with same sub-directory structure
as the rest of STIR (so don't merge them yet with current version of STIR).
- import into CVS. It is probably a good idea to have different branch-numbers for 
every contributor (to make sure that there are no conflicts between the different
contributions). Example:

cvs -d $CVSROOT import -d -b1.1.3 -m "Import of Shape3D enhancements by C. Ross Schmidtlein, and Assen S. Kirov" parapet RS_AK SHAPE3D_UPDATES_1_00


After this, it is possible to get only the contributed files by doing

cvs -d $CVSROOT checkout -r SHAPE3D_UPDATES_1_00 parapet


You should then be able to merge these files into current STIR using the
cvs update -j options, but I haven't tried that yet. Ideally you would
now a tag from which the contributed files were generated.

