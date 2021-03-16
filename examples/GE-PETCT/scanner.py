>>> import stir
p=stir.ProjData.read_from_file('rdf_f1b1.hs')
p=stir.ProjData.read_from_file('output/fullefffactorsspan2.hs')
s=p.get_proj_data_info().get_scanner()

% not for ST
import math
import numpy
s=stir.Scanner.get_scanner_from_name('Discovery MI3')
#s=stir.Scanner.get_scanner_from_name('GE Signa PET/MR')
nc=s.get_num_crystals_per_ring()
nc=s.get_num_detectors_per_ring()
ncb=s.get_num_transaxial_crystals_per_block()
nbb=s.get_num_transaxial_blocks_per_bucket()
R=s.get_effective_ring_radius()
crystal_size=2*math.pi*R/nc
print(crystal_size)
#crystal_size=4.2


print(numpy.arctan(ncb*nbb/2*crystal_size / R)*180/math.pi)
print(s.get_intrinsic_azimuthal_tilt()*180/math.pi)
