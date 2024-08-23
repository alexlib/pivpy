Instructions for the analysis of case B: 
Strong vortex
28.10.2000 		okamoto@tokai.t.u-tokyo.ac.jp

Flow Field
 * The flow field was strong vortex generated on the computer.
 * The field parameter is not known.

Image generation
 * The particle scatter was assumed to be the gaussian distribution.
 * The oclusion particle was considered.
    The intensity of the particles were linerly added.
 * Fill factor was assumed to be 0.7.
 * Particle diameters were randomly determined.
    The histogram of the particle diameter was gaussian distribution.
 * Laser light sheet intensity was assumed to be gaussian.
    Maximum out-of-plane velocity was set to be about 30% of
    laser light thickness.


The reference analysis for this case is :

:ev_IS_size_x                           = 32;
:ev_IS_size_y                           = 32;
:ev_IS_size_unit			= "pixel";
:ev_IS_grid_distance_x                	= 16; 
:ev_IS_grid_distance_y               	= 16; 
:ev_IS_grid_distance_unit		= "pixel"; 
:ev_origin_of_evaluation		= 16, 16;
:ev_origin_of_evaluation_units    	= "pixel";
:ev_IS_offset                        	= 0, 0;
:ev_IS_offset_units                	= "pixel";
:ev_cf_fill_ratio                       = 0.7;



The mandatory data to be provided are :

Raw data : B00?_team_ref_raw.nc

-  Location of the Vortex center (pixel unit)
-  Vorticity at the vortex center (pixel/pixel unit)
-  Circulation (Gamma) at infinity (pixel^2 unit)


