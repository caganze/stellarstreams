# generating the stream that is used in the N-body simulation
df = ms.FardalStreamDF()
        
gd1_init = gc.GD1Koposov10(phi1 = -13*u.degree, phi2=0*u.degree, distance=8.84*u.kpc,
                          pm_phi1_cosphi2=-10.28*u.mas/u.yr,
                          pm_phi2=-2.43*u.mas/u.yr,
                          radial_velocity = -182*u.km/u.s)
rep = gd1_init.transform_to(coord.Galactocentric).data
gd1_w0 = gd.PhaseSpacePosition(rep)
gd1_mass = 5e3 * u.Msun
gd1_pot = gp.PlummerPotential(m=gd1_mass, b=5*u.pc, units=galactic)
mw = gp.MilkyWayPotential()
gen_gd1 = ms.MockStreamGenerator(df, mw, progenitor_potential=gd1_pot)
stream, nbody = gen_gd1.run(gd1_w0, gd1_mass,
                                dt=-1 * u.Myr, n_steps=3000)

w0_now = gd.PhaseSpacePosition(stream.data, stream.vel)


orbit = mw.integrate_orbit(w0_now, dt=-1*u.Myr, n_steps=int(t_int))
old_stream = orbit[-1]

site_at_impact_w0 = gd.PhaseSpacePosition(pos=np.mean(old_stream.pos), vel=[vxstream, vystream, vzstream]*u.km/u.s)
#vxstream, vystream, vzstream not defined in this file but its just the velocity of the stream

# ---------------------------------------------------#


### These are the important definitions. The lines above are just to show how the stuff used in here are originally defined
def get_cyl_rotation(): #borrowed from Adrian Price-Whelan's streampunch github repo
    L = site_at_impact_w0.angular_momentum()
    v = site_at_impact_w0.v_xyz

    new_z = v / np.linalg.norm(v, axis=0)
    new_x = L / np.linalg.norm(L, axis=0)
    new_y = -np.cross(new_x, new_z)
    R = np.stack((new_x, new_y, new_z))
    return R

def get_perturber_w0_at_impact():

    # Get the rotation matrix to rotate from Galactocentric to cylindrical
    # impact coordinates at the impact site along the stream
    R = get_cyl_rotation()

    b, psi, z, v_z, vpsi = b * u.pc, psi * u.deg, z * u.kpc, v_z * u.km/u.s, vpsi * u.km/u.s

    # Define the position of the perturber at the time of impact in the
    # cylindrical impact coordinates:
    perturber_pos = coord.CylindricalRepresentation(rho=b,
                                                    phi=psi,
                                                    z=z)

    # Define the velocity in the cylindrical impact coordinates:
    perturber_vel = coord.CylindricalDifferential(
        d_rho=0*u.km/u.s,  # Fixed by definition: b is closest approach
        d_phi=(vpsi / b).to(u.rad/u.Myr, u.dimensionless_angles()),
        d_z=v_z)

    # Transform from the cylindrical impact coordinates to Galactocentric
    perturber_rep = perturber_pos.with_differentials(perturber_vel)
    perturber_rep = perturber_rep.represent_as(
        coord.CartesianRepresentation, coord.CartesianDifferential)
    perturber_rep = perturber_rep.transform(R.T)

    pos = perturber_rep.without_differentials() + site_at_impact_w0.pos
    vel = perturber_rep.differentials['s'] + site_at_impact_w0.vel

    # This should be in Galactocentric Cartesian coordinates now!
    return gd.PhaseSpacePosition(pos, vel)