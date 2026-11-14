# Simplified artificial-gravity dynamics

The spacecraft spins about a fixed inertial X axis. The crew is stationary relative to the rotating ring at reference radius `R`; the engine produces signed axial acceleration. Gravity below is **apparent** (non-inertial acceleration experienced in the cabin).

### Steady spin

`a_radial = R * omega_ring_absolute**2`. Because the ring/hub motor exerts equal and opposite torques, the hub can move: **absolute ring speed** determines centrifugal acceleration, not the ring encoder's ring-minus-hub relative speed.

For purely axial engine thrust perpendicular to the ring's radial direction:

`g_eff = sqrt(axial_accel**2 + (R*omega**2)**2)`.

The ideal steady target `omega = sqrt( sqrt(g_target**2 - axial_accel**2) / R )` is feasible only if `abs(axial_accel) <= g_target`.

### Spin transitions

The state is `(theta_ring,theta_hub,omega_ring,omega_hub,commanded_torque,spacecraft_position,spacecraft_velocity)` and Newtonian inertial equations use:

```
ring_alpha = (tau_motor - b*(omega_ring-omega_hub) + tau_external_ring)/I_ring
hub_alpha = (-tau_motor + b*(omega_ring-omega_hub) + tau_external_hub)/I_hub
```

The actuator is bounded by torque, rate of change and electrical power. Numerical integration in the reference simulator uses symplectic Euler for angular rates and trapezoidal/constant-acceleration translation.

For a ring-fixed crew, the local apparent down-vector in radial, tangential, and axial coordinates is:

```
G = [R*omega_ring**2, -R*alpha_ring, -axial_acceleration]
```

Hence `|G|^2 = R^2*omega^4 + R^2*alpha^2 + axial_acceleration^2` under these simplifying assumptions. The head-to-foot gradient is computed at R vs R - 1.75m. Crew motion adds Coriolis terms; for spin axis `+X`, in the radial/tangential plane `a_coriolis = (2*omega*v_tan, -2*omega*v_rad)`.

### Critical limitations

- The craft has one fixed spin axis; gyroscopic coupling, full 6-DOF attitude, flexural vibrations, fuel mass/slosh and docking have not been solved.
- Torque/power/inertias are assumed **illustrative engineering parameters**, not hardware characterizations.
- At zero spin without engine thrust, local artificial gravity is zero.
- Total force magnitude equaling 1G does not imply tolerable loading or a correctly oriented crew floor.
- No Earth-to-Mars orbital mechanics is implemented; the mission is a signed 1D acceleration test.
- A biological health response cannot be calculated by this mechanics-only model.

The 3D scene uses normalized drawing coordinates while its rotation angle comes from the integrated **physical** ring speed. Playback rate and perspective camera do not feed back into the equations.
