# Phase-Field-Dislocation-Dynamics-(PFDD)
Phase Field Dislocation Dynamics (PFDD) 

Open Source Number: C17113

Project Scope:

Phase Field Dislocation Dynamics (PFDD) is a phase field approach for tracking the motion and interactions of individual dislocations in materials, mostly face-centered cubic (fcc) metals. The phase field approach relies on scalar order parameters or phase field variables to track a quantity of interest.  In PFDD, these phase field variables represent the passage of perfect dislocations across each active slip system, with one order parameter defined in each slip system.  Partial dislocations are represented with linear combinations of the order parameters.  The system is evolved through minimization of the total system energy, which is directly dependent on the phase field variables.  All of the physics is described by the total energy functional that is derived from a thermodynamic basis, and the system kinetics are evolved using a Ginzburg-Landau (GL) equation

Thus far, the codes here have been primarily used by a small group of researchers to study dislocation-dislocation interactions in metals, and the impact these interactions have on the overall deformation and material response of these materials.  Some related references including details about this formulation are:

1.	M. Koslowski, A.M. Cuitino, M. Ortiz, “A phase-field theory of dislocation dynamics, strain hardening and hystersis in ductile single crystals”, Journal of the Mechanics and Physics of Solids, 50 (2002) 2597-2635.
2.	A. Hunter, I. J. Beyerlein, T. C. Germann, and M. Koslowski, “Influence of stacking fault energy surface on partial dislocations in fcc metals with a three-dimensional phase field dislocation dynamics model”, Physical Review B, 84 (2011) 144108.
3.	Y. Zeng, A. Hunter, I. J. Beyerlein, and M. Koslowski “A phase field dislocation dynamics model for a bicrystal interface system: An investigation into dislocation slip transmission across cube-on-cube interfaces”, International Journal of Plasticity, 79 (2016) 293-313.

Related review articles:

1.	A. Hunter, B. Leu, and I. J. Beyerlein, “A review of slip transfer: applications of mesoscale techniques”, Journal of Materials Science, 53 (2018) 5584-5603.
2.	I. J. Beyerlein and A. Hunter, “Understanding dislocation mechanics at the meso-scale using phase field dislocation dynamics”, Philosophical Transactions A, 374 (2016).

PFDD is distributed as open source software available under the BSD-3 license.
 
