# Physics Informed Neural Networks (PINN) for satellite state estimation
Educational repo for learning about physics informed neural networks (PINNs) its application  to 
estimate the orbit of a satellite flying in geostationary orbit.

## Problem overview
The following repo is an attempt to implement this paper by [Varey et al.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10521414&casa_token=qRdAnbemqw0AAAAA:k8Se-ca4IhB_WpUVeIuZUWhEdUxOT2i97c8MHb1u9JqmxdZ-Wufy7RdmvR01AEhKuY9c6g&tag=1). Fitting an orbit of satellite 
requires an accurate model of the forces acting on it. High-quality, physics-based models designed for this purpose have existed for decades. 
However, most of these models only account for two-body keplerian motion + Earth gravity perturbations (e.g., J2-J6) + atmospheric drag, etc.
These models are quite accurate for estimating and propagating orbits of non-maneuvering satellites, but fail when there are anomalous, unaccounted
accelerations, such as the ones that would be observed if the satellite were equipped with low-thrust electric propulsion.

