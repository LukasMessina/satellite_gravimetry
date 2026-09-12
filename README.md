# Project Overview

## Summary

The TU Delft Astrodynamics Toolbox (Tudat) is an astrodynamics simulation tool in active development at the Space Engineering Department, Aerospace Faculty of TU Delft.

Tudat is a powerful framework for “sandbox”-type astrodynamics simulations. It is also capable of processing range data, such as Doppler measurements or laser ranging, from Earth to spacecraft orbiting planetary bodies. This capability allows gravity fields to be estimated from these types of observations. However, Tudat currently lacks the ability to process much more accurate low-low Satellite-to-Satellite Tracking (ll-SST) observations.

Low-low SST data are considered the best candidates for dedicated gravimetric missions beyond Earth, as already demonstrated by the success of the GRAIL mission on the Moon and by the MaQuIs mission concept proposed for Mars.

Determining the static gravity field of Mars would allow us to answer important questions related to the origin of the planet and the global crustal dichotomy. In addition, an ll-SST mission around Mars would enable the retrieval of the time-variable gravity field. This would make it possible to monitor the seasonal CO₂ cycle and study the effect of orbital motion on Martian climate.

On Earth, the availability of GNSS enables accurate satellite orbit positions to be used as observations, also referred to as low-low Satellite-to-Satellite Tracking, to recover the low-degree components of the gravity field.

---

## Objectives

The main objective of this research is to implement the processing of:

* Low-low Satellite-to-Satellite Tracking (ll-SST) data
* High-low Satellite-to-Satellite Tracking (hl-SST) data

for gravity field estimation within Tudat.

Although verification of the developed software will be performed using simulated data, the primary objective is to use real observations from:

* GRACE
* GRACE-FO
* Swarm

These datasets will be used to validate the resulting gravity field models.

---


