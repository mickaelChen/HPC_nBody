# HPC_nBody

This repository contains my master's project about distributed computing for the n-bodies problem.

The goal of the project was to parallelize and distribute a given sequential programme. 
In this case, we adapted an implementation of a n-body simulation that used a leap-frog scheme.

We proposed a multi-process implementation using MPI, an hybrid version MPI+OMP with additional exploited multi-threading, and also experimented with SIMD vectorisation.

Check the report (in french) for more information and results.
