
all :
	$(MAKE) -C NBODY_direct-MPI
	$(MAKE) -C NBODY_direct-MPI-AVX
	$(MAKE) -C NBODY_direct-MPI-OMP
	$(MAKE) -C NBODY_direct-MPI-OMP-AVX

clean:
	$(MAKE) -C NBODY_direct-MPI clean
	$(MAKE) -C NBODY_direct-MPI-AVX clean
	$(MAKE) -C NBODY_direct-MPI-OMP clean
	$(MAKE) -C NBODY_direct-MPI-OMP-AVX clean

vclean:
	$(MAKE) -C NBODY_direct-MPI vclean
	$(MAKE) -C NBODY_direct-MPI-AVX vclean
	$(MAKE) -C NBODY_direct-MPI-OMP vclean
	$(MAKE) -C NBODY_direct-MPI-OMP-AVX vclean