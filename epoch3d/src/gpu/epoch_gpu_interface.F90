!> @file epoch_gpu_interface.F90
!> @brief High-level interface for GPU-accelerated EPOCH operations - 3D version
!> @details This module provides the integration layer between EPOCH's
!>          main simulation loop and the GPU-accelerated kernels.
!>
!> @author EPOCH-GPU Development Team
!> @date December 2025

MODULE epoch_gpu_interface

  USE, INTRINSIC :: iso_fortran_env, ONLY: REAL64, INT64
#ifdef GPU_ENABLED
  USE gpu_config
  USE fields_gpu
  USE particles_gpu
  USE particle_data_gpu
  USE particle_converter
#endif

  IMPLICIT NONE
  PRIVATE

  INTEGER, PARAMETER :: num = REAL64
  INTEGER, PARAMETER :: i8 = INT64

  ! Public interface
  PUBLIC :: gpu_push_all_species
  PUBLIC :: copy_fields_to_gpu
  PUBLIC :: copy_fields_from_gpu

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief Copy field arrays to GPU (3D version)
  !---------------------------------------------------------------------------
  SUBROUTINE copy_fields_to_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                imin, imax, jmin, jmax, kmin, kmax)
    INTEGER, INTENT(IN) :: imin, imax, jmin, jmax, kmin, kmax
    REAL(num), INTENT(IN) :: ex(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(IN) :: ey(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(IN) :: ez(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(IN) :: bx(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(IN) :: by(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(IN) :: bz(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(IN) :: jx(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(IN) :: jy(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(IN) :: jz(imin:imax, jmin:jmax, kmin:kmax)

#ifdef GPU_ENABLED
    !$ACC ENTER DATA COPYIN(ex, ey, ez, bx, by, bz, jx, jy, jz)
#endif

  END SUBROUTINE copy_fields_to_gpu


  !---------------------------------------------------------------------------
  !> @brief Copy field arrays from GPU (3D version)
  !---------------------------------------------------------------------------
  SUBROUTINE copy_fields_from_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                  imin, imax, jmin, jmax, kmin, kmax)
    INTEGER, INTENT(IN) :: imin, imax, jmin, jmax, kmin, kmax
    REAL(num), INTENT(INOUT) :: ex(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(INOUT) :: ey(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(INOUT) :: ez(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(INOUT) :: bx(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(INOUT) :: by(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(INOUT) :: bz(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(INOUT) :: jx(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(INOUT) :: jy(imin:imax, jmin:jmax, kmin:kmax)
    REAL(num), INTENT(INOUT) :: jz(imin:imax, jmin:jmax, kmin:kmax)

#ifdef GPU_ENABLED
    !$ACC UPDATE HOST(ex, ey, ez, bx, by, bz, jx, jy, jz)
    !$ACC EXIT DATA DELETE(ex, ey, ez, bx, by, bz, jx, jy, jz)
#endif

  END SUBROUTINE copy_fields_from_gpu


  !---------------------------------------------------------------------------
  !> @brief Push all particle species on GPU (3D version)
  !>
  !> This handles field interpolation, Boris push, and current deposition
  !> for all species in a GPU-optimized manner.
  !---------------------------------------------------------------------------
  SUBROUTINE gpu_push_all_species(species_soa, n_species, species_list, &
                                  ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                  nx, ny, nz, dt, dx, dy, dz)
#ifdef GPU_ENABLED
    USE shared_data, ONLY: particle_species, x_min, y_min, z_min, ng
#endif
    
    INTEGER, INTENT(IN) :: n_species, nx, ny, nz
    REAL(num), INTENT(IN) :: dt, dx, dy, dz
#ifdef GPU_ENABLED
    TYPE(particle_soa_3d), INTENT(INOUT) :: species_soa(:)
    TYPE(particle_species), INTENT(IN) :: species_list(:)
    REAL(num), INTENT(INOUT) :: ex(:,:,:), ey(:,:,:), ez(:,:,:)
    REAL(num), INTENT(INOUT) :: bx(:,:,:), by(:,:,:), bz(:,:,:)
    REAL(num), INTENT(INOUT) :: jx(:,:,:), jy(:,:,:), jz(:,:,:)
    
    INTEGER :: ispecies
    INTEGER(i8) :: np, ip
    REAL(num), ALLOCATABLE :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), ALLOCATABLE :: bx_p(:), by_p(:), bz_p(:)
    REAL(num), ALLOCATABLE :: pos_x_old(:), pos_y_old(:), pos_z_old(:)
    REAL(num) :: charge, mass, weight_val
    
    DO ispecies = 1, n_species
      np = species_soa(ispecies)%count
      IF (np <= 0) CYCLE
      
      charge = species_list(ispecies)%charge
      mass = species_list(ispecies)%mass
      weight_val = species_list(ispecies)%weight
      
      ! Allocate temporary arrays for interpolated fields
      ALLOCATE(ex_p(np), ey_p(np), ez_p(np))
      ALLOCATE(bx_p(np), by_p(np), bz_p(np))
      ALLOCATE(pos_x_old(np), pos_y_old(np), pos_z_old(np))
      
      !$ACC ENTER DATA CREATE(ex_p, ey_p, ez_p, bx_p, by_p, bz_p)
      !$ACC ENTER DATA CREATE(pos_x_old, pos_y_old, pos_z_old)
      
      ! Save old positions ON GPU before particle push
      !$ACC PARALLEL LOOP PRESENT(pos_x_old, pos_y_old, pos_z_old) &
      !$ACC   PRESENT(species_soa(ispecies)%pos_x, species_soa(ispecies)%pos_y, species_soa(ispecies)%pos_z)
      DO ip = 1, np
        pos_x_old(ip) = species_soa(ispecies)%pos_x(ip)
        pos_y_old(ip) = species_soa(ispecies)%pos_y(ip)
        pos_z_old(ip) = species_soa(ispecies)%pos_z(ip)
      END DO
      !$ACC END PARALLEL LOOP
      
      ! Interpolate fields to particle positions (3D: trilinear interpolation)
      CALL interpolate_fields_to_particles_gpu(np, &
                                               species_soa(ispecies)%pos_x, &
                                               species_soa(ispecies)%pos_y, &
                                               species_soa(ispecies)%pos_z, &
                                               ex, ey, ez, bx, by, bz, &
                                               ex_p, ey_p, ez_p, &
                                               bx_p, by_p, bz_p, &
                                               x_min, y_min, z_min, &
                                               dx, dy, dz, nx, ny, nz, ng)
      
      ! Push particles using Boris algorithm
      CALL push_particles_gpu(np, &
                              species_soa(ispecies)%pos_x, &
                              species_soa(ispecies)%pos_y, &
                              species_soa(ispecies)%pos_z, &
                              species_soa(ispecies)%px, &
                              species_soa(ispecies)%py, &
                              species_soa(ispecies)%pz, &
                              ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                              charge, mass, dt)
      
      ! Deposit current (using old and new positions)
      CALL deposit_current_gpu(np, &
                               species_soa(ispecies)%pos_x, &
                               species_soa(ispecies)%pos_y, &
                               species_soa(ispecies)%pos_z, &
                               pos_x_old, pos_y_old, pos_z_old, &
                               species_soa(ispecies)%px, &
                               species_soa(ispecies)%py, &
                               species_soa(ispecies)%pz, &
                               weight_val, charge, mass, &
                               jx, jy, jz, &
                               x_min, y_min, z_min, &
                               dx, dy, dz, dt, nx, ny, nz, ng)
      
      ! Clean up temporary arrays on GPU
      !$ACC EXIT DATA DELETE(ex_p, ey_p, ez_p, bx_p, by_p, bz_p)
      !$ACC EXIT DATA DELETE(pos_x_old, pos_y_old, pos_z_old)
      
      ! Deallocate CPU arrays
      DEALLOCATE(ex_p, ey_p, ez_p)
      DEALLOCATE(bx_p, by_p, bz_p)
      DEALLOCATE(pos_x_old, pos_y_old, pos_z_old)
    END DO
#else
    TYPE(*), INTENT(INOUT) :: species_soa(:)
    TYPE(*), INTENT(IN) :: species_list(:)
    REAL(num), INTENT(INOUT) :: ex(:,:,:), ey(:,:,:), ez(:,:,:)
    REAL(num), INTENT(INOUT) :: bx(:,:,:), by(:,:,:), bz(:,:,:)
    REAL(num), INTENT(INOUT) :: jx(:,:,:), jy(:,:,:), jz(:,:,:)
    ! Suppress unused warnings
    IF (.FALSE.) THEN
      CONTINUE
    END IF
#endif
    
  END SUBROUTINE gpu_push_all_species

END MODULE epoch_gpu_interface

