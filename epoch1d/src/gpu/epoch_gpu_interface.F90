!> @file epoch_gpu_interface.F90
!> @brief High-level interface for GPU-accelerated EPOCH operations - 1D version
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
  !> @brief Copy field arrays to GPU (1D version)
  !---------------------------------------------------------------------------
  SUBROUTINE copy_fields_to_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                imin, imax)
    INTEGER, INTENT(IN) :: imin, imax
    REAL(num), INTENT(IN) :: ex(imin:imax)
    REAL(num), INTENT(IN) :: ey(imin:imax)
    REAL(num), INTENT(IN) :: ez(imin:imax)
    REAL(num), INTENT(IN) :: bx(imin:imax)
    REAL(num), INTENT(IN) :: by(imin:imax)
    REAL(num), INTENT(IN) :: bz(imin:imax)
    REAL(num), INTENT(IN) :: jx(imin:imax)
    REAL(num), INTENT(IN) :: jy(imin:imax)
    REAL(num), INTENT(IN) :: jz(imin:imax)

#ifdef GPU_ENABLED
    !$ACC ENTER DATA COPYIN(ex, ey, ez, bx, by, bz, jx, jy, jz)
#endif

  END SUBROUTINE copy_fields_to_gpu


  !---------------------------------------------------------------------------
  !> @brief Copy field arrays from GPU (1D version)
  !---------------------------------------------------------------------------
  SUBROUTINE copy_fields_from_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                  imin, imax)
    INTEGER, INTENT(IN) :: imin, imax
    REAL(num), INTENT(INOUT) :: ex(imin:imax)
    REAL(num), INTENT(INOUT) :: ey(imin:imax)
    REAL(num), INTENT(INOUT) :: ez(imin:imax)
    REAL(num), INTENT(INOUT) :: bx(imin:imax)
    REAL(num), INTENT(INOUT) :: by(imin:imax)
    REAL(num), INTENT(INOUT) :: bz(imin:imax)
    REAL(num), INTENT(INOUT) :: jx(imin:imax)
    REAL(num), INTENT(INOUT) :: jy(imin:imax)
    REAL(num), INTENT(INOUT) :: jz(imin:imax)

#ifdef GPU_ENABLED
    !$ACC UPDATE HOST(ex, ey, ez, bx, by, bz, jx, jy, jz)
    !$ACC EXIT DATA DELETE(ex, ey, ez, bx, by, bz, jx, jy, jz)
#endif

  END SUBROUTINE copy_fields_from_gpu


  !---------------------------------------------------------------------------
  !> @brief Push all particle species on GPU (1D version)
  !>
  !> This handles field interpolation, Boris push, and current deposition
  !> for all species in a GPU-optimized manner.
  !---------------------------------------------------------------------------
  SUBROUTINE gpu_push_all_species(species_soa, n_species, species_list, &
                                  ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                  nx, dt, dx)
#ifdef GPU_ENABLED
    USE shared_data, ONLY: particle_species, x_min, ng
    USE constants, ONLY: q0, m0
#endif
    
    INTEGER, INTENT(IN) :: n_species, nx
    REAL(num), INTENT(IN) :: dt, dx
#ifdef GPU_ENABLED
    TYPE(particle_soa_1d), INTENT(INOUT) :: species_soa(:)
    TYPE(particle_species), INTENT(IN) :: species_list(:)
    REAL(num), INTENT(INOUT) :: ex(:), ey(:), ez(:)
    REAL(num), INTENT(INOUT) :: bx(:), by(:), bz(:)
    REAL(num), INTENT(INOUT) :: jx(:), jy(:), jz(:)
    
    INTEGER :: ispecies
    INTEGER(i8) :: np, ip
    REAL(num), ALLOCATABLE :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), ALLOCATABLE :: bx_p(:), by_p(:), bz_p(:)
    REAL(num), ALLOCATABLE :: pos_x_old(:)  ! For Esirkepov method
    REAL(num) :: charge, mass
    
    ! Transfer field arrays to GPU
    !$ACC ENTER DATA COPYIN(ex, ey, ez, bx, by, bz)
    
    ! Clear current arrays on CPU first (CRITICAL for correct physics!)
    ! The CPU clears these at the start of each time step
    jx = 0.0_num
    jy = 0.0_num
    jz = 0.0_num
    
    ! #region agent log - Debug after clearing (H0)
    WRITE(*,'(A,E12.4)') 'DEBUG_1D_H0: jx_max_after_clear_CPU = ', MAXVAL(ABS(jx))
    ! #endregion
    
    ! Transfer current arrays to GPU (now zeroed)
    !$ACC ENTER DATA COPYIN(jx, jy, jz)
    
    DO ispecies = 1, n_species
      np = species_soa(ispecies)%count
      IF (np <= 0) CYCLE
      
      ! Get particle charge and mass (already in SI units in EPOCH)
      charge = species_list(ispecies)%charge
      mass = species_list(ispecies)%mass
      
      
      ! Allocate temporary arrays
      ALLOCATE(ex_p(np), ey_p(np), ez_p(np))
      ALLOCATE(bx_p(np), by_p(np), bz_p(np))
      ALLOCATE(pos_x_old(np))
      
      ! Save old positions: GPU -> CPU -> pos_x_old -> GPU
      !$ACC UPDATE HOST(species_soa(ispecies)%pos_x(1:np))
      pos_x_old(1:np) = species_soa(ispecies)%pos_x(1:np)
      
      ! Transfer to GPU
      !$ACC ENTER DATA COPYIN(pos_x_old(1:np))
      !$ACC ENTER DATA CREATE(ex_p, ey_p, ez_p, bx_p, by_p, bz_p)
      
      ! Interpolate fields to particle positions (1D: linear interpolation)
      CALL interpolate_fields_to_particles_gpu(np, &
                                               species_soa(ispecies)%pos_x, &
                                               ex, ey, ez, bx, by, bz, &
                                               ex_p, ey_p, ez_p, &
                                               bx_p, by_p, bz_p, &
                                               nx, ng, x_min, dx)
      
      ! Push particles using Boris algorithm (updates positions to new)
      CALL push_particles_gpu(np, &
                              species_soa(ispecies)%pos_x, &
                              species_soa(ispecies)%px, &
                              species_soa(ispecies)%py, &
                              species_soa(ispecies)%pz, &
                              ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                              charge, mass, dt)
      
      ! #region agent log - Debug before current deposition (on GPU)
      !$ACC UPDATE HOST(jx)
      WRITE(*,'(A,E12.4)') 'DEBUG_1D_H1: jx_max_on_GPU_before_deposit = ', MAXVAL(ABS(jx))
      ! #endregion
      
      ! Deposit current using Esirkepov charge-conserving method
      ! Uses old positions (pos_x_old) and new positions (pos_x)
      CALL deposit_current_gpu(np, &
                               pos_x_old, &
                               species_soa(ispecies)%pos_x, &
                               species_soa(ispecies)%px, &
                               species_soa(ispecies)%py, &
                               species_soa(ispecies)%pz, &
                               species_soa(ispecies)%weight, &
                               jx, jy, jz, &
                               nx, ng, x_min, dx, dt, mass, charge)
      
      ! Clean up temporary arrays on GPU
      !$ACC EXIT DATA DELETE(ex_p, ey_p, ez_p, bx_p, by_p, bz_p)
      !$ACC EXIT DATA DELETE(pos_x_old)
      
      ! Deallocate CPU arrays
      DEALLOCATE(ex_p, ey_p, ez_p)
      DEALLOCATE(bx_p, by_p, bz_p)
      DEALLOCATE(pos_x_old)
    END DO
    
    ! Sync current back to CPU for field update
    !$ACC UPDATE HOST(jx, jy, jz)
    
    ! #region agent log - Debug after current deposition (H2)
    WRITE(*,'(A,E12.4)') 'DEBUG_1D_H2: jx_max_after = ', MAXVAL(ABS(jx))
    WRITE(*,'(A,E12.4)') 'DEBUG_1D_H2: jx_sum = ', SUM(jx)
    WRITE(*,'(A,E12.4,A,E12.4,A,E12.4)') 'DEBUG_1D_H2: jx(1)=', jx(1), ' jx(64)=', jx(64), ' jx(nx)=', jx(nx)
    ! #endregion
    
    ! Clean up field arrays from GPU
    !$ACC EXIT DATA DELETE(ex, ey, ez, bx, by, bz)
    !$ACC EXIT DATA DELETE(jx, jy, jz)
#else
    TYPE(*), INTENT(INOUT) :: species_soa(:)
    TYPE(*), INTENT(IN) :: species_list(:)
    REAL(num), INTENT(INOUT) :: ex(:), ey(:), ez(:)
    REAL(num), INTENT(INOUT) :: bx(:), by(:), bz(:)
    REAL(num), INTENT(INOUT) :: jx(:), jy(:), jz(:)
    ! Suppress unused warnings
    IF (.FALSE.) THEN
      CONTINUE
    END IF
#endif
    
  END SUBROUTINE gpu_push_all_species

END MODULE epoch_gpu_interface

