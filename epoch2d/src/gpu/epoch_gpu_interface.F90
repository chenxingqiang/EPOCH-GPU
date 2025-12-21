! Copyright (C) 2025 EPOCH-GPU Development Team
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! @file epoch_gpu_interface.F90
! @brief High-level interface for GPU-accelerated EPOCH operations
! @details This module provides the integration layer between EPOCH's
!          main simulation loop and the GPU-accelerated kernels.

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
  PUBLIC :: epoch_gpu_initialize
  PUBLIC :: epoch_gpu_finalize
  PUBLIC :: epoch_gpu_push_particles
  PUBLIC :: epoch_gpu_update_fields
  PUBLIC :: epoch_gpu_sync_fields_to_device
  PUBLIC :: epoch_gpu_sync_fields_to_host
  PUBLIC :: epoch_gpu_is_enabled
  PUBLIC :: gpu_push_all_species
  PUBLIC :: copy_fields_to_gpu
  PUBLIC :: copy_fields_from_gpu

  ! Module state
  LOGICAL, SAVE :: gpu_initialized = .FALSE.
  LOGICAL, SAVE :: use_gpu = .FALSE.

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief Initialize GPU acceleration for EPOCH
  !> @param[in] use_gpu_flag Enable/disable GPU acceleration
  !> @param[in] device_id GPU device ID to use (default: 0)
  !> @return Error code (0 = success)
  !---------------------------------------------------------------------------
  FUNCTION epoch_gpu_initialize(use_gpu_flag, device_id) RESULT(ierr)
    LOGICAL, INTENT(IN) :: use_gpu_flag
    INTEGER, INTENT(IN), OPTIONAL :: device_id
    INTEGER :: ierr

#ifdef GPU_ENABLED
    INTEGER :: dev_id

    ierr = 0
    use_gpu = use_gpu_flag

    IF (.NOT. use_gpu) THEN
      WRITE(*,*) 'EPOCH-GPU: GPU acceleration disabled by user'
      RETURN
    END IF

    ! Initialize GPU subsystem
    IF (PRESENT(device_id)) THEN
      dev_id = device_id
    ELSE
      dev_id = 0
    END IF

    ierr = gpu_init(dev_id)
    IF (ierr /= 0) THEN
      WRITE(*,*) 'EPOCH-GPU: Failed to initialize GPU, falling back to CPU'
      use_gpu = .FALSE.
      RETURN
    END IF

    ! Print device info
    CALL gpu_print_device_info()

    gpu_initialized = .TRUE.
    WRITE(*,*) 'EPOCH-GPU: GPU acceleration initialized successfully'
#else
    ! GPU not enabled at compile time
    ierr = 0
    use_gpu = .FALSE.
    WRITE(*,*) 'EPOCH-GPU: Compiled without GPU support'
#endif

  END FUNCTION epoch_gpu_initialize


  !---------------------------------------------------------------------------
  !> @brief Finalize GPU acceleration
  !---------------------------------------------------------------------------
  SUBROUTINE epoch_gpu_finalize()

#ifdef GPU_ENABLED
    IF (gpu_initialized) THEN
      CALL gpu_finalize()
      gpu_initialized = .FALSE.
      use_gpu = .FALSE.
    END IF
#endif

  END SUBROUTINE epoch_gpu_finalize


  !---------------------------------------------------------------------------
  !> @brief Check if GPU acceleration is enabled
  !---------------------------------------------------------------------------
  FUNCTION epoch_gpu_is_enabled() RESULT(enabled)
    LOGICAL :: enabled

    enabled = use_gpu .AND. gpu_initialized

  END FUNCTION epoch_gpu_is_enabled


  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated particle push
  !> @details Pushes all particles using GPU if available, otherwise CPU
  !---------------------------------------------------------------------------
  SUBROUTINE epoch_gpu_push_particles(n_particles, pos_x, pos_y, px, py, pz, &
                                       ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                                       charge, mass, dt)
    INTEGER, PARAMETER :: num = 8
    INTEGER, PARAMETER :: i8 = SELECTED_INT_KIND(15)
    
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(INOUT) :: pos_x(:), pos_y(:)
    REAL(num), INTENT(INOUT) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), INTENT(IN) :: bx_p(:), by_p(:), bz_p(:)
    REAL(num), INTENT(IN) :: charge, mass, dt

#ifdef GPU_ENABLED
    IF (epoch_gpu_is_enabled()) THEN
      CALL push_particles_gpu(n_particles, pos_x, pos_y, px, py, pz, &
                              ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                              charge, mass, dt)
    ELSE
      ! Fallback to CPU - would call original push_particles here
      WRITE(*,*) 'Warning: GPU not available, CPU fallback not implemented'
    END IF
#else
    WRITE(*,*) 'Warning: GPU not compiled, CPU fallback not implemented'
#endif

  END SUBROUTINE epoch_gpu_push_particles


  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated field update
  !---------------------------------------------------------------------------
  SUBROUTINE epoch_gpu_update_fields(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                      nx, ny, ng, cnx, cny, fac, hdtx, hdty)
    INTEGER, PARAMETER :: num = 8
    
    INTEGER, INTENT(IN) :: nx, ny, ng
    REAL(num), INTENT(INOUT) :: ex(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ey(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ez(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: by(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: jx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: jy(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: jz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: cnx, cny, fac, hdtx, hdty

#ifdef GPU_ENABLED
    IF (epoch_gpu_is_enabled()) THEN
      ! Update E-field
      CALL update_e_field_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                              nx, ny, ng, cnx, cny, fac)
      ! Update B-field
      CALL update_b_field_gpu(ex, ey, ez, bx, by, bz, &
                              nx, ny, ng, hdtx, hdty)
    ELSE
      WRITE(*,*) 'Warning: GPU not available, CPU fallback not implemented'
    END IF
#else
    WRITE(*,*) 'Warning: GPU not compiled, CPU fallback not implemented'
#endif

  END SUBROUTINE epoch_gpu_update_fields


  !---------------------------------------------------------------------------
  !> @brief Synchronize field arrays from host to GPU device
  !---------------------------------------------------------------------------
  SUBROUTINE epoch_gpu_sync_fields_to_device(ex, ey, ez, bx, by, bz, &
                                              jx, jy, jz)
    INTEGER, PARAMETER :: num = 8
    
    REAL(num), INTENT(IN) :: ex(:,:), ey(:,:), ez(:,:)
    REAL(num), INTENT(IN) :: bx(:,:), by(:,:), bz(:,:)
    REAL(num), INTENT(IN) :: jx(:,:), jy(:,:), jz(:,:)

#ifdef GPU_ENABLED
    IF (epoch_gpu_is_enabled()) THEN
      !$ACC UPDATE DEVICE(ex, ey, ez, bx, by, bz, jx, jy, jz)
    END IF
#endif

  END SUBROUTINE epoch_gpu_sync_fields_to_device


  !---------------------------------------------------------------------------
  !> @brief Synchronize field arrays from GPU device to host
  !---------------------------------------------------------------------------
  SUBROUTINE epoch_gpu_sync_fields_to_host(ex, ey, ez, bx, by, bz)
    INTEGER, PARAMETER :: num = 8
    
    REAL(num), INTENT(INOUT) :: ex(:,:), ey(:,:), ez(:,:)
    REAL(num), INTENT(INOUT) :: bx(:,:), by(:,:), bz(:,:)

#ifdef GPU_ENABLED
    IF (epoch_gpu_is_enabled()) THEN
      !$ACC UPDATE HOST(ex, ey, ez, bx, by, bz)
    END IF
#endif

  END SUBROUTINE epoch_gpu_sync_fields_to_host


  !---------------------------------------------------------------------------
  !> @brief Copy field arrays to GPU
  !---------------------------------------------------------------------------
  SUBROUTINE copy_fields_to_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                imin, imax, jmin, jmax)
    INTEGER, INTENT(IN) :: imin, imax, jmin, jmax
    REAL(num), INTENT(IN) :: ex(imin:imax, jmin:jmax)
    REAL(num), INTENT(IN) :: ey(imin:imax, jmin:jmax)
    REAL(num), INTENT(IN) :: ez(imin:imax, jmin:jmax)
    REAL(num), INTENT(IN) :: bx(imin:imax, jmin:jmax)
    REAL(num), INTENT(IN) :: by(imin:imax, jmin:jmax)
    REAL(num), INTENT(IN) :: bz(imin:imax, jmin:jmax)
    REAL(num), INTENT(IN) :: jx(imin:imax, jmin:jmax)
    REAL(num), INTENT(IN) :: jy(imin:imax, jmin:jmax)
    REAL(num), INTENT(IN) :: jz(imin:imax, jmin:jmax)

#ifdef GPU_ENABLED
    !$ACC ENTER DATA COPYIN(ex, ey, ez, bx, by, bz, jx, jy, jz)
#endif

  END SUBROUTINE copy_fields_to_gpu


  !---------------------------------------------------------------------------
  !> @brief Copy field arrays from GPU
  !---------------------------------------------------------------------------
  SUBROUTINE copy_fields_from_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                  imin, imax, jmin, jmax)
    INTEGER, INTENT(IN) :: imin, imax, jmin, jmax
    REAL(num), INTENT(INOUT) :: ex(imin:imax, jmin:jmax)
    REAL(num), INTENT(INOUT) :: ey(imin:imax, jmin:jmax)
    REAL(num), INTENT(INOUT) :: ez(imin:imax, jmin:jmax)
    REAL(num), INTENT(INOUT) :: bx(imin:imax, jmin:jmax)
    REAL(num), INTENT(INOUT) :: by(imin:imax, jmin:jmax)
    REAL(num), INTENT(INOUT) :: bz(imin:imax, jmin:jmax)
    REAL(num), INTENT(INOUT) :: jx(imin:imax, jmin:jmax)
    REAL(num), INTENT(INOUT) :: jy(imin:imax, jmin:jmax)
    REAL(num), INTENT(INOUT) :: jz(imin:imax, jmin:jmax)

#ifdef GPU_ENABLED
    !$ACC UPDATE HOST(ex, ey, ez, bx, by, bz, jx, jy, jz)
    !$ACC EXIT DATA DELETE(ex, ey, ez, bx, by, bz, jx, jy, jz)
#endif

  END SUBROUTINE copy_fields_from_gpu


  !---------------------------------------------------------------------------
  !> @brief Push all particle species on GPU
  !>
  !> This handles field interpolation, Boris push, and current deposition
  !> for all species in a GPU-optimized manner.
  !>
  !> NOTE: This is a simplified version. For full functionality, the caller
  !> must provide x_min, y_min, ng (ghost cells) from shared_data.
  !---------------------------------------------------------------------------
  SUBROUTINE gpu_push_all_species(species_soa, n_species, species_list, &
                                  ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                  nx, ny, dt, dx, dy)
#ifdef GPU_ENABLED
    USE shared_data, ONLY: particle_species, x_min, y_min, ng
    USE constants, ONLY: q0, m0
#endif
    
    INTEGER, INTENT(IN) :: n_species, nx, ny
    REAL(num), INTENT(IN) :: dt, dx, dy
#ifdef GPU_ENABLED
    TYPE(particle_soa_t), INTENT(INOUT) :: species_soa(:)
    TYPE(particle_species), INTENT(IN) :: species_list(:)
    REAL(num), INTENT(INOUT) :: ex(:,:), ey(:,:), ez(:,:)
    REAL(num), INTENT(INOUT) :: bx(:,:), by(:,:), bz(:,:)
    REAL(num), INTENT(INOUT) :: jx(:,:), jy(:,:), jz(:,:)
    
    INTEGER :: ispecies
    INTEGER(i8) :: np
    REAL(num), ALLOCATABLE :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), ALLOCATABLE :: bx_p(:), by_p(:), bz_p(:)
    REAL(num), ALLOCATABLE :: pos_x_old(:), pos_y_old(:)
    REAL(num) :: charge, mass
    
    ! Transfer field and current arrays to GPU
    !$ACC ENTER DATA COPYIN(ex, ey, ez, bx, by, bz)
    !$ACC ENTER DATA COPYIN(jx, jy, jz)
    
    DO ispecies = 1, n_species
      np = species_soa(ispecies)%count
      IF (np <= 0) CYCLE
      
      ! Get particle charge and mass (already in SI units in EPOCH)
      charge = species_list(ispecies)%charge
      mass = species_list(ispecies)%mass
      
      ! #region agent log - Debug particle properties
      IF (ispecies == 1) THEN
        WRITE(*,'(A,I5,A,I10)') 'DEBUG_2D: species ', ispecies, ' np = ', INT(np)
        WRITE(*,'(A,E15.7)') 'DEBUG_2D: charge = ', charge
        WRITE(*,'(A,E15.7)') 'DEBUG_2D: mass = ', mass
      END IF
      ! #endregion
      
      ! Allocate temporary arrays (like 1D)
      ALLOCATE(ex_p(np), ey_p(np), ez_p(np))
      ALLOCATE(bx_p(np), by_p(np), bz_p(np))
      ALLOCATE(pos_x_old(np), pos_y_old(np))
      
      ! Get current positions and save as old
      !$ACC UPDATE HOST(species_soa(ispecies)%pos_x(1:np))
      !$ACC UPDATE HOST(species_soa(ispecies)%pos_y(1:np))
      pos_x_old(1:np) = species_soa(ispecies)%pos_x(1:np)
      pos_y_old(1:np) = species_soa(ispecies)%pos_y(1:np)
      
      !$ACC ENTER DATA COPYIN(pos_x_old(1:np), pos_y_old(1:np))
      !$ACC ENTER DATA CREATE(ex_p, ey_p, ez_p, bx_p, by_p, bz_p)
      
      ! Interpolate fields to particle positions
      CALL interpolate_fields_to_particles_gpu(np, &
                                               species_soa(ispecies)%pos_x, &
                                               species_soa(ispecies)%pos_y, &
                                               ex, ey, ez, bx, by, bz, &
                                               ex_p, ey_p, ez_p, &
                                               bx_p, by_p, bz_p, &
                                               x_min, y_min, dx, dy, nx, ny, ng)
      
      ! Push particles using Boris algorithm (updates positions to new)
      CALL push_particles_gpu(np, &
                              species_soa(ispecies)%pos_x, &
                              species_soa(ispecies)%pos_y, &
                              species_soa(ispecies)%px, &
                              species_soa(ispecies)%py, &
                              species_soa(ispecies)%pz, &
                              ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                              charge, mass, dt)
      
      ! Deposit current using Esirkepov
      CALL deposit_current_gpu(np, &
                               pos_x_old, pos_y_old, &
                               species_soa(ispecies)%pos_x, &
                               species_soa(ispecies)%pos_y, &
                               species_soa(ispecies)%px, &
                               species_soa(ispecies)%py, &
                               species_soa(ispecies)%pz, &
                               species_soa(ispecies)%weight, &
                               charge, mass, &
                               jx, jy, jz, &
                               x_min, y_min, dx, dy, dt, nx, ny, ng)
      
      ! Clean up temporary arrays on GPU
      !$ACC EXIT DATA DELETE(ex_p, ey_p, ez_p, bx_p, by_p, bz_p)
      !$ACC EXIT DATA DELETE(pos_x_old, pos_y_old)
      
      ! Deallocate CPU arrays
      DEALLOCATE(ex_p, ey_p, ez_p)
      DEALLOCATE(bx_p, by_p, bz_p)
      DEALLOCATE(pos_x_old, pos_y_old)
    END DO
    
    ! Sync current back to CPU for field update
    !$ACC UPDATE HOST(jx, jy, jz)
    
    ! #region agent log - Debug 2D current values
    WRITE(*,'(A)') '=== 2D DEBUG OUTPUT ==='
    WRITE(*,'(A,E15.7)') 'DEBUG_2D: jx_max = ', MAXVAL(ABS(jx))
    WRITE(*,'(A,E15.7)') 'DEBUG_2D: jx_sum = ', SUM(jx)
    WRITE(*,'(A,I10,A,I10)') 'DEBUG_2D: nx = ', nx, ', ny = ', ny
    WRITE(*,'(A,I10)') 'DEBUG_2D: ng = ', ng
    WRITE(*,'(A,E15.7,A,E15.7)') 'DEBUG_2D: dx = ', dx, ', dy = ', dy
    WRITE(*,'(A,E15.7)') 'DEBUG_2D: dt = ', dt
    WRITE(*,'(A,E15.7,A,E15.7)') 'DEBUG_2D: x_min = ', x_min, ', y_min = ', y_min
    ! #endregion
    
    ! Clean up field arrays from GPU
    !$ACC EXIT DATA DELETE(ex, ey, ez, bx, by, bz)
    !$ACC EXIT DATA DELETE(jx, jy, jz)
#else
    TYPE(*), INTENT(INOUT) :: species_soa(:)
    TYPE(*), INTENT(IN) :: species_list(:)
    REAL(num), INTENT(INOUT) :: ex(:,:), ey(:,:), ez(:,:)
    REAL(num), INTENT(INOUT) :: bx(:,:), by(:,:), bz(:,:)
    REAL(num), INTENT(INOUT) :: jx(:,:), jy(:,:), jz(:,:)
    ! Suppress unused warnings
    IF (.FALSE.) THEN
      CONTINUE
    END IF
#endif
    
  END SUBROUTINE gpu_push_all_species

END MODULE epoch_gpu_interface

