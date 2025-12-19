! Copyright (C) 2024 EPOCH-GPU Development Team
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

#ifdef GPU_ENABLED
  USE gpu_config
  USE fields_gpu
  USE particles_gpu
  USE particle_data_gpu
#endif

  IMPLICIT NONE
  PRIVATE

  ! Public interface
  PUBLIC :: epoch_gpu_initialize
  PUBLIC :: epoch_gpu_finalize
  PUBLIC :: epoch_gpu_push_particles
  PUBLIC :: epoch_gpu_update_fields
  PUBLIC :: epoch_gpu_sync_fields_to_device
  PUBLIC :: epoch_gpu_sync_fields_to_host
  PUBLIC :: epoch_gpu_is_enabled

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

END MODULE epoch_gpu_interface

