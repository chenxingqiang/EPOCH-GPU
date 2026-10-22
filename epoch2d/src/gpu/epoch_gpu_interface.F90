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
  PUBLIC :: gpu_yee_resident
  PUBLIC :: gpu_update_eb_fields_half
  PUBLIC :: gpu_update_eb_fields_final
  PUBLIC :: gpu_sync_current_to_device
  PUBLIC :: gpu_publish_snapshot
  PUBLIC :: gpu_reload_species_from_host
  PUBLIC :: gpu_reload_fields_from_host
  PUBLIC :: gpu_bind_species
  PUBLIC :: gpu_step_touches_host_particles

  ! Module state
  LOGICAL, SAVE :: gpu_initialized = .FALSE.
  LOGICAL, SAVE :: use_gpu = .FALSE.
#ifdef GPU_ENABLED
  LOGICAL, SAVE :: fields_resident = .FALSE.
  INTEGER(i8), SAVE :: gather_capacity = 0
  REAL(num), ALLOCATABLE, SAVE :: g_ex(:), g_ey(:), g_ez(:)
  REAL(num), ALLOCATABLE, SAVE :: g_bx(:), g_by(:), g_bz(:)
  TYPE(particle_soa_t), POINTER, SAVE :: resident_soa(:) => NULL()
#endif

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
    fields_resident = .TRUE.
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
    fields_resident = .FALSE.
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
    
    INTEGER :: ispecies, ix, iy, i1, i2, j1, j2
    INTEGER(i8) :: np
    REAL(num) :: charge, mass

    IF (.NOT. fields_resident) THEN
      !$ACC ENTER DATA COPYIN(ex, ey, ez, bx, by, bz, jx, jy, jz)
    ELSE IF (.NOT. gpu_yee_resident()) THEN
      !$ACC UPDATE DEVICE(ex, ey, ez, bx, by, bz)
    END IF

    i1 = LBOUND(jx, 1)
    i2 = UBOUND(jx, 1)
    j1 = LBOUND(jx, 2)
    j2 = UBOUND(jx, 2)
    !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(jx, jy, jz)
    DO iy = j1, j2
      DO ix = i1, i2
        jx(ix, iy) = 0.0_num
        jy(ix, iy) = 0.0_num
        jz(ix, iy) = 0.0_num
      END DO
    END DO
    !$ACC END PARALLEL LOOP

    DO ispecies = 1, n_species
      np = species_soa(ispecies)%count
      IF (np <= 0) CYCLE

      charge = species_list(ispecies)%charge
      mass = species_list(ispecies)%mass
      CALL save_old_positions_2d(np, species_soa(ispecies)%pos_x, &
          species_soa(ispecies)%pos_y, species_soa(ispecies)%pos_x_old, &
          species_soa(ispecies)%pos_y_old)
      CALL ensure_gather_buffers(np)

      CALL interpolate_fields_to_particles_gpu(np, &
                                               species_soa(ispecies)%pos_x, &
                                               species_soa(ispecies)%pos_y, &
                                               ex, ey, ez, bx, by, bz, &
                                               g_ex, g_ey, g_ez, &
                                               g_bx, g_by, g_bz, &
                                               x_min, y_min, dx, dy, nx, ny, ng)

      CALL push_particles_gpu(np, &
                              species_soa(ispecies)%pos_x, &
                              species_soa(ispecies)%pos_y, &
                              species_soa(ispecies)%px, &
                              species_soa(ispecies)%py, &
                              species_soa(ispecies)%pz, &
                              g_ex, g_ey, g_ez, g_bx, g_by, g_bz, &
                              charge, mass, dt)

      CALL deposit_current_gpu(np, &
                               species_soa(ispecies)%pos_x_old, &
                               species_soa(ispecies)%pos_y_old, &
                               species_soa(ispecies)%pos_x, &
                               species_soa(ispecies)%pos_y, &
                               species_soa(ispecies)%px, &
                               species_soa(ispecies)%py, &
                               species_soa(ispecies)%pz, &
                               species_soa(ispecies)%weight, &
                               charge, mass, &
                               jx, jy, jz, &
                               x_min, y_min, dx, dy, dt, nx, ny, ng)
    END DO

    !$ACC UPDATE HOST(jx, jy, jz)
    IF (.NOT. fields_resident) THEN
      !$ACC EXIT DATA DELETE(ex, ey, ez, bx, by, bz, jx, jy, jz)
    END IF
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


  SUBROUTINE save_old_positions_2d(np, pos_x, pos_y, pos_x_old, pos_y_old)
    INTEGER(i8), INTENT(IN) :: np
    REAL(num), INTENT(IN) :: pos_x(:), pos_y(:)
    REAL(num), INTENT(OUT) :: pos_x_old(:), pos_y_old(:)
    INTEGER(i8) :: ip
#ifdef GPU_ENABLED
    !$ACC PARALLEL LOOP PRESENT(pos_x(1:np), pos_y(1:np), &
    !$ACC& pos_x_old(1:np), pos_y_old(1:np))
    DO ip = 1, np
      pos_x_old(ip) = pos_x(ip)
      pos_y_old(ip) = pos_y(ip)
    END DO
    !$ACC END PARALLEL LOOP
#else
    IF (np > 0) THEN
      pos_x_old(1) = pos_x(1)
      pos_y_old(1) = pos_y(1)
    END IF
    ip = 0
#endif
  END SUBROUTINE save_old_positions_2d


  SUBROUTINE ensure_gather_buffers(np)
    INTEGER(i8), INTENT(IN) :: np
#ifdef GPU_ENABLED
    IF (gather_capacity >= np) RETURN
    IF (gather_capacity > 0) THEN
      !$ACC EXIT DATA DELETE(g_ex, g_ey, g_ez, g_bx, g_by, g_bz)
      DEALLOCATE(g_ex, g_ey, g_ez, g_bx, g_by, g_bz)
    END IF
    ALLOCATE(g_ex(np), g_ey(np), g_ez(np), g_bx(np), g_by(np), g_bz(np))
    !$ACC ENTER DATA CREATE(g_ex, g_ey, g_ez, g_bx, g_by, g_bz)
    gather_capacity = np
#else
    IF (np < 0) RETURN
#endif
  END SUBROUTINE ensure_gather_buffers


  FUNCTION gpu_yee_resident() RESULT(active)
#ifdef GPU_ENABLED
    USE shared_data, ONLY: cpml_boundaries, maxwell_solver
    USE constants, ONLY: c_maxwell_solver_yee
    USE fields, ONLY: field_order
#endif
    LOGICAL :: active
    active = .FALSE.
#ifdef GPU_ENABLED
    active = fields_resident .AND. field_order == 2 &
        .AND. .NOT. cpml_boundaries &
        .AND. maxwell_solver == c_maxwell_solver_yee
#endif
  END FUNCTION gpu_yee_resident


  SUBROUTINE gpu_update_eb_fields_half()
#ifdef GPU_ENABLED
    USE shared_data, ONLY: ex, ey, ez, bx, by, bz, jx, jy, jz, &
        nx, ny, ng, dt, dx, dy
    USE constants, ONLY: c, epsilon0
    USE boundary, ONLY: efield_bcs, bfield_bcs
    REAL(num) :: hdt, hdtx, hdty, cnx, cny, fac

    hdt = 0.5_num * dt
    hdtx = hdt / dx
    hdty = hdt / dy
    cnx = hdtx * c**2
    cny = hdty * c**2
    fac = hdt / epsilon0

    CALL update_e_field_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
        nx, ny, ng, cnx, cny, fac)
    !$ACC UPDATE HOST(ex, ey, ez)
    CALL efield_bcs
    !$ACC UPDATE DEVICE(ex, ey, ez)

    CALL update_b_field_gpu(ex, ey, ez, bx, by, bz, nx, ny, ng, hdtx, hdty)
    !$ACC UPDATE HOST(bx, by, bz)
    CALL bfield_bcs(.TRUE.)
    !$ACC UPDATE DEVICE(bx, by, bz)
#endif
  END SUBROUTINE gpu_update_eb_fields_half


  SUBROUTINE gpu_update_eb_fields_final()
#ifdef GPU_ENABLED
    USE shared_data, ONLY: ex, ey, ez, bx, by, bz, jx, jy, jz, &
        nx, ny, ng, dt, dx, dy
    USE constants, ONLY: c, epsilon0
    USE boundary, ONLY: efield_bcs, bfield_final_bcs
    REAL(num) :: hdt, hdtx, hdty, cnx, cny, fac

    hdt = 0.5_num * dt
    hdtx = hdt / dx
    hdty = hdt / dy
    cnx = hdtx * c**2
    cny = hdty * c**2
    fac = hdt / epsilon0

    CALL update_b_field_gpu(ex, ey, ez, bx, by, bz, nx, ny, ng, hdtx, hdty)
    !$ACC UPDATE HOST(bx, by, bz)
    CALL bfield_final_bcs
    !$ACC UPDATE DEVICE(bx, by, bz)

    CALL update_e_field_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
        nx, ny, ng, cnx, cny, fac)
    !$ACC UPDATE HOST(ex, ey, ez)
    CALL efield_bcs
    !$ACC UPDATE DEVICE(ex, ey, ez)
#endif
  END SUBROUTINE gpu_update_eb_fields_final


  SUBROUTINE gpu_sync_current_to_device()
#ifdef GPU_ENABLED
    USE shared_data, ONLY: jx, jy, jz
    IF (.NOT. fields_resident) RETURN
    !$ACC UPDATE DEVICE(jx, jy, jz)
#endif
  END SUBROUTINE gpu_sync_current_to_device


  SUBROUTINE gpu_reload_fields_from_host()
#ifdef GPU_ENABLED
    USE shared_data, ONLY: ex, ey, ez, bx, by, bz, jx, jy, jz
    IF (.NOT. fields_resident) RETURN
    !$ACC UPDATE DEVICE(ex, ey, ez, bx, by, bz, jx, jy, jz)
#endif
  END SUBROUTINE gpu_reload_fields_from_host


  SUBROUTINE gpu_bind_species(soa)
#ifdef GPU_ENABLED
    TYPE(particle_soa_t), TARGET, INTENT(IN) :: soa(:)
    resident_soa => soa
#else
    TYPE(*), INTENT(IN) :: soa(:)
    IF (.FALSE.) THEN
      IF (SIZE(soa) < 0) RETURN
    END IF
#endif
  END SUBROUTINE gpu_bind_species


  SUBROUTINE gpu_publish_snapshot()
#ifdef GPU_ENABLED
    USE shared_data, ONLY: species_list, ex, ey, ez, bx, by, bz, jx, jy, jz
    INTEGER :: ispecies
    IF (.NOT. ASSOCIATED(resident_soa)) RETURN
    IF (fields_resident) THEN
      !$ACC UPDATE HOST(ex, ey, ez, bx, by, bz, jx, jy, jz)
    END IF
    DO ispecies = 1, SIZE(resident_soa)
      CALL publish_species_from_gpu(resident_soa(ispecies), &
          species_list(ispecies))
    END DO
#endif
  END SUBROUTINE gpu_publish_snapshot


  SUBROUTINE gpu_reload_species_from_host()
#ifdef GPU_ENABLED
    USE shared_data, ONLY: species_list
    INTEGER :: ispecies
    IF (.NOT. ASSOCIATED(resident_soa)) RETURN
    DO ispecies = 1, SIZE(resident_soa)
      CALL reload_species_to_gpu(species_list(ispecies), resident_soa(ispecies))
    END DO
#endif
  END SUBROUTINE gpu_reload_species_from_host


  FUNCTION gpu_step_touches_host_particles() RESULT(yes)
#ifdef GPU_ENABLED
    USE shared_data, ONLY: use_particle_lists, use_collisions, &
        coll_n_step, use_collisional_ionisation, ci_n_step, step, &
        use_background_collisions, use_particle_migration, use_field_ionisation
    USE split_particle, ONLY: use_split
#endif
    LOGICAL :: yes
#ifdef GPU_ENABLED
    LOGICAL :: collision_step, coll_ion_step
    collision_step = use_particle_lists .AND. use_collisions &
        .AND. MODULO(step, coll_n_step) == coll_n_step - 1
    coll_ion_step = use_particle_lists .AND. use_collisional_ionisation &
        .AND. MODULO(step, ci_n_step) == ci_n_step - 1
    yes = collision_step .OR. coll_ion_step &
        .OR. (use_particle_lists .AND. use_split) &
        .OR. use_background_collisions .OR. use_particle_migration &
        .OR. use_field_ionisation
#else
    yes = .FALSE.
#endif
  END FUNCTION gpu_step_touches_host_particles

END MODULE epoch_gpu_interface

