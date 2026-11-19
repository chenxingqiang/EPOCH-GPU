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
  PUBLIC :: gpu_yee_resident
  PUBLIC :: gpu_update_eb_fields_half
  PUBLIC :: gpu_update_eb_fields_final
  PUBLIC :: gpu_sync_current_to_device
  PUBLIC :: gpu_publish_snapshot
  PUBLIC :: gpu_reload_species_from_host
  PUBLIC :: gpu_reload_fields_from_host
  PUBLIC :: gpu_bind_species
  PUBLIC :: gpu_step_touches_host_particles

#ifdef GPU_ENABLED
  LOGICAL, SAVE :: fields_resident = .FALSE.
  INTEGER(i8), SAVE :: gather_capacity = 0
  REAL(num), ALLOCATABLE, SAVE :: g_ex(:), g_ey(:), g_ez(:)
  REAL(num), ALLOCATABLE, SAVE :: g_bx(:), g_by(:), g_bz(:)
  TYPE(particle_soa_1d), POINTER, SAVE :: resident_soa(:) => NULL()
#endif

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
    fields_resident = .TRUE.
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
    fields_resident = .FALSE.
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
#endif

    INTEGER, INTENT(IN) :: n_species, nx
    REAL(num), INTENT(IN) :: dt, dx
#ifdef GPU_ENABLED
    TYPE(particle_soa_1d), INTENT(INOUT) :: species_soa(:)
    TYPE(particle_species), INTENT(IN) :: species_list(:)
    REAL(num), INTENT(INOUT) :: ex(:), ey(:), ez(:)
    REAL(num), INTENT(INOUT) :: bx(:), by(:), bz(:)
    REAL(num), INTENT(INOUT) :: jx(:), jy(:), jz(:)

    INTEGER :: ispecies, ix, i1, i2
    INTEGER(i8) :: np
    REAL(num) :: charge, mass

    IF (.NOT. fields_resident) THEN
      !$ACC ENTER DATA COPYIN(ex, ey, ez, bx, by, bz, jx, jy, jz)
    ELSE IF (.NOT. gpu_yee_resident()) THEN
      !$ACC UPDATE DEVICE(ex, ey, ez, bx, by, bz)
    END IF

    i1 = LBOUND(jx, 1)
    i2 = UBOUND(jx, 1)
    !$ACC PARALLEL LOOP PRESENT(jx, jy, jz)
    DO ix = i1, i2
      jx(ix) = 0.0_num
      jy(ix) = 0.0_num
      jz(ix) = 0.0_num
    END DO
    !$ACC END PARALLEL LOOP

    DO ispecies = 1, n_species
      np = species_soa(ispecies)%count
      IF (np <= 0) CYCLE

      charge = species_list(ispecies)%charge
      mass = species_list(ispecies)%mass
      CALL save_old_positions_1d(np, species_soa(ispecies)%pos_x, &
          species_soa(ispecies)%pos_x_old)
      CALL ensure_gather_buffers(np)

      CALL interpolate_fields_to_particles_gpu(np, &
                                               species_soa(ispecies)%pos_x, &
                                               ex, ey, ez, bx, by, bz, &
                                               g_ex, g_ey, g_ez, &
                                               g_bx, g_by, g_bz, &
                                               nx, ng, x_min, dx)

      CALL push_particles_gpu(np, &
                              species_soa(ispecies)%pos_x, &
                              species_soa(ispecies)%px, &
                              species_soa(ispecies)%py, &
                              species_soa(ispecies)%pz, &
                              g_ex, g_ey, g_ez, g_bx, g_by, g_bz, &
                              charge, mass, dt)

      CALL deposit_current_gpu(np, &
                               species_soa(ispecies)%pos_x_old, &
                               species_soa(ispecies)%pos_x, &
                               species_soa(ispecies)%px, &
                               species_soa(ispecies)%py, &
                               species_soa(ispecies)%pz, &
                               species_soa(ispecies)%weight, &
                               jx, jy, jz, &
                               nx, ng, x_min, dx, dt, mass, charge)
    END DO

    !$ACC UPDATE HOST(jx, jy, jz)
    IF (.NOT. fields_resident) THEN
      !$ACC EXIT DATA DELETE(ex, ey, ez, bx, by, bz, jx, jy, jz)
    END IF
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


  SUBROUTINE save_old_positions_1d(np, pos_x, pos_x_old)
    INTEGER(i8), INTENT(IN) :: np
    REAL(num), INTENT(IN) :: pos_x(:)
    REAL(num), INTENT(OUT) :: pos_x_old(:)
    INTEGER(i8) :: ip
#ifdef GPU_ENABLED
    !$ACC PARALLEL LOOP PRESENT(pos_x(1:np), pos_x_old(1:np))
    DO ip = 1, np
      pos_x_old(ip) = pos_x(ip)
    END DO
    !$ACC END PARALLEL LOOP
#else
    IF (np > 0) THEN
      pos_x_old(1) = pos_x(1)
    END IF
    ip = 0
#endif
  END SUBROUTINE save_old_positions_1d


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
        nx, ng, dt, dx
    USE constants, ONLY: c, epsilon0
    USE boundary, ONLY: efield_bcs, bfield_bcs
    REAL(num) :: hdt, hdtx, cnx, fac

    hdt = 0.5_num * dt
    hdtx = hdt / dx
    cnx = hdtx * c**2
    fac = hdt / epsilon0

    CALL update_e_field_gpu_1d(ex, ey, ez, bx, by, bz, jx, jy, jz, &
        nx, ng, cnx, fac)
    !$ACC UPDATE HOST(ex, ey, ez)
    CALL efield_bcs
    !$ACC UPDATE DEVICE(ex, ey, ez)

    CALL update_b_field_gpu_1d(ex, ey, ez, bx, by, bz, nx, ng, hdtx)
    !$ACC UPDATE HOST(bx, by, bz)
    CALL bfield_bcs(.TRUE.)
    !$ACC UPDATE DEVICE(bx, by, bz)
#endif
  END SUBROUTINE gpu_update_eb_fields_half


  SUBROUTINE gpu_update_eb_fields_final()
#ifdef GPU_ENABLED
    USE shared_data, ONLY: ex, ey, ez, bx, by, bz, jx, jy, jz, &
        nx, ng, dt, dx
    USE constants, ONLY: c, epsilon0
    USE boundary, ONLY: efield_bcs, bfield_final_bcs
    REAL(num) :: hdt, hdtx, cnx, fac

    hdt = 0.5_num * dt
    hdtx = hdt / dx
    cnx = hdtx * c**2
    fac = hdt / epsilon0

    CALL update_b_field_gpu_1d(ex, ey, ez, bx, by, bz, nx, ng, hdtx)
    !$ACC UPDATE HOST(bx, by, bz)
    CALL bfield_final_bcs
    !$ACC UPDATE DEVICE(bx, by, bz)

    CALL update_e_field_gpu_1d(ex, ey, ez, bx, by, bz, jx, jy, jz, &
        nx, ng, cnx, fac)
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
    TYPE(particle_soa_1d), TARGET, INTENT(IN) :: soa(:)
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
