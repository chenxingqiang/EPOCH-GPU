! Copyright (C) 2025 EPOCH-GPU Development Team
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! EPOCH main loop GPU integration module
! Provides GPU-accelerated versions of the main time step routines

MODULE epoch_main_gpu

  USE gpu_config
  USE fields_gpu
  USE particles_gpu
  USE mpi_gpu
  IMPLICIT NONE

  INTEGER, PARAMETER :: num = 8
  INTEGER, PARAMETER :: i8 = SELECTED_INT_KIND(15)

  ! Time step parameters
  REAL(num) :: dt_gpu
  REAL(num) :: dx_gpu, dy_gpu
  REAL(num) :: x_min_gpu, y_min_gpu

  ! Grid parameters
  INTEGER :: nx_gpu, ny_gpu, ng_gpu

  ! MPI parameters
  INTEGER :: comm_gpu
  INTEGER :: proc_x_min_gpu, proc_x_max_gpu
  INTEGER :: proc_y_min_gpu, proc_y_max_gpu
  INTEGER :: rank_gpu, nproc_gpu

  ! Field arrays (device resident)
  REAL(num), ALLOCATABLE :: ex_gpu(:,:), ey_gpu(:,:), ez_gpu(:,:)
  REAL(num), ALLOCATABLE :: bx_gpu(:,:), by_gpu(:,:), bz_gpu(:,:)
  REAL(num), ALLOCATABLE :: jx_gpu(:,:), jy_gpu(:,:), jz_gpu(:,:)

  ! Particle arrays per species
  INTEGER :: n_species_gpu
  TYPE :: species_gpu_data
    REAL(num), ALLOCATABLE :: pos_x(:), pos_y(:)
    REAL(num), ALLOCATABLE :: px(:), py(:), pz(:)
    REAL(num), ALLOCATABLE :: weight(:)
    REAL(num), ALLOCATABLE :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), ALLOCATABLE :: bx_p(:), by_p(:), bz_p(:)
    REAL(num) :: charge, mass
    INTEGER(i8) :: count
  END TYPE species_gpu_data

  TYPE(species_gpu_data), ALLOCATABLE :: species_gpu(:)

  ! Performance counters
  REAL(num) :: time_field_update = 0.0_num
  REAL(num) :: time_particle_push = 0.0_num
  REAL(num) :: time_current_deposit = 0.0_num
  REAL(num) :: time_field_interp = 0.0_num
  REAL(num) :: time_mpi_comm = 0.0_num

  LOGICAL :: gpu_main_initialized = .FALSE.

CONTAINS

  !============================================================================
  ! Initialize GPU main loop
  !============================================================================
  SUBROUTINE epoch_gpu_main_init(nx, ny, ng, dx, dy, x_min, y_min, dt, &
                                  rank, nproc, comm, &
                                  proc_x_min, proc_x_max, &
                                  proc_y_min, proc_y_max, &
                                  n_species)
    INTEGER, INTENT(IN) :: nx, ny, ng
    REAL(num), INTENT(IN) :: dx, dy, x_min, y_min, dt
    INTEGER, INTENT(IN) :: rank, nproc, comm
    INTEGER, INTENT(IN) :: proc_x_min, proc_x_max
    INTEGER, INTENT(IN) :: proc_y_min, proc_y_max
    INTEGER, INTENT(IN) :: n_species

    ! Store parameters
    nx_gpu = nx
    ny_gpu = ny
    ng_gpu = ng
    dx_gpu = dx
    dy_gpu = dy
    x_min_gpu = x_min
    y_min_gpu = y_min
    dt_gpu = dt
    rank_gpu = rank
    nproc_gpu = nproc
    comm_gpu = comm
    proc_x_min_gpu = proc_x_min
    proc_x_max_gpu = proc_x_max
    proc_y_min_gpu = proc_y_min
    proc_y_max_gpu = proc_y_max
    n_species_gpu = n_species

    ! Initialize GPU-aware MPI
    CALL mpi_gpu_init(rank, nproc, nx, ny, ng)

    ! Allocate field arrays
    ALLOCATE(ex_gpu(1-ng:nx+ng, 1-ng:ny+ng))
    ALLOCATE(ey_gpu(1-ng:nx+ng, 1-ng:ny+ng))
    ALLOCATE(ez_gpu(1-ng:nx+ng, 1-ng:ny+ng))
    ALLOCATE(bx_gpu(1-ng:nx+ng, 1-ng:ny+ng))
    ALLOCATE(by_gpu(1-ng:nx+ng, 1-ng:ny+ng))
    ALLOCATE(bz_gpu(1-ng:nx+ng, 1-ng:ny+ng))
    ALLOCATE(jx_gpu(1-ng:nx+ng, 1-ng:ny+ng))
    ALLOCATE(jy_gpu(1-ng:nx+ng, 1-ng:ny+ng))
    ALLOCATE(jz_gpu(1-ng:nx+ng, 1-ng:ny+ng))

    ! Initialize to zero
    ex_gpu = 0.0_num; ey_gpu = 0.0_num; ez_gpu = 0.0_num
    bx_gpu = 0.0_num; by_gpu = 0.0_num; bz_gpu = 0.0_num
    jx_gpu = 0.0_num; jy_gpu = 0.0_num; jz_gpu = 0.0_num

    ! Allocate species array
    ALLOCATE(species_gpu(n_species))

    ! Create device data
    IF (GPU_ENABLED) THEN
      !$ACC ENTER DATA COPYIN(ex_gpu, ey_gpu, ez_gpu)
      !$ACC ENTER DATA COPYIN(bx_gpu, by_gpu, bz_gpu)
      !$ACC ENTER DATA COPYIN(jx_gpu, jy_gpu, jz_gpu)
    END IF

    gpu_main_initialized = .TRUE.

    IF (rank == 0) THEN
      WRITE(*,*) '============================================'
      WRITE(*,*) '     EPOCH-GPU Main Loop Initialized'
      WRITE(*,*) '============================================'
      WRITE(*,*) 'Grid:    ', nx, 'x', ny
      WRITE(*,*) 'Species: ', n_species
      WRITE(*,*) 'Ranks:   ', nproc
      WRITE(*,*) '============================================'
    END IF

  END SUBROUTINE epoch_gpu_main_init


  !============================================================================
  ! Finalize GPU main loop
  !============================================================================
  SUBROUTINE epoch_gpu_main_finalize()
    INTEGER :: isp

    IF (.NOT. gpu_main_initialized) RETURN

    ! Print performance summary
    IF (rank_gpu == 0) THEN
      WRITE(*,*) ''
      WRITE(*,*) '============================================'
      WRITE(*,*) '     EPOCH-GPU Performance Summary'
      WRITE(*,*) '============================================'
      WRITE(*,'(A,F10.3,A)') '  Field update:     ', time_field_update, ' s'
      WRITE(*,'(A,F10.3,A)') '  Particle push:    ', time_particle_push, ' s'
      WRITE(*,'(A,F10.3,A)') '  Current deposit:  ', time_current_deposit, ' s'
      WRITE(*,'(A,F10.3,A)') '  Field interp:     ', time_field_interp, ' s'
      WRITE(*,'(A,F10.3,A)') '  MPI communication:', time_mpi_comm, ' s'
      WRITE(*,*) '============================================'
    END IF

    ! Free device memory
    IF (GPU_ENABLED) THEN
      !$ACC EXIT DATA DELETE(ex_gpu, ey_gpu, ez_gpu)
      !$ACC EXIT DATA DELETE(bx_gpu, by_gpu, bz_gpu)
      !$ACC EXIT DATA DELETE(jx_gpu, jy_gpu, jz_gpu)

      ! Free species data
      DO isp = 1, n_species_gpu
        IF (ALLOCATED(species_gpu(isp)%pos_x)) THEN
          !$ACC EXIT DATA DELETE(species_gpu(isp)%pos_x, species_gpu(isp)%pos_y)
          !$ACC EXIT DATA DELETE(species_gpu(isp)%px, species_gpu(isp)%py, species_gpu(isp)%pz)
          !$ACC EXIT DATA DELETE(species_gpu(isp)%weight)
          !$ACC EXIT DATA DELETE(species_gpu(isp)%ex_p, species_gpu(isp)%ey_p, species_gpu(isp)%ez_p)
          !$ACC EXIT DATA DELETE(species_gpu(isp)%bx_p, species_gpu(isp)%by_p, species_gpu(isp)%bz_p)
        END IF
      END DO
    END IF

    ! Deallocate arrays
    IF (ALLOCATED(ex_gpu)) DEALLOCATE(ex_gpu)
    IF (ALLOCATED(ey_gpu)) DEALLOCATE(ey_gpu)
    IF (ALLOCATED(ez_gpu)) DEALLOCATE(ez_gpu)
    IF (ALLOCATED(bx_gpu)) DEALLOCATE(bx_gpu)
    IF (ALLOCATED(by_gpu)) DEALLOCATE(by_gpu)
    IF (ALLOCATED(bz_gpu)) DEALLOCATE(bz_gpu)
    IF (ALLOCATED(jx_gpu)) DEALLOCATE(jx_gpu)
    IF (ALLOCATED(jy_gpu)) DEALLOCATE(jy_gpu)
    IF (ALLOCATED(jz_gpu)) DEALLOCATE(jz_gpu)

    DO isp = 1, n_species_gpu
      IF (ALLOCATED(species_gpu(isp)%pos_x)) DEALLOCATE(species_gpu(isp)%pos_x)
      IF (ALLOCATED(species_gpu(isp)%pos_y)) DEALLOCATE(species_gpu(isp)%pos_y)
      IF (ALLOCATED(species_gpu(isp)%px)) DEALLOCATE(species_gpu(isp)%px)
      IF (ALLOCATED(species_gpu(isp)%py)) DEALLOCATE(species_gpu(isp)%py)
      IF (ALLOCATED(species_gpu(isp)%pz)) DEALLOCATE(species_gpu(isp)%pz)
      IF (ALLOCATED(species_gpu(isp)%weight)) DEALLOCATE(species_gpu(isp)%weight)
      IF (ALLOCATED(species_gpu(isp)%ex_p)) DEALLOCATE(species_gpu(isp)%ex_p)
      IF (ALLOCATED(species_gpu(isp)%ey_p)) DEALLOCATE(species_gpu(isp)%ey_p)
      IF (ALLOCATED(species_gpu(isp)%ez_p)) DEALLOCATE(species_gpu(isp)%ez_p)
      IF (ALLOCATED(species_gpu(isp)%bx_p)) DEALLOCATE(species_gpu(isp)%bx_p)
      IF (ALLOCATED(species_gpu(isp)%by_p)) DEALLOCATE(species_gpu(isp)%by_p)
      IF (ALLOCATED(species_gpu(isp)%bz_p)) DEALLOCATE(species_gpu(isp)%bz_p)
    END DO
    IF (ALLOCATED(species_gpu)) DEALLOCATE(species_gpu)

    CALL mpi_gpu_finalize()

    gpu_main_initialized = .FALSE.

  END SUBROUTINE epoch_gpu_main_finalize


  !============================================================================
  ! Initialize species on GPU
  !============================================================================
  SUBROUTINE init_species_gpu(isp, n_particles, charge, mass)
    INTEGER, INTENT(IN) :: isp
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: charge, mass

    species_gpu(isp)%count = n_particles
    species_gpu(isp)%charge = charge
    species_gpu(isp)%mass = mass

    ! Allocate particle arrays
    ALLOCATE(species_gpu(isp)%pos_x(n_particles))
    ALLOCATE(species_gpu(isp)%pos_y(n_particles))
    ALLOCATE(species_gpu(isp)%px(n_particles))
    ALLOCATE(species_gpu(isp)%py(n_particles))
    ALLOCATE(species_gpu(isp)%pz(n_particles))
    ALLOCATE(species_gpu(isp)%weight(n_particles))
    ALLOCATE(species_gpu(isp)%ex_p(n_particles))
    ALLOCATE(species_gpu(isp)%ey_p(n_particles))
    ALLOCATE(species_gpu(isp)%ez_p(n_particles))
    ALLOCATE(species_gpu(isp)%bx_p(n_particles))
    ALLOCATE(species_gpu(isp)%by_p(n_particles))
    ALLOCATE(species_gpu(isp)%bz_p(n_particles))

    ! Initialize
    species_gpu(isp)%pos_x = 0.0_num
    species_gpu(isp)%pos_y = 0.0_num
    species_gpu(isp)%px = 0.0_num
    species_gpu(isp)%py = 0.0_num
    species_gpu(isp)%pz = 0.0_num
    species_gpu(isp)%weight = 0.0_num
    species_gpu(isp)%ex_p = 0.0_num
    species_gpu(isp)%ey_p = 0.0_num
    species_gpu(isp)%ez_p = 0.0_num
    species_gpu(isp)%bx_p = 0.0_num
    species_gpu(isp)%by_p = 0.0_num
    species_gpu(isp)%bz_p = 0.0_num

    IF (GPU_ENABLED) THEN
      !$ACC ENTER DATA COPYIN(species_gpu(isp)%pos_x, species_gpu(isp)%pos_y)
      !$ACC ENTER DATA COPYIN(species_gpu(isp)%px, species_gpu(isp)%py, species_gpu(isp)%pz)
      !$ACC ENTER DATA COPYIN(species_gpu(isp)%weight)
      !$ACC ENTER DATA CREATE(species_gpu(isp)%ex_p, species_gpu(isp)%ey_p, species_gpu(isp)%ez_p)
      !$ACC ENTER DATA CREATE(species_gpu(isp)%bx_p, species_gpu(isp)%by_p, species_gpu(isp)%bz_p)
    END IF

  END SUBROUTINE init_species_gpu


  !============================================================================
  ! Copy fields from CPU to GPU
  !============================================================================
  SUBROUTINE copy_fields_to_gpu(ex, ey, ez, bx, by, bz)
    REAL(num), INTENT(IN) :: ex(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(IN) :: ey(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(IN) :: ez(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(IN) :: bx(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(IN) :: by(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(IN) :: bz(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)

    ex_gpu = ex; ey_gpu = ey; ez_gpu = ez
    bx_gpu = bx; by_gpu = by; bz_gpu = bz

    IF (GPU_ENABLED) THEN
      !$ACC UPDATE DEVICE(ex_gpu, ey_gpu, ez_gpu, bx_gpu, by_gpu, bz_gpu)
    END IF

  END SUBROUTINE copy_fields_to_gpu


  !============================================================================
  ! Copy fields from GPU to CPU
  !============================================================================
  SUBROUTINE copy_fields_from_gpu(ex, ey, ez, bx, by, bz)
    REAL(num), INTENT(OUT) :: ex(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(OUT) :: ey(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(OUT) :: ez(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(OUT) :: bx(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(OUT) :: by(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)
    REAL(num), INTENT(OUT) :: bz(1-ng_gpu:nx_gpu+ng_gpu, 1-ng_gpu:ny_gpu+ng_gpu)

    IF (GPU_ENABLED) THEN
      !$ACC UPDATE HOST(ex_gpu, ey_gpu, ez_gpu, bx_gpu, by_gpu, bz_gpu)
    END IF

    ex = ex_gpu; ey = ey_gpu; ez = ez_gpu
    bx = bx_gpu; by = by_gpu; bz = bz_gpu

  END SUBROUTINE copy_fields_from_gpu


  !============================================================================
  ! GPU-accelerated main time step
  !============================================================================
  SUBROUTINE epoch_gpu_time_step()
    INTEGER :: isp
    REAL(num) :: t_start, t_end
    REAL(num) :: cnx, cny, fac, hdtx, hdty
    REAL(num), PARAMETER :: c = 299792458.0_num
    REAL(num), PARAMETER :: epsilon_0 = 8.854187817e-12_num

    ! Compute coefficients
    cnx = c * dt_gpu / dx_gpu
    cny = c * dt_gpu / dy_gpu
    fac = dt_gpu / (epsilon_0 * dx_gpu * dy_gpu)
    hdtx = 0.5_num * dt_gpu / dx_gpu
    hdty = 0.5_num * dt_gpu / dy_gpu

    !--------------------------------------------------------------------------
    ! Step 1: Half B update
    !--------------------------------------------------------------------------
    CALL CPU_TIME(t_start)
    CALL update_b_field_gpu(ex_gpu, ey_gpu, ez_gpu, bx_gpu, by_gpu, bz_gpu, &
                            nx_gpu, ny_gpu, ng_gpu, hdtx, hdty)
    !$ACC WAIT
    CALL CPU_TIME(t_end)
    time_field_update = time_field_update + (t_end - t_start)

    !--------------------------------------------------------------------------
    ! Step 2: Boundary exchange for B
    !--------------------------------------------------------------------------
    CALL CPU_TIME(t_start)
    CALL exchange_fields_gpu(ex_gpu, ey_gpu, ez_gpu, bx_gpu, by_gpu, bz_gpu, &
                             nx_gpu, ny_gpu, ng_gpu, &
                             proc_x_min_gpu, proc_x_max_gpu, &
                             proc_y_min_gpu, proc_y_max_gpu, comm_gpu)
    CALL CPU_TIME(t_end)
    time_mpi_comm = time_mpi_comm + (t_end - t_start)

    !--------------------------------------------------------------------------
    ! Step 3: Particle operations for each species
    !--------------------------------------------------------------------------
    DO isp = 1, n_species_gpu
      IF (species_gpu(isp)%count == 0) CYCLE

      ! Interpolate fields to particles
      CALL CPU_TIME(t_start)
      CALL interpolate_fields_to_particles_gpu( &
          species_gpu(isp)%count, &
          species_gpu(isp)%pos_x, species_gpu(isp)%pos_y, &
          ex_gpu, ey_gpu, ez_gpu, bx_gpu, by_gpu, bz_gpu, &
          species_gpu(isp)%ex_p, species_gpu(isp)%ey_p, species_gpu(isp)%ez_p, &
          species_gpu(isp)%bx_p, species_gpu(isp)%by_p, species_gpu(isp)%bz_p, &
          x_min_gpu, y_min_gpu, dx_gpu, dy_gpu, nx_gpu, ny_gpu, ng_gpu)
      !$ACC WAIT
      CALL CPU_TIME(t_end)
      time_field_interp = time_field_interp + (t_end - t_start)

      ! Push particles (Boris)
      CALL CPU_TIME(t_start)
      CALL push_particles_gpu( &
          species_gpu(isp)%count, &
          species_gpu(isp)%pos_x, species_gpu(isp)%pos_y, &
          species_gpu(isp)%px, species_gpu(isp)%py, species_gpu(isp)%pz, &
          species_gpu(isp)%ex_p, species_gpu(isp)%ey_p, species_gpu(isp)%ez_p, &
          species_gpu(isp)%bx_p, species_gpu(isp)%by_p, species_gpu(isp)%bz_p, &
          species_gpu(isp)%charge, species_gpu(isp)%mass, dt_gpu)
      !$ACC WAIT
      CALL CPU_TIME(t_end)
      time_particle_push = time_particle_push + (t_end - t_start)

      ! Deposit current
      CALL CPU_TIME(t_start)
      CALL deposit_current_gpu( &
          species_gpu(isp)%count, &
          species_gpu(isp)%pos_x, species_gpu(isp)%pos_y, &
          species_gpu(isp)%pos_x, species_gpu(isp)%pos_y, &  ! Use same for old (simplified)
          species_gpu(isp)%px, species_gpu(isp)%py, species_gpu(isp)%pz, &
          1.0_num, &  ! Scalar weight
          species_gpu(isp)%charge, species_gpu(isp)%mass, &
          jx_gpu, jy_gpu, jz_gpu, &
          x_min_gpu, y_min_gpu, dx_gpu, dy_gpu, dt_gpu, &
          nx_gpu, ny_gpu, ng_gpu)
      !$ACC WAIT
      CALL CPU_TIME(t_end)
      time_current_deposit = time_current_deposit + (t_end - t_start)
    END DO

    !--------------------------------------------------------------------------
    ! Step 4: E field update with current
    !--------------------------------------------------------------------------
    CALL CPU_TIME(t_start)
    CALL update_e_field_gpu(ex_gpu, ey_gpu, ez_gpu, bx_gpu, by_gpu, bz_gpu, &
                            jx_gpu, jy_gpu, jz_gpu, &
                            nx_gpu, ny_gpu, ng_gpu, cnx, cny, fac)
    !$ACC WAIT
    CALL CPU_TIME(t_end)
    time_field_update = time_field_update + (t_end - t_start)

    !--------------------------------------------------------------------------
    ! Step 5: Boundary exchange for E
    !--------------------------------------------------------------------------
    CALL CPU_TIME(t_start)
    CALL exchange_fields_gpu(ex_gpu, ey_gpu, ez_gpu, bx_gpu, by_gpu, bz_gpu, &
                             nx_gpu, ny_gpu, ng_gpu, &
                             proc_x_min_gpu, proc_x_max_gpu, &
                             proc_y_min_gpu, proc_y_max_gpu, comm_gpu)
    CALL CPU_TIME(t_end)
    time_mpi_comm = time_mpi_comm + (t_end - t_start)

    !--------------------------------------------------------------------------
    ! Step 6: Second half B update
    !--------------------------------------------------------------------------
    CALL CPU_TIME(t_start)
    CALL update_b_field_gpu(ex_gpu, ey_gpu, ez_gpu, bx_gpu, by_gpu, bz_gpu, &
                            nx_gpu, ny_gpu, ng_gpu, hdtx, hdty)
    !$ACC WAIT
    CALL CPU_TIME(t_end)
    time_field_update = time_field_update + (t_end - t_start)

    ! Reset current for next step
    !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(jx_gpu, jy_gpu, jz_gpu)
    DO iy = 1-ng_gpu, ny_gpu+ng_gpu
      DO ix = 1-ng_gpu, nx_gpu+ng_gpu
        jx_gpu(ix, iy) = 0.0_num
        jy_gpu(ix, iy) = 0.0_num
        jz_gpu(ix, iy) = 0.0_num
      END DO
    END DO
    !$ACC END PARALLEL LOOP

  END SUBROUTINE epoch_gpu_time_step

END MODULE epoch_main_gpu

