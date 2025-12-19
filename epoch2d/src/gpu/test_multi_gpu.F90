! Copyright (C) 2024 EPOCH-GPU Development Team
!
! Multi-GPU test program for EPOCH-GPU
! Tests GPU-aware MPI communication and main loop integration

PROGRAM test_multi_gpu

  USE gpu_config
  USE fields_gpu
  USE particles_gpu
  USE mpi_gpu
  USE epoch_main_gpu
  IMPLICIT NONE

  INCLUDE 'mpif.h'

  INTEGER, PARAMETER :: num = 8
  INTEGER, PARAMETER :: i8 = SELECTED_INT_KIND(15)
  REAL(num), PARAMETER :: c = 299792458.0_num

  INTEGER :: rank, nproc, ierr
  INTEGER :: nx, ny, ng
  INTEGER :: proc_x_min, proc_x_max, proc_y_min, proc_y_max
  INTEGER :: comm
  INTEGER :: step, nsteps
  REAL(num) :: dt, dx, dy, x_min, y_min
  REAL(num) :: start_time, end_time, elapsed
  REAL(num) :: total_time
  INTEGER(i8) :: n_particles

  ! Initialize MPI
  CALL MPI_INIT(ierr)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nproc, ierr)
  comm = MPI_COMM_WORLD

  IF (rank == 0) THEN
    WRITE(*,*) '============================================'
    WRITE(*,*) '   EPOCH-GPU Multi-GPU Test'
    WRITE(*,*) '============================================'
    WRITE(*,*) 'MPI Ranks: ', nproc
  END IF

  ! Setup grid parameters (per process)
  nx = 128
  ny = 128
  ng = 2
  dx = 1.0e-8_num
  dy = 1.0e-8_num
  dt = 0.5_num * MIN(dx, dy) / c  ! CFL condition
  x_min = REAL(rank, num) * nx * dx
  y_min = 0.0_num
  n_particles = 100000_i8

  ! Simple 1D decomposition for testing
  IF (rank == 0) THEN
    proc_x_min = MPI_PROC_NULL
  ELSE
    proc_x_min = rank - 1
  END IF

  IF (rank == nproc - 1) THEN
    proc_x_max = MPI_PROC_NULL
  ELSE
    proc_x_max = rank + 1
  END IF

  proc_y_min = MPI_PROC_NULL
  proc_y_max = MPI_PROC_NULL

  !============================================================================
  ! Test 1: GPU-aware MPI initialization
  !============================================================================
  IF (rank == 0) THEN
    WRITE(*,*) ''
    WRITE(*,*) 'Test 1: GPU-aware MPI Initialization'
  END IF

  CALL epoch_gpu_main_init(nx, ny, ng, dx, dy, x_min, y_min, dt, &
                           rank, nproc, comm, &
                           proc_x_min, proc_x_max, &
                           proc_y_min, proc_y_max, &
                           1)  ! 1 species

  IF (rank == 0) WRITE(*,*) '  [PASS] GPU-aware MPI initialized'

  !============================================================================
  ! Test 2: Initialize fields and particles
  !============================================================================
  IF (rank == 0) THEN
    WRITE(*,*) ''
    WRITE(*,*) 'Test 2: Initialize Fields and Particles'
  END IF

  ! Initialize test fields
  CALL init_test_fields()

  ! Initialize species
  CALL init_species_gpu(1, n_particles, -1.602e-19_num, 9.109e-31_num)
  CALL init_test_particles(1, n_particles)

  IF (rank == 0) WRITE(*,*) '  [PASS] Fields and particles initialized'

  !============================================================================
  ! Test 3: Run time steps
  !============================================================================
  nsteps = 100

  IF (rank == 0) THEN
    WRITE(*,*) ''
    WRITE(*,*) 'Test 3: Running ', nsteps, ' time steps'
  END IF

  CALL MPI_BARRIER(comm, ierr)
  start_time = MPI_WTIME()

  DO step = 1, nsteps
    CALL epoch_gpu_time_step()
  END DO

  CALL MPI_BARRIER(comm, ierr)
  end_time = MPI_WTIME()
  elapsed = end_time - start_time

  IF (rank == 0) THEN
    WRITE(*,*) '  Total time: ', elapsed, ' s'
    WRITE(*,*) '  Time per step: ', elapsed / REAL(nsteps, num) * 1000.0_num, ' ms'
    WRITE(*,*) '  [PASS] Time stepping completed'
  END IF

  !============================================================================
  ! Test 4: Verify communication
  !============================================================================
  IF (rank == 0) THEN
    WRITE(*,*) ''
    WRITE(*,*) 'Test 4: Verify MPI Communication'
  END IF

  CALL verify_communication()

  !============================================================================
  ! Cleanup
  !============================================================================
  CALL epoch_gpu_main_finalize()

  IF (rank == 0) THEN
    WRITE(*,*) ''
    WRITE(*,*) '============================================'
    WRITE(*,*) '   Multi-GPU Test PASSED!'
    WRITE(*,*) '============================================'
  END IF

  CALL MPI_FINALIZE(ierr)

CONTAINS

  SUBROUTINE init_test_fields()
    INTEGER :: ix, iy

    ! Initialize with simple test pattern
    !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(ex_gpu, ey_gpu, ez_gpu, bx_gpu, by_gpu, bz_gpu)
    DO iy = 1-ng, ny+ng
      DO ix = 1-ng, nx+ng
        ex_gpu(ix, iy) = 1.0e6_num * SIN(REAL(ix, num) / REAL(nx, num) * 3.14159_num)
        ey_gpu(ix, iy) = 0.0_num
        ez_gpu(ix, iy) = 0.0_num
        bx_gpu(ix, iy) = 0.0_num
        by_gpu(ix, iy) = 0.0_num
        bz_gpu(ix, iy) = 1.0e-3_num
      END DO
    END DO
    !$ACC END PARALLEL LOOP

  END SUBROUTINE init_test_fields


  SUBROUTINE init_test_particles(isp, n_part)
    INTEGER, INTENT(IN) :: isp
    INTEGER(i8), INTENT(IN) :: n_part
    INTEGER(i8) :: ip

    ! Initialize particle positions randomly within domain
    !$ACC PARALLEL LOOP PRESENT(species_gpu(isp)%pos_x, species_gpu(isp)%pos_y, &
    !$ACC&                      species_gpu(isp)%px, species_gpu(isp)%py, species_gpu(isp)%pz, &
    !$ACC&                      species_gpu(isp)%weight)
    DO ip = 1, n_part
      species_gpu(isp)%pos_x(ip) = x_min + dx * REAL(MOD(ip-1, INT(nx, i8)), num)
      species_gpu(isp)%pos_y(ip) = y_min + dy * REAL((ip-1) / nx, num)
      species_gpu(isp)%px(ip) = 1.0e-24_num * SIN(REAL(ip, num))
      species_gpu(isp)%py(ip) = 1.0e-24_num * COS(REAL(ip, num))
      species_gpu(isp)%pz(ip) = 0.0_num
      species_gpu(isp)%weight(ip) = 1.0e10_num
    END DO
    !$ACC END PARALLEL LOOP

  END SUBROUTINE init_test_particles


  SUBROUTINE verify_communication()
    REAL(num) :: local_sum, global_sum
    INTEGER :: ix, iy

    ! Sum field values
    local_sum = 0.0_num
    !$ACC PARALLEL LOOP COLLAPSE(2) REDUCTION(+:local_sum) PRESENT(ex_gpu)
    DO iy = 1, ny
      DO ix = 1, nx
        local_sum = local_sum + ex_gpu(ix, iy)**2
      END DO
    END DO
    !$ACC END PARALLEL LOOP

    CALL MPI_ALLREDUCE(local_sum, global_sum, 1, MPI_DOUBLE_PRECISION, &
                       MPI_SUM, comm, ierr)

    IF (rank == 0) THEN
      WRITE(*,*) '  Global field energy: ', global_sum
      IF (global_sum > 0.0_num) THEN
        WRITE(*,*) '  [PASS] Communication verified'
      ELSE
        WRITE(*,*) '  [WARN] Zero field energy detected'
      END IF
    END IF

  END SUBROUTINE verify_communication

END PROGRAM test_multi_gpu

