! Copyright (C) 2024 EPOCH-GPU Development Team
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! GPU-aware MPI module for multi-GPU EPOCH simulations
! Provides direct GPU-to-GPU communication when CUDA-aware MPI is available

MODULE mpi_gpu

  USE gpu_config
  IMPLICIT NONE

  INTEGER, PARAMETER :: num = 8
  INTEGER, PARAMETER :: i8 = SELECTED_INT_KIND(15)

  ! GPU-aware MPI configuration
  LOGICAL :: cuda_aware_mpi = .FALSE.
  LOGICAL :: gpu_direct_enabled = .FALSE.
  INTEGER :: local_gpu_id = 0
  INTEGER :: gpus_per_node = 1

  ! Communication buffers on device
  REAL(num), ALLOCATABLE :: send_buf_x_min(:,:)
  REAL(num), ALLOCATABLE :: send_buf_x_max(:,:)
  REAL(num), ALLOCATABLE :: send_buf_y_min(:,:)
  REAL(num), ALLOCATABLE :: send_buf_y_max(:,:)
  REAL(num), ALLOCATABLE :: recv_buf_x_min(:,:)
  REAL(num), ALLOCATABLE :: recv_buf_x_max(:,:)
  REAL(num), ALLOCATABLE :: recv_buf_y_min(:,:)
  REAL(num), ALLOCATABLE :: recv_buf_y_max(:,:)

  ! Buffer sizes
  INTEGER :: buf_size_x, buf_size_y
  INTEGER :: ng_comm  ! Ghost cells for communication

CONTAINS

  !============================================================================
  ! Initialize GPU-aware MPI
  !============================================================================
  SUBROUTINE mpi_gpu_init(rank, nproc, nx, ny, ng)
    INTEGER, INTENT(IN) :: rank, nproc, nx, ny, ng
    INTEGER :: local_rank, ierr
    CHARACTER(LEN=256) :: env_val

    ng_comm = ng

    ! Detect CUDA-aware MPI
    CALL detect_cuda_aware_mpi()

    ! Assign GPU based on local rank
    CALL get_local_rank(rank, local_rank)
    CALL detect_gpus_per_node()

    local_gpu_id = MOD(local_rank, gpus_per_node)

    IF (GPU_ENABLED) THEN
      CALL gpu_init(local_gpu_id)
    END IF

    IF (rank == 0) THEN
      WRITE(*,*) '============================================'
      WRITE(*,*) '       GPU-aware MPI Configuration'
      WRITE(*,*) '============================================'
      WRITE(*,*) 'Total MPI ranks:     ', nproc
      WRITE(*,*) 'GPUs per node:       ', gpus_per_node
      IF (cuda_aware_mpi) THEN
        WRITE(*,*) 'CUDA-aware MPI:      ENABLED'
        WRITE(*,*) 'GPU Direct:          ', gpu_direct_enabled
      ELSE
        WRITE(*,*) 'CUDA-aware MPI:      DISABLED (using staged transfers)'
      END IF
      WRITE(*,*) '============================================'
    END IF

    ! Allocate communication buffers
    buf_size_x = ny + 2 * ng
    buf_size_y = nx + 2 * ng

    ALLOCATE(send_buf_x_min(ng, buf_size_x))
    ALLOCATE(send_buf_x_max(ng, buf_size_x))
    ALLOCATE(recv_buf_x_min(ng, buf_size_x))
    ALLOCATE(recv_buf_x_max(ng, buf_size_x))

    ALLOCATE(send_buf_y_min(buf_size_y, ng))
    ALLOCATE(send_buf_y_max(buf_size_y, ng))
    ALLOCATE(recv_buf_y_min(buf_size_y, ng))
    ALLOCATE(recv_buf_y_max(buf_size_y, ng))

    ! Initialize buffers
    send_buf_x_min = 0.0_num
    send_buf_x_max = 0.0_num
    recv_buf_x_min = 0.0_num
    recv_buf_x_max = 0.0_num
    send_buf_y_min = 0.0_num
    send_buf_y_max = 0.0_num
    recv_buf_y_min = 0.0_num
    recv_buf_y_max = 0.0_num

    IF (GPU_ENABLED) THEN
      !$ACC ENTER DATA CREATE(send_buf_x_min, send_buf_x_max)
      !$ACC ENTER DATA CREATE(recv_buf_x_min, recv_buf_x_max)
      !$ACC ENTER DATA CREATE(send_buf_y_min, send_buf_y_max)
      !$ACC ENTER DATA CREATE(recv_buf_y_min, recv_buf_y_max)
    END IF

  END SUBROUTINE mpi_gpu_init


  !============================================================================
  ! Finalize GPU-aware MPI
  !============================================================================
  SUBROUTINE mpi_gpu_finalize()

    IF (GPU_ENABLED) THEN
      !$ACC EXIT DATA DELETE(send_buf_x_min, send_buf_x_max)
      !$ACC EXIT DATA DELETE(recv_buf_x_min, recv_buf_x_max)
      !$ACC EXIT DATA DELETE(send_buf_y_min, send_buf_y_max)
      !$ACC EXIT DATA DELETE(recv_buf_y_min, recv_buf_y_max)
    END IF

    IF (ALLOCATED(send_buf_x_min)) DEALLOCATE(send_buf_x_min)
    IF (ALLOCATED(send_buf_x_max)) DEALLOCATE(send_buf_x_max)
    IF (ALLOCATED(recv_buf_x_min)) DEALLOCATE(recv_buf_x_min)
    IF (ALLOCATED(recv_buf_x_max)) DEALLOCATE(recv_buf_x_max)
    IF (ALLOCATED(send_buf_y_min)) DEALLOCATE(send_buf_y_min)
    IF (ALLOCATED(send_buf_y_max)) DEALLOCATE(send_buf_y_max)
    IF (ALLOCATED(recv_buf_y_min)) DEALLOCATE(recv_buf_y_min)
    IF (ALLOCATED(recv_buf_y_max)) DEALLOCATE(recv_buf_y_max)

    CALL gpu_finalize()

  END SUBROUTINE mpi_gpu_finalize


  !============================================================================
  ! Detect if MPI is CUDA-aware
  !============================================================================
  SUBROUTINE detect_cuda_aware_mpi()
    CHARACTER(LEN=256) :: env_val
    INTEGER :: status

    cuda_aware_mpi = .FALSE.
    gpu_direct_enabled = .FALSE.

    ! Check for NVIDIA HPC SDK MPI
    CALL GET_ENVIRONMENT_VARIABLE('OMPI_MCA_mpi_cuda_support', env_val, STATUS=status)
    IF (status == 0 .AND. TRIM(env_val) == '1') THEN
      cuda_aware_mpi = .TRUE.
    END IF

    ! Check for OpenMPI CUDA support
    CALL GET_ENVIRONMENT_VARIABLE('OMPI_MCA_btl_openib_want_cuda_gdr', env_val, STATUS=status)
    IF (status == 0 .AND. TRIM(env_val) == '1') THEN
      gpu_direct_enabled = .TRUE.
    END IF

    ! Check for MVAPICH CUDA support
    CALL GET_ENVIRONMENT_VARIABLE('MV2_USE_CUDA', env_val, STATUS=status)
    IF (status == 0 .AND. TRIM(env_val) == '1') THEN
      cuda_aware_mpi = .TRUE.
    END IF

    ! Check for explicit enable flag
    CALL GET_ENVIRONMENT_VARIABLE('EPOCH_GPU_AWARE_MPI', env_val, STATUS=status)
    IF (status == 0) THEN
      IF (TRIM(env_val) == '1' .OR. TRIM(env_val) == 'true') THEN
        cuda_aware_mpi = .TRUE.
      ELSE IF (TRIM(env_val) == '0' .OR. TRIM(env_val) == 'false') THEN
        cuda_aware_mpi = .FALSE.
      END IF
    END IF

  END SUBROUTINE detect_cuda_aware_mpi


  !============================================================================
  ! Get local rank within node
  !============================================================================
  SUBROUTINE get_local_rank(global_rank, local_rank)
    INTEGER, INTENT(IN) :: global_rank
    INTEGER, INTENT(OUT) :: local_rank
    CHARACTER(LEN=256) :: env_val
    INTEGER :: status

    local_rank = 0

    ! Try OpenMPI
    CALL GET_ENVIRONMENT_VARIABLE('OMPI_COMM_WORLD_LOCAL_RANK', env_val, STATUS=status)
    IF (status == 0) THEN
      READ(env_val, *) local_rank
      RETURN
    END IF

    ! Try MVAPICH
    CALL GET_ENVIRONMENT_VARIABLE('MV2_COMM_WORLD_LOCAL_RANK', env_val, STATUS=status)
    IF (status == 0) THEN
      READ(env_val, *) local_rank
      RETURN
    END IF

    ! Try SLURM
    CALL GET_ENVIRONMENT_VARIABLE('SLURM_LOCALID', env_val, STATUS=status)
    IF (status == 0) THEN
      READ(env_val, *) local_rank
      RETURN
    END IF

    ! Fallback: use global rank modulo GPUs per node
    local_rank = MOD(global_rank, gpus_per_node)

  END SUBROUTINE get_local_rank


  !============================================================================
  ! Detect GPUs per node
  !============================================================================
  SUBROUTINE detect_gpus_per_node()
    CHARACTER(LEN=256) :: env_val
    INTEGER :: status

    gpus_per_node = 1

    ! Try environment variable
    CALL GET_ENVIRONMENT_VARIABLE('EPOCH_GPUS_PER_NODE', env_val, STATUS=status)
    IF (status == 0) THEN
      READ(env_val, *) gpus_per_node
      RETURN
    END IF

    ! Try CUDA_VISIBLE_DEVICES count
    CALL GET_ENVIRONMENT_VARIABLE('CUDA_VISIBLE_DEVICES', env_val, STATUS=status)
    IF (status == 0) THEN
      gpus_per_node = count_devices(env_val)
      RETURN
    END IF

    ! Default to querying device count (would need CUDA API)
    gpus_per_node = 1

  END SUBROUTINE detect_gpus_per_node


  !============================================================================
  ! Count devices in CUDA_VISIBLE_DEVICES string
  !============================================================================
  FUNCTION count_devices(devices_str) RESULT(n)
    CHARACTER(LEN=*), INTENT(IN) :: devices_str
    INTEGER :: n
    INTEGER :: i

    n = 1
    DO i = 1, LEN_TRIM(devices_str)
      IF (devices_str(i:i) == ',') n = n + 1
    END DO

  END FUNCTION count_devices


  !============================================================================
  ! Exchange field boundary data (X direction)
  !============================================================================
  SUBROUTINE exchange_field_x_gpu(field, nx, ny, ng, proc_x_min, proc_x_max, comm)
    REAL(num), INTENT(INOUT) :: field(1-ng:nx+ng, 1-ng:ny+ng)
    INTEGER, INTENT(IN) :: nx, ny, ng
    INTEGER, INTENT(IN) :: proc_x_min, proc_x_max, comm

    INTEGER :: ix, iy, ierr
    INTEGER :: sendcount, recvcount
    INTEGER :: requests(4), statuses(MPI_STATUS_SIZE, 4)
    INTEGER, PARAMETER :: tag_xmin = 100, tag_xmax = 101

    INCLUDE 'mpif.h'

    sendcount = ng * (ny + 2*ng)
    recvcount = sendcount

    IF (GPU_ENABLED) THEN
      ! Pack send buffers on GPU
      !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(field, send_buf_x_min, send_buf_x_max)
      DO iy = 1-ng, ny+ng
        DO ix = 1, ng
          send_buf_x_min(ix, iy+ng) = field(ix, iy)
          send_buf_x_max(ix, iy+ng) = field(nx-ng+ix, iy)
        END DO
      END DO
      !$ACC END PARALLEL LOOP

      IF (cuda_aware_mpi) THEN
        ! Direct GPU-to-GPU communication
        !$ACC HOST_DATA USE_DEVICE(send_buf_x_min, send_buf_x_max, &
        !$ACC&                     recv_buf_x_min, recv_buf_x_max)

        CALL MPI_IRECV(recv_buf_x_min, recvcount, MPI_DOUBLE_PRECISION, &
                       proc_x_min, tag_xmax, comm, requests(1), ierr)
        CALL MPI_IRECV(recv_buf_x_max, recvcount, MPI_DOUBLE_PRECISION, &
                       proc_x_max, tag_xmin, comm, requests(2), ierr)
        CALL MPI_ISEND(send_buf_x_min, sendcount, MPI_DOUBLE_PRECISION, &
                       proc_x_min, tag_xmin, comm, requests(3), ierr)
        CALL MPI_ISEND(send_buf_x_max, sendcount, MPI_DOUBLE_PRECISION, &
                       proc_x_max, tag_xmax, comm, requests(4), ierr)

        CALL MPI_WAITALL(4, requests, statuses, ierr)

        !$ACC END HOST_DATA
      ELSE
        ! Staged transfer: GPU -> Host -> MPI -> Host -> GPU
        !$ACC UPDATE HOST(send_buf_x_min, send_buf_x_max)

        CALL MPI_IRECV(recv_buf_x_min, recvcount, MPI_DOUBLE_PRECISION, &
                       proc_x_min, tag_xmax, comm, requests(1), ierr)
        CALL MPI_IRECV(recv_buf_x_max, recvcount, MPI_DOUBLE_PRECISION, &
                       proc_x_max, tag_xmin, comm, requests(2), ierr)
        CALL MPI_ISEND(send_buf_x_min, sendcount, MPI_DOUBLE_PRECISION, &
                       proc_x_min, tag_xmin, comm, requests(3), ierr)
        CALL MPI_ISEND(send_buf_x_max, sendcount, MPI_DOUBLE_PRECISION, &
                       proc_x_max, tag_xmax, comm, requests(4), ierr)

        CALL MPI_WAITALL(4, requests, statuses, ierr)

        !$ACC UPDATE DEVICE(recv_buf_x_min, recv_buf_x_max)
      END IF

      ! Unpack receive buffers on GPU
      !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(field, recv_buf_x_min, recv_buf_x_max)
      DO iy = 1-ng, ny+ng
        DO ix = 1, ng
          field(1-ng+ix-1, iy) = recv_buf_x_min(ix, iy+ng)
          field(nx+ix, iy) = recv_buf_x_max(ix, iy+ng)
        END DO
      END DO
      !$ACC END PARALLEL LOOP

    ELSE
      ! CPU fallback - standard MPI
      ! Pack send buffers
      DO iy = 1-ng, ny+ng
        DO ix = 1, ng
          send_buf_x_min(ix, iy+ng) = field(ix, iy)
          send_buf_x_max(ix, iy+ng) = field(nx-ng+ix, iy)
        END DO
      END DO

      CALL MPI_IRECV(recv_buf_x_min, recvcount, MPI_DOUBLE_PRECISION, &
                     proc_x_min, tag_xmax, comm, requests(1), ierr)
      CALL MPI_IRECV(recv_buf_x_max, recvcount, MPI_DOUBLE_PRECISION, &
                     proc_x_max, tag_xmin, comm, requests(2), ierr)
      CALL MPI_ISEND(send_buf_x_min, sendcount, MPI_DOUBLE_PRECISION, &
                     proc_x_min, tag_xmin, comm, requests(3), ierr)
      CALL MPI_ISEND(send_buf_x_max, sendcount, MPI_DOUBLE_PRECISION, &
                     proc_x_max, tag_xmax, comm, requests(4), ierr)

      CALL MPI_WAITALL(4, requests, statuses, ierr)

      ! Unpack receive buffers
      DO iy = 1-ng, ny+ng
        DO ix = 1, ng
          field(1-ng+ix-1, iy) = recv_buf_x_min(ix, iy+ng)
          field(nx+ix, iy) = recv_buf_x_max(ix, iy+ng)
        END DO
      END DO
    END IF

  END SUBROUTINE exchange_field_x_gpu


  !============================================================================
  ! Exchange field boundary data (Y direction)
  !============================================================================
  SUBROUTINE exchange_field_y_gpu(field, nx, ny, ng, proc_y_min, proc_y_max, comm)
    REAL(num), INTENT(INOUT) :: field(1-ng:nx+ng, 1-ng:ny+ng)
    INTEGER, INTENT(IN) :: nx, ny, ng
    INTEGER, INTENT(IN) :: proc_y_min, proc_y_max, comm

    INTEGER :: ix, iy, ierr
    INTEGER :: sendcount, recvcount
    INTEGER :: requests(4), statuses(MPI_STATUS_SIZE, 4)
    INTEGER, PARAMETER :: tag_ymin = 200, tag_ymax = 201

    INCLUDE 'mpif.h'

    sendcount = (nx + 2*ng) * ng
    recvcount = sendcount

    IF (GPU_ENABLED) THEN
      ! Pack send buffers on GPU
      !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(field, send_buf_y_min, send_buf_y_max)
      DO iy = 1, ng
        DO ix = 1-ng, nx+ng
          send_buf_y_min(ix+ng, iy) = field(ix, iy)
          send_buf_y_max(ix+ng, iy) = field(ix, ny-ng+iy)
        END DO
      END DO
      !$ACC END PARALLEL LOOP

      IF (cuda_aware_mpi) THEN
        !$ACC HOST_DATA USE_DEVICE(send_buf_y_min, send_buf_y_max, &
        !$ACC&                     recv_buf_y_min, recv_buf_y_max)

        CALL MPI_IRECV(recv_buf_y_min, recvcount, MPI_DOUBLE_PRECISION, &
                       proc_y_min, tag_ymax, comm, requests(1), ierr)
        CALL MPI_IRECV(recv_buf_y_max, recvcount, MPI_DOUBLE_PRECISION, &
                       proc_y_max, tag_ymin, comm, requests(2), ierr)
        CALL MPI_ISEND(send_buf_y_min, sendcount, MPI_DOUBLE_PRECISION, &
                       proc_y_min, tag_ymin, comm, requests(3), ierr)
        CALL MPI_ISEND(send_buf_y_max, sendcount, MPI_DOUBLE_PRECISION, &
                       proc_y_max, tag_ymax, comm, requests(4), ierr)

        CALL MPI_WAITALL(4, requests, statuses, ierr)

        !$ACC END HOST_DATA
      ELSE
        !$ACC UPDATE HOST(send_buf_y_min, send_buf_y_max)

        CALL MPI_IRECV(recv_buf_y_min, recvcount, MPI_DOUBLE_PRECISION, &
                       proc_y_min, tag_ymax, comm, requests(1), ierr)
        CALL MPI_IRECV(recv_buf_y_max, recvcount, MPI_DOUBLE_PRECISION, &
                       proc_y_max, tag_ymin, comm, requests(2), ierr)
        CALL MPI_ISEND(send_buf_y_min, sendcount, MPI_DOUBLE_PRECISION, &
                       proc_y_min, tag_ymin, comm, requests(3), ierr)
        CALL MPI_ISEND(send_buf_y_max, sendcount, MPI_DOUBLE_PRECISION, &
                       proc_y_max, tag_ymax, comm, requests(4), ierr)

        CALL MPI_WAITALL(4, requests, statuses, ierr)

        !$ACC UPDATE DEVICE(recv_buf_y_min, recv_buf_y_max)
      END IF

      ! Unpack receive buffers on GPU
      !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(field, recv_buf_y_min, recv_buf_y_max)
      DO iy = 1, ng
        DO ix = 1-ng, nx+ng
          field(ix, 1-ng+iy-1) = recv_buf_y_min(ix+ng, iy)
          field(ix, ny+iy) = recv_buf_y_max(ix+ng, iy)
        END DO
      END DO
      !$ACC END PARALLEL LOOP

    ELSE
      ! CPU fallback
      DO iy = 1, ng
        DO ix = 1-ng, nx+ng
          send_buf_y_min(ix+ng, iy) = field(ix, iy)
          send_buf_y_max(ix+ng, iy) = field(ix, ny-ng+iy)
        END DO
      END DO

      CALL MPI_IRECV(recv_buf_y_min, recvcount, MPI_DOUBLE_PRECISION, &
                     proc_y_min, tag_ymax, comm, requests(1), ierr)
      CALL MPI_IRECV(recv_buf_y_max, recvcount, MPI_DOUBLE_PRECISION, &
                     proc_y_max, tag_ymin, comm, requests(2), ierr)
      CALL MPI_ISEND(send_buf_y_min, sendcount, MPI_DOUBLE_PRECISION, &
                     proc_y_min, tag_ymin, comm, requests(3), ierr)
      CALL MPI_ISEND(send_buf_y_max, sendcount, MPI_DOUBLE_PRECISION, &
                     proc_y_max, tag_ymax, comm, requests(4), ierr)

      CALL MPI_WAITALL(4, requests, statuses, ierr)

      DO iy = 1, ng
        DO ix = 1-ng, nx+ng
          field(ix, 1-ng+iy-1) = recv_buf_y_min(ix+ng, iy)
          field(ix, ny+iy) = recv_buf_y_max(ix+ng, iy)
        END DO
      END DO
    END IF

  END SUBROUTINE exchange_field_y_gpu


  !============================================================================
  ! Exchange all field boundaries
  !============================================================================
  SUBROUTINE exchange_fields_gpu(ex, ey, ez, bx, by, bz, &
                                  nx, ny, ng, &
                                  proc_x_min, proc_x_max, &
                                  proc_y_min, proc_y_max, comm)
    REAL(num), INTENT(INOUT) :: ex(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ey(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ez(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: by(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bz(1-ng:nx+ng, 1-ng:ny+ng)
    INTEGER, INTENT(IN) :: nx, ny, ng
    INTEGER, INTENT(IN) :: proc_x_min, proc_x_max
    INTEGER, INTENT(IN) :: proc_y_min, proc_y_max
    INTEGER, INTENT(IN) :: comm

    ! Exchange in X direction
    CALL exchange_field_x_gpu(ex, nx, ny, ng, proc_x_min, proc_x_max, comm)
    CALL exchange_field_x_gpu(ey, nx, ny, ng, proc_x_min, proc_x_max, comm)
    CALL exchange_field_x_gpu(ez, nx, ny, ng, proc_x_min, proc_x_max, comm)
    CALL exchange_field_x_gpu(bx, nx, ny, ng, proc_x_min, proc_x_max, comm)
    CALL exchange_field_x_gpu(by, nx, ny, ng, proc_x_min, proc_x_max, comm)
    CALL exchange_field_x_gpu(bz, nx, ny, ng, proc_x_min, proc_x_max, comm)

    ! Exchange in Y direction
    CALL exchange_field_y_gpu(ex, nx, ny, ng, proc_y_min, proc_y_max, comm)
    CALL exchange_field_y_gpu(ey, nx, ny, ng, proc_y_min, proc_y_max, comm)
    CALL exchange_field_y_gpu(ez, nx, ny, ng, proc_y_min, proc_y_max, comm)
    CALL exchange_field_y_gpu(bx, nx, ny, ng, proc_y_min, proc_y_max, comm)
    CALL exchange_field_y_gpu(by, nx, ny, ng, proc_y_min, proc_y_max, comm)
    CALL exchange_field_y_gpu(bz, nx, ny, ng, proc_y_min, proc_y_max, comm)

  END SUBROUTINE exchange_fields_gpu

END MODULE mpi_gpu

