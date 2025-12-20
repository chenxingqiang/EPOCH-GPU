! Copyright (C) 2024 EPOCH-GPU Development Team
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.

!> @file gpu_config.F90
!> @brief GPU configuration and device management module
!> @details This module provides GPU device initialization, configuration,
!>          and runtime management for EPOCH-GPU

MODULE gpu_config

#ifdef GPU_ENABLED
  USE cudafor
#endif
  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT32, INT64, REAL64
  
  IMPLICIT NONE
  PRIVATE
  
  !---------------------------------------------------------------------------
  ! Public interfaces
  !---------------------------------------------------------------------------
  PUBLIC :: gpu_init, gpu_finalize, gpu_sync
  PUBLIC :: gpu_get_device_count, gpu_set_device, gpu_get_device
  PUBLIC :: gpu_get_memory_info, gpu_print_device_info
  PUBLIC :: gpu_enabled, gpu_device_id
  PUBLIC :: BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_PARTICLE
  ! Wrapper functions for epoch2d integration
  PUBLIC :: gpu_init_all, gpu_finalize_all, gpu_check_available
  
  !---------------------------------------------------------------------------
  ! GPU configuration parameters
  !---------------------------------------------------------------------------
  
  !> Enable/disable GPU acceleration at runtime
  LOGICAL, SAVE :: gpu_enabled = .FALSE.
  
  !> Current GPU device ID (0-based)
  INTEGER, SAVE :: gpu_device_id = 0
  
  !> Number of available GPU devices
  INTEGER, SAVE :: gpu_device_count = 0
  
  !> GPU compute capability (major.minor)
  INTEGER, SAVE :: gpu_compute_major = 0
  INTEGER, SAVE :: gpu_compute_minor = 0
  
  !> GPU memory information (bytes)
  INTEGER(INT64), SAVE :: gpu_total_memory = 0
  INTEGER(INT64), SAVE :: gpu_free_memory = 0
  
  !> GPU name
  CHARACTER(LEN=256), SAVE :: gpu_device_name = ''
  
  !---------------------------------------------------------------------------
  ! CUDA kernel launch configuration
  !---------------------------------------------------------------------------
  
  !> Thread block size for field operations (2D)
  INTEGER, PARAMETER :: BLOCK_SIZE_X = 16
  INTEGER, PARAMETER :: BLOCK_SIZE_Y = 16
  
  !> Thread block size for particle operations (1D)
  INTEGER, PARAMETER :: BLOCK_SIZE_PARTICLE = 256
  
  !> Maximum threads per block
  INTEGER, PARAMETER :: MAX_THREADS_PER_BLOCK = 1024
  
  !> Warp size (NVIDIA GPUs)
  INTEGER, PARAMETER :: WARP_SIZE = 32

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief Initialize GPU subsystem
  !> @param[in] device_id Optional device ID to use (default: 0)
  !> @return Error code (0 = success)
  !---------------------------------------------------------------------------
  FUNCTION gpu_init(device_id) RESULT(ierr)
    INTEGER, INTENT(IN), OPTIONAL :: device_id
    INTEGER :: ierr
    
#ifdef GPU_ENABLED
    TYPE(cudaDeviceProp) :: prop
    INTEGER :: requested_device
    
    ierr = 0
    
    ! Get device count
    ierr = cudaGetDeviceCount(gpu_device_count)
    IF (ierr /= 0 .OR. gpu_device_count == 0) THEN
      gpu_enabled = .FALSE.
      WRITE(*,*) 'Warning: No CUDA-capable GPU found. Running in CPU mode.'
      RETURN
    END IF
    
    ! Set requested device
    IF (PRESENT(device_id)) THEN
      requested_device = device_id
    ELSE
      requested_device = 0
    END IF
    
    ! Validate device ID
    IF (requested_device < 0 .OR. requested_device >= gpu_device_count) THEN
      WRITE(*,*) 'Error: Invalid GPU device ID:', requested_device
      WRITE(*,*) 'Available devices: 0 to', gpu_device_count - 1
      ierr = -1
      RETURN
    END IF
    
    ! Set device
    ierr = cudaSetDevice(requested_device)
    IF (ierr /= 0) THEN
      WRITE(*,*) 'Error: Failed to set GPU device:', requested_device
      RETURN
    END IF
    gpu_device_id = requested_device
    
    ! Get device properties
    ierr = cudaGetDeviceProperties(prop, gpu_device_id)
    IF (ierr /= 0) THEN
      WRITE(*,*) 'Warning: Failed to get GPU device properties'
    ELSE
      gpu_device_name = TRIM(prop%name)
      gpu_compute_major = prop%major
      gpu_compute_minor = prop%minor
      gpu_total_memory = prop%totalGlobalMem
    END IF
    
    ! Get memory info
    ierr = cudaMemGetInfo(gpu_free_memory, gpu_total_memory)
    
    gpu_enabled = .TRUE.
    ierr = 0
    
#else
    ! GPU support not compiled in
    gpu_enabled = .FALSE.
    gpu_device_count = 0
    ierr = 0
    WRITE(*,*) 'Note: GPU support not enabled at compile time.'
#endif
    
  END FUNCTION gpu_init


  !---------------------------------------------------------------------------
  !> @brief Finalize GPU subsystem
  !---------------------------------------------------------------------------
  SUBROUTINE gpu_finalize()
    
#ifdef GPU_ENABLED
    INTEGER :: ierr
    
    IF (gpu_enabled) THEN
      ierr = cudaDeviceReset()
      gpu_enabled = .FALSE.
    END IF
#endif
    
  END SUBROUTINE gpu_finalize


  !---------------------------------------------------------------------------
  !> @brief Synchronize GPU device
  !---------------------------------------------------------------------------
  SUBROUTINE gpu_sync()
    
#ifdef GPU_ENABLED
    INTEGER :: ierr
    
    IF (gpu_enabled) THEN
      ierr = cudaDeviceSynchronize()
    END IF
#endif
    
  END SUBROUTINE gpu_sync


  !---------------------------------------------------------------------------
  !> @brief Get number of available GPU devices
  !---------------------------------------------------------------------------
  FUNCTION gpu_get_device_count() RESULT(count)
    INTEGER :: count
    
    count = gpu_device_count
    
  END FUNCTION gpu_get_device_count


  !---------------------------------------------------------------------------
  !> @brief Set active GPU device
  !---------------------------------------------------------------------------
  FUNCTION gpu_set_device(device_id) RESULT(ierr)
    INTEGER, INTENT(IN) :: device_id
    INTEGER :: ierr
    
#ifdef GPU_ENABLED
    IF (device_id >= 0 .AND. device_id < gpu_device_count) THEN
      ierr = cudaSetDevice(device_id)
      IF (ierr == 0) gpu_device_id = device_id
    ELSE
      ierr = -1
    END IF
#else
    ierr = -1
#endif
    
  END FUNCTION gpu_set_device


  !---------------------------------------------------------------------------
  !> @brief Get current GPU device ID
  !---------------------------------------------------------------------------
  FUNCTION gpu_get_device() RESULT(device_id)
    INTEGER :: device_id
    
    device_id = gpu_device_id
    
  END FUNCTION gpu_get_device


  !---------------------------------------------------------------------------
  !> @brief Get GPU memory information
  !> @param[out] free_mem Free memory in bytes
  !> @param[out] total_mem Total memory in bytes
  !---------------------------------------------------------------------------
  SUBROUTINE gpu_get_memory_info(free_mem, total_mem)
    INTEGER(INT64), INTENT(OUT) :: free_mem, total_mem
    
#ifdef GPU_ENABLED
    INTEGER :: ierr
    
    IF (gpu_enabled) THEN
      ierr = cudaMemGetInfo(free_mem, total_mem)
    ELSE
      free_mem = 0
      total_mem = 0
    END IF
#else
    free_mem = 0
    total_mem = 0
#endif
    
  END SUBROUTINE gpu_get_memory_info


  !---------------------------------------------------------------------------
  !> @brief Print GPU device information
  !---------------------------------------------------------------------------
  SUBROUTINE gpu_print_device_info()
    
    WRITE(*,*)
    WRITE(*,*) '============================================'
    WRITE(*,*) '           GPU Configuration'
    WRITE(*,*) '============================================'
    
    IF (gpu_enabled) THEN
      WRITE(*,*) 'GPU Acceleration: ENABLED'
      WRITE(*,*) 'Device Count:    ', gpu_device_count
      WRITE(*,*) 'Active Device:   ', gpu_device_id
      WRITE(*,*) 'Device Name:     ', TRIM(gpu_device_name)
      WRITE(*,'(A,I1,A,I1)') '  Compute Capability: ', &
          gpu_compute_major, '.', gpu_compute_minor
      WRITE(*,'(A,F8.2,A)') '  Total Memory:    ', &
          REAL(gpu_total_memory) / (1024.0**3), ' GB'
      WRITE(*,'(A,F8.2,A)') '  Free Memory:     ', &
          REAL(gpu_free_memory) / (1024.0**3), ' GB'
    ELSE
      WRITE(*,*) 'GPU Acceleration: DISABLED'
#ifdef GPU_ENABLED
      WRITE(*,*) 'Reason: No CUDA-capable device found'
#else
      WRITE(*,*) 'Reason: Not compiled with GPU support'
#endif
    END IF
    
    WRITE(*,*) '============================================'
    WRITE(*,*)
    
  END SUBROUTINE gpu_print_device_info


  !---------------------------------------------------------------------------
  !> @brief Initialize all GPU resources (wrapper for epoch2d integration)
  !---------------------------------------------------------------------------
  SUBROUTINE gpu_init_all()
    INTEGER :: ierr
    
    ierr = gpu_init(0)
    
  END SUBROUTINE gpu_init_all
  
  
  !---------------------------------------------------------------------------
  !> @brief Finalize all GPU resources (wrapper for epoch2d integration)
  !---------------------------------------------------------------------------
  SUBROUTINE gpu_finalize_all()
    
    CALL gpu_finalize()
    
  END SUBROUTINE gpu_finalize_all
  
  
  !---------------------------------------------------------------------------
  !> @brief Check if GPU is available and enabled
  !> @return .TRUE. if GPU is available and enabled
  !---------------------------------------------------------------------------
  FUNCTION gpu_check_available() RESULT(available)
    LOGICAL :: available
    
    available = gpu_enabled .AND. (gpu_device_count > 0)
    
  END FUNCTION gpu_check_available

END MODULE gpu_config

