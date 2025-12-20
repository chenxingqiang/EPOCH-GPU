! Copyright (C) 2025 EPOCH-GPU Development Team
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

!> @file particle_data_gpu.F90
!> @brief GPU-optimized Structure of Arrays (SoA) particle data structures
!> @details This module provides GPU-friendly particle data layouts that replace
!>          the traditional linked-list Array of Structures (AoS) approach.
!>          SoA layout enables coalesced memory access on GPU.

MODULE particle_data_gpu

  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT32, INT64, REAL64
  USE gpu_config
  
  IMPLICIT NONE
  PRIVATE
  
  !---------------------------------------------------------------------------
  ! Public types and interfaces
  !---------------------------------------------------------------------------
  PUBLIC :: particle_soa_t
  PUBLIC :: soa_allocate, soa_deallocate, soa_resize
  PUBLIC :: soa_copy_to_device, soa_copy_to_host
  PUBLIC :: soa_get_count, soa_set_count
  PUBLIC :: soa_compact, soa_sort_by_cell
  
  !---------------------------------------------------------------------------
  ! Precision parameters (matching EPOCH's num type)
  !---------------------------------------------------------------------------
  INTEGER, PARAMETER :: num = REAL64
  INTEGER, PARAMETER :: i8 = INT64
  
  !---------------------------------------------------------------------------
  ! Particle Structure of Arrays (SoA) type
  !---------------------------------------------------------------------------
  !> @brief GPU-optimized particle data structure using SoA layout
  !> @details Each particle property is stored in a separate array to enable
  !>          coalesced memory access patterns on GPU. This is critical for
  !>          achieving high memory bandwidth utilization.
  TYPE :: particle_soa_t
    !-----------------------------------------------------------------------
    ! Particle count and capacity
    !-----------------------------------------------------------------------
    INTEGER(i8) :: count = 0           !< Current number of particles
    INTEGER(i8) :: capacity = 0        !< Allocated capacity
    INTEGER :: species_id = 0          !< Species identifier
    
    !-----------------------------------------------------------------------
    ! Position arrays (2D: x, y)
    ! Note: Using standard allocatable arrays with OpenACC data directives
    !       for GPU memory management. CUDA Fortran DEVICE arrays require
    !       separate implementation with .cuf extension.
    !-----------------------------------------------------------------------
    REAL(num), ALLOCATABLE :: pos_x(:)
    REAL(num), ALLOCATABLE :: pos_y(:)
    
    !-----------------------------------------------------------------------
    ! Momentum arrays (3D: px, py, pz)
    !-----------------------------------------------------------------------
    REAL(num), ALLOCATABLE :: px(:)
    REAL(num), ALLOCATABLE :: py(:)
    REAL(num), ALLOCATABLE :: pz(:)
    
    !-----------------------------------------------------------------------
    ! Particle weight
    !-----------------------------------------------------------------------
    REAL(num), ALLOCATABLE :: weight(:)
    
    !-----------------------------------------------------------------------
    ! Cell index (for sorting and cache optimization)
    !-----------------------------------------------------------------------
    INTEGER, ALLOCATABLE :: cell_idx(:)
    
    !-----------------------------------------------------------------------
    ! Boundary flags (0=interior, >0=needs boundary processing)
    !-----------------------------------------------------------------------
    INTEGER, ALLOCATABLE :: boundary_flag(:)
    
    !-----------------------------------------------------------------------
    ! Host-side mirror arrays (for data transfer)
    !-----------------------------------------------------------------------
    REAL(num), ALLOCATABLE :: h_pos_x(:)
    REAL(num), ALLOCATABLE :: h_pos_y(:)
    REAL(num), ALLOCATABLE :: h_px(:)
    REAL(num), ALLOCATABLE :: h_py(:)
    REAL(num), ALLOCATABLE :: h_pz(:)
    REAL(num), ALLOCATABLE :: h_weight(:)
    INTEGER, ALLOCATABLE :: h_cell_idx(:)
    INTEGER, ALLOCATABLE :: h_boundary_flag(:)
    
    !-----------------------------------------------------------------------
    ! Status flags
    !-----------------------------------------------------------------------
    LOGICAL :: is_allocated = .FALSE.
    LOGICAL :: data_on_device = .FALSE.
    
  END TYPE particle_soa_t

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief Allocate SoA particle arrays
  !> @param[inout] soa Particle SoA structure
  !> @param[in] n_particles Number of particles to allocate
  !> @param[in] species_id Species identifier
  !> @param[in] growth_factor Optional capacity growth factor (default: 1.2)
  !---------------------------------------------------------------------------
  SUBROUTINE soa_allocate(soa, n_particles, species_id, growth_factor)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    INTEGER(i8), INTENT(IN) :: n_particles
    INTEGER, INTENT(IN) :: species_id
    REAL(num), INTENT(IN), OPTIONAL :: growth_factor
    
    REAL(num) :: gf
    INTEGER(i8) :: cap
    INTEGER :: ierr
    
    ! Clean up if already allocated
    IF (soa%is_allocated) THEN
      CALL soa_deallocate(soa)
    END IF
    
    ! Set growth factor
    IF (PRESENT(growth_factor)) THEN
      gf = growth_factor
    ELSE
      gf = 1.2_num
    END IF
    
    ! Calculate capacity with growth margin
    cap = INT(REAL(n_particles, num) * gf, i8)
    IF (cap < 1) cap = 1
    
    soa%count = n_particles
    soa%capacity = cap
    soa%species_id = species_id
    
    ! Allocate host arrays
    ALLOCATE(soa%h_pos_x(cap), STAT=ierr)
    ALLOCATE(soa%h_pos_y(cap), STAT=ierr)
    ALLOCATE(soa%h_px(cap), STAT=ierr)
    ALLOCATE(soa%h_py(cap), STAT=ierr)
    ALLOCATE(soa%h_pz(cap), STAT=ierr)
    ALLOCATE(soa%h_weight(cap), STAT=ierr)
    ALLOCATE(soa%h_cell_idx(cap), STAT=ierr)
    ALLOCATE(soa%h_boundary_flag(cap), STAT=ierr)
    
    ! Initialize host arrays
    soa%h_pos_x = 0.0_num
    soa%h_pos_y = 0.0_num
    soa%h_px = 0.0_num
    soa%h_py = 0.0_num
    soa%h_pz = 0.0_num
    soa%h_weight = 0.0_num
    soa%h_cell_idx = 0
    soa%h_boundary_flag = 0
    
    ! Allocate main arrays (used for both CPU and GPU via OpenACC)
    ALLOCATE(soa%pos_x(cap), STAT=ierr)
    ALLOCATE(soa%pos_y(cap), STAT=ierr)
    ALLOCATE(soa%px(cap), STAT=ierr)
    ALLOCATE(soa%py(cap), STAT=ierr)
    ALLOCATE(soa%pz(cap), STAT=ierr)
    ALLOCATE(soa%weight(cap), STAT=ierr)
    ALLOCATE(soa%cell_idx(cap), STAT=ierr)
    ALLOCATE(soa%boundary_flag(cap), STAT=ierr)
    
    soa%pos_x = 0.0_num
    soa%pos_y = 0.0_num
    soa%px = 0.0_num
    soa%py = 0.0_num
    soa%pz = 0.0_num
    soa%weight = 0.0_num
    soa%cell_idx = 0
    soa%boundary_flag = 0
    
    soa%is_allocated = .TRUE.
    soa%data_on_device = .FALSE.
    
  END SUBROUTINE soa_allocate


  !---------------------------------------------------------------------------
  !> @brief Deallocate SoA particle arrays
  !> @param[inout] soa Particle SoA structure
  !---------------------------------------------------------------------------
  SUBROUTINE soa_deallocate(soa)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    
    INTEGER :: ierr
    
    ! Deallocate host arrays
    IF (ALLOCATED(soa%h_pos_x)) DEALLOCATE(soa%h_pos_x, STAT=ierr)
    IF (ALLOCATED(soa%h_pos_y)) DEALLOCATE(soa%h_pos_y, STAT=ierr)
    IF (ALLOCATED(soa%h_px)) DEALLOCATE(soa%h_px, STAT=ierr)
    IF (ALLOCATED(soa%h_py)) DEALLOCATE(soa%h_py, STAT=ierr)
    IF (ALLOCATED(soa%h_pz)) DEALLOCATE(soa%h_pz, STAT=ierr)
    IF (ALLOCATED(soa%h_weight)) DEALLOCATE(soa%h_weight, STAT=ierr)
    IF (ALLOCATED(soa%h_cell_idx)) DEALLOCATE(soa%h_cell_idx, STAT=ierr)
    IF (ALLOCATED(soa%h_boundary_flag)) DEALLOCATE(soa%h_boundary_flag, STAT=ierr)
    
    ! Deallocate device/main arrays
    IF (ALLOCATED(soa%pos_x)) DEALLOCATE(soa%pos_x, STAT=ierr)
    IF (ALLOCATED(soa%pos_y)) DEALLOCATE(soa%pos_y, STAT=ierr)
    IF (ALLOCATED(soa%px)) DEALLOCATE(soa%px, STAT=ierr)
    IF (ALLOCATED(soa%py)) DEALLOCATE(soa%py, STAT=ierr)
    IF (ALLOCATED(soa%pz)) DEALLOCATE(soa%pz, STAT=ierr)
    IF (ALLOCATED(soa%weight)) DEALLOCATE(soa%weight, STAT=ierr)
    IF (ALLOCATED(soa%cell_idx)) DEALLOCATE(soa%cell_idx, STAT=ierr)
    IF (ALLOCATED(soa%boundary_flag)) DEALLOCATE(soa%boundary_flag, STAT=ierr)
    
    soa%count = 0
    soa%capacity = 0
    soa%species_id = 0
    soa%is_allocated = .FALSE.
    soa%data_on_device = .FALSE.
    
  END SUBROUTINE soa_deallocate


  !---------------------------------------------------------------------------
  !> @brief Resize SoA arrays while preserving data
  !> @param[inout] soa Particle SoA structure
  !> @param[in] new_capacity New capacity
  !---------------------------------------------------------------------------
  SUBROUTINE soa_resize(soa, new_capacity)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    INTEGER(i8), INTENT(IN) :: new_capacity
    
    REAL(num), ALLOCATABLE :: temp_real(:)
    INTEGER, ALLOCATABLE :: temp_int(:)
    INTEGER(i8) :: copy_count
    INTEGER :: ierr
    
    IF (.NOT. soa%is_allocated) RETURN
    IF (new_capacity <= soa%capacity) RETURN
    
    copy_count = MIN(soa%count, new_capacity)
    
    ! Resize host arrays
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%h_pos_x(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%h_pos_x)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%h_pos_y(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%h_pos_y)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%h_px(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%h_px)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%h_py(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%h_py)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%h_pz(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%h_pz)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%h_weight(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%h_weight)
    
    ALLOCATE(temp_int(new_capacity), STAT=ierr)
    temp_int(1:copy_count) = soa%h_cell_idx(1:copy_count)
    CALL MOVE_ALLOC(temp_int, soa%h_cell_idx)
    
    ALLOCATE(temp_int(new_capacity), STAT=ierr)
    temp_int(1:copy_count) = soa%h_boundary_flag(1:copy_count)
    CALL MOVE_ALLOC(temp_int, soa%h_boundary_flag)
    
    ! Resize main arrays
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%pos_x(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%pos_x)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%pos_y(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%pos_y)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%px(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%px)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%py(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%py)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%pz(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%pz)
    
    ALLOCATE(temp_real(new_capacity), STAT=ierr)
    temp_real(1:copy_count) = soa%weight(1:copy_count)
    CALL MOVE_ALLOC(temp_real, soa%weight)
    
    ALLOCATE(temp_int(new_capacity), STAT=ierr)
    temp_int(1:copy_count) = soa%cell_idx(1:copy_count)
    CALL MOVE_ALLOC(temp_int, soa%cell_idx)
    
    ALLOCATE(temp_int(new_capacity), STAT=ierr)
    temp_int(1:copy_count) = soa%boundary_flag(1:copy_count)
    CALL MOVE_ALLOC(temp_int, soa%boundary_flag)
    
    ! Mark data as no longer on device (needs re-upload after resize)
    soa%data_on_device = .FALSE.
    
    soa%capacity = new_capacity
    
  END SUBROUTINE soa_resize


  !---------------------------------------------------------------------------
  !> @brief Copy particle data from host to device using OpenACC
  !> @param[inout] soa Particle SoA structure
  !---------------------------------------------------------------------------
  SUBROUTINE soa_copy_to_device(soa)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    
    INTEGER(i8) :: n
    
    IF (.NOT. soa%is_allocated) RETURN
    IF (.NOT. gpu_enabled) RETURN
    
    n = soa%count
    
    ! Copy from host mirror arrays to main arrays
    soa%pos_x(1:n) = soa%h_pos_x(1:n)
    soa%pos_y(1:n) = soa%h_pos_y(1:n)
    soa%px(1:n) = soa%h_px(1:n)
    soa%py(1:n) = soa%h_py(1:n)
    soa%pz(1:n) = soa%h_pz(1:n)
    soa%weight(1:n) = soa%h_weight(1:n)
    soa%cell_idx(1:n) = soa%h_cell_idx(1:n)
    soa%boundary_flag(1:n) = soa%h_boundary_flag(1:n)
    
    ! OpenACC: update device with main arrays
    !$ACC UPDATE DEVICE(soa%pos_x(1:n), soa%pos_y(1:n)) &
    !$ACC   DEVICE(soa%px(1:n), soa%py(1:n), soa%pz(1:n)) &
    !$ACC   DEVICE(soa%weight(1:n), soa%cell_idx(1:n), soa%boundary_flag(1:n))
    
    soa%data_on_device = .TRUE.
    
  END SUBROUTINE soa_copy_to_device


  !---------------------------------------------------------------------------
  !> @brief Copy particle data from device to host using OpenACC
  !> @param[inout] soa Particle SoA structure
  !---------------------------------------------------------------------------
  SUBROUTINE soa_copy_to_host(soa)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    
    INTEGER(i8) :: n
    
    IF (.NOT. soa%is_allocated) RETURN
    IF (.NOT. gpu_enabled) RETURN
    IF (.NOT. soa%data_on_device) RETURN
    
    n = soa%count
    
    ! OpenACC: update host with device data
    !$ACC UPDATE HOST(soa%pos_x(1:n), soa%pos_y(1:n)) &
    !$ACC   HOST(soa%px(1:n), soa%py(1:n), soa%pz(1:n)) &
    !$ACC   HOST(soa%weight(1:n), soa%cell_idx(1:n), soa%boundary_flag(1:n))
    
    ! Copy to host mirror arrays
    soa%h_pos_x(1:n) = soa%pos_x(1:n)
    soa%h_pos_y(1:n) = soa%pos_y(1:n)
    soa%h_px(1:n) = soa%px(1:n)
    soa%h_py(1:n) = soa%py(1:n)
    soa%h_pz(1:n) = soa%pz(1:n)
    soa%h_weight(1:n) = soa%weight(1:n)
    soa%h_cell_idx(1:n) = soa%cell_idx(1:n)
    soa%h_boundary_flag(1:n) = soa%boundary_flag(1:n)
    
  END SUBROUTINE soa_copy_to_host


  !---------------------------------------------------------------------------
  !> @brief Get particle count
  !---------------------------------------------------------------------------
  FUNCTION soa_get_count(soa) RESULT(count)
    TYPE(particle_soa_t), INTENT(IN) :: soa
    INTEGER(i8) :: count
    
    count = soa%count
    
  END FUNCTION soa_get_count


  !---------------------------------------------------------------------------
  !> @brief Set particle count
  !---------------------------------------------------------------------------
  SUBROUTINE soa_set_count(soa, count)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    INTEGER(i8), INTENT(IN) :: count
    
    IF (count <= soa%capacity) THEN
      soa%count = count
    ELSE
      CALL soa_resize(soa, INT(REAL(count, num) * 1.5_num, i8))
      soa%count = count
    END IF
    
  END SUBROUTINE soa_set_count


  !---------------------------------------------------------------------------
  !> @brief Compact particles (remove flagged particles)
  !> @param[inout] soa Particle SoA structure
  !> @details Removes particles with boundary_flag < 0 (deleted particles)
  !---------------------------------------------------------------------------
  SUBROUTINE soa_compact(soa)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    
    INTEGER(i8) :: i, j, n
    
    IF (.NOT. soa%is_allocated) RETURN
    
    IF (gpu_enabled .AND. soa%data_on_device) THEN
      CALL soa_copy_to_host(soa)
    END IF
    
    n = soa%count
    j = 0
    
    DO i = 1, n
      IF (soa%h_boundary_flag(i) >= 0) THEN
        j = j + 1
        IF (j /= i) THEN
          soa%h_pos_x(j) = soa%h_pos_x(i)
          soa%h_pos_y(j) = soa%h_pos_y(i)
          soa%h_px(j) = soa%h_px(i)
          soa%h_py(j) = soa%h_py(i)
          soa%h_pz(j) = soa%h_pz(i)
          soa%h_weight(j) = soa%h_weight(i)
          soa%h_cell_idx(j) = soa%h_cell_idx(i)
          soa%h_boundary_flag(j) = soa%h_boundary_flag(i)
        END IF
      END IF
    END DO
    
    soa%count = j
    
    ! Update main arrays from host arrays
    soa%pos_x(1:j) = soa%h_pos_x(1:j)
    soa%pos_y(1:j) = soa%h_pos_y(1:j)
    soa%px(1:j) = soa%h_px(1:j)
    soa%py(1:j) = soa%h_py(1:j)
    soa%pz(1:j) = soa%h_pz(1:j)
    soa%weight(1:j) = soa%h_weight(1:j)
    soa%cell_idx(1:j) = soa%h_cell_idx(1:j)
    soa%boundary_flag(1:j) = soa%h_boundary_flag(1:j)
    
    soa%data_on_device = .FALSE.
    
  END SUBROUTINE soa_compact


  !---------------------------------------------------------------------------
  !> @brief Sort particles by cell index for cache optimization
  !> @param[inout] soa Particle SoA structure
  !> @details Uses counting sort for efficiency with bounded cell indices
  !---------------------------------------------------------------------------
  SUBROUTINE soa_sort_by_cell(soa, max_cell_idx)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    INTEGER, INTENT(IN) :: max_cell_idx
    
    INTEGER, ALLOCATABLE :: count_arr(:), offset(:)
    INTEGER(i8), ALLOCATABLE :: sorted_idx(:)
    REAL(num), ALLOCATABLE :: temp_real(:)
    INTEGER, ALLOCATABLE :: temp_int(:)
    INTEGER(i8) :: i, n, idx
    INTEGER :: cell
    
    IF (.NOT. soa%is_allocated) RETURN
    
    IF (gpu_enabled .AND. soa%data_on_device) THEN
      CALL soa_copy_to_host(soa)
    END IF
    
    n = soa%count
    IF (n <= 1) RETURN
    
    ! Counting sort
    ALLOCATE(count_arr(0:max_cell_idx))
    ALLOCATE(offset(0:max_cell_idx))
    ALLOCATE(sorted_idx(n))
    
    count_arr = 0
    
    ! Count particles per cell
    DO i = 1, n
      cell = soa%h_cell_idx(i)
      IF (cell >= 0 .AND. cell <= max_cell_idx) THEN
        count_arr(cell) = count_arr(cell) + 1
      END IF
    END DO
    
    ! Compute offsets
    offset(0) = 1
    DO cell = 1, max_cell_idx
      offset(cell) = offset(cell-1) + count_arr(cell-1)
    END DO
    
    ! Place particles
    count_arr = 0
    DO i = 1, n
      cell = soa%h_cell_idx(i)
      IF (cell >= 0 .AND. cell <= max_cell_idx) THEN
        idx = offset(cell) + count_arr(cell)
        sorted_idx(idx) = i
        count_arr(cell) = count_arr(cell) + 1
      END IF
    END DO
    
    ! Apply permutation to all arrays
    ALLOCATE(temp_real(n))
    ALLOCATE(temp_int(n))
    
    DO i = 1, n
      temp_real(i) = soa%h_pos_x(sorted_idx(i))
    END DO
    soa%h_pos_x(1:n) = temp_real
    
    DO i = 1, n
      temp_real(i) = soa%h_pos_y(sorted_idx(i))
    END DO
    soa%h_pos_y(1:n) = temp_real
    
    DO i = 1, n
      temp_real(i) = soa%h_px(sorted_idx(i))
    END DO
    soa%h_px(1:n) = temp_real
    
    DO i = 1, n
      temp_real(i) = soa%h_py(sorted_idx(i))
    END DO
    soa%h_py(1:n) = temp_real
    
    DO i = 1, n
      temp_real(i) = soa%h_pz(sorted_idx(i))
    END DO
    soa%h_pz(1:n) = temp_real
    
    DO i = 1, n
      temp_real(i) = soa%h_weight(sorted_idx(i))
    END DO
    soa%h_weight(1:n) = temp_real
    
    DO i = 1, n
      temp_int(i) = soa%h_cell_idx(sorted_idx(i))
    END DO
    soa%h_cell_idx(1:n) = temp_int
    
    DO i = 1, n
      temp_int(i) = soa%h_boundary_flag(sorted_idx(i))
    END DO
    soa%h_boundary_flag(1:n) = temp_int
    
    DEALLOCATE(count_arr, offset, sorted_idx, temp_real, temp_int)
    
    ! Update main arrays from host arrays
    soa%pos_x(1:n) = soa%h_pos_x(1:n)
    soa%pos_y(1:n) = soa%h_pos_y(1:n)
    soa%px(1:n) = soa%h_px(1:n)
    soa%py(1:n) = soa%h_py(1:n)
    soa%pz(1:n) = soa%h_pz(1:n)
    soa%weight(1:n) = soa%h_weight(1:n)
    soa%cell_idx(1:n) = soa%h_cell_idx(1:n)
    soa%boundary_flag(1:n) = soa%h_boundary_flag(1:n)
    
    soa%data_on_device = .FALSE.
    
  END SUBROUTINE soa_sort_by_cell

END MODULE particle_data_gpu

