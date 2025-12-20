! Copyright (C) 2025 EPOCH-GPU Development Team
!
! @file particle_data_gpu.F90
! @brief GPU-optimized particle data structures for EPOCH1D
! @details Implements Structure of Arrays (SoA) layout for GPU efficiency

MODULE particle_data_gpu

#ifdef GPU_ENABLED
  USE cudafor
#endif
  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT32, INT64, REAL64
  
  IMPLICIT NONE
  PRIVATE
  
  INTEGER, PARAMETER :: num = REAL64
  INTEGER, PARAMETER :: i8 = INT64
  
  PUBLIC :: particle_soa_1d
  PUBLIC :: allocate_particle_soa_1d, deallocate_particle_soa_1d
  PUBLIC :: copy_particles_to_device_1d, copy_particles_to_host_1d
  
  !---------------------------------------------------------------------------
  !> @brief Structure of Arrays for 1D particle data
  !---------------------------------------------------------------------------
  TYPE :: particle_soa_1d
    ! Positions (1D)
    REAL(num), ALLOCATABLE :: pos_x(:)
    
    ! Momenta (3 components even in 1D for full 3V)
    REAL(num), ALLOCATABLE :: px(:)
    REAL(num), ALLOCATABLE :: py(:)
    REAL(num), ALLOCATABLE :: pz(:)
    
    ! Particle weight
    REAL(num), ALLOCATABLE :: weight(:)
    
    ! Cell index for sorting
    INTEGER, ALLOCATABLE :: cell_idx(:)
    
    ! Boundary flags
    INTEGER, ALLOCATABLE :: boundary_flag(:)
    
    ! Particle count and capacity
    INTEGER(i8) :: count = 0
    INTEGER(i8) :: capacity = 0
    
    ! Species ID
    INTEGER :: species_id = 0
    
  END TYPE particle_soa_1d

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief Allocate SoA particle arrays for 1D
  !---------------------------------------------------------------------------
  SUBROUTINE allocate_particle_soa_1d(soa, n_particles, species_id)
    TYPE(particle_soa_1d), INTENT(INOUT) :: soa
    INTEGER(i8), INTENT(IN) :: n_particles
    INTEGER, INTENT(IN) :: species_id
    
    INTEGER(i8) :: n
    
    ! Allocate with 20% buffer for particle creation
    n = INT(REAL(n_particles, num) * 1.2_num, i8)
    IF (n < 1000) n = 1000
    
    soa%capacity = n
    soa%count = n_particles
    soa%species_id = species_id
    
    ! Allocate arrays
    ALLOCATE(soa%pos_x(n))
    ALLOCATE(soa%px(n))
    ALLOCATE(soa%py(n))
    ALLOCATE(soa%pz(n))
    ALLOCATE(soa%weight(n))
    ALLOCATE(soa%cell_idx(n))
    ALLOCATE(soa%boundary_flag(n))
    
    ! Initialize
    soa%pos_x = 0.0_num
    soa%px = 0.0_num
    soa%py = 0.0_num
    soa%pz = 0.0_num
    soa%weight = 0.0_num
    soa%cell_idx = 0
    soa%boundary_flag = 0
    
  END SUBROUTINE allocate_particle_soa_1d


  !---------------------------------------------------------------------------
  !> @brief Deallocate SoA particle arrays
  !---------------------------------------------------------------------------
  SUBROUTINE deallocate_particle_soa_1d(soa)
    TYPE(particle_soa_1d), INTENT(INOUT) :: soa
    
#ifdef GPU_ENABLED
    !$ACC EXIT DATA DELETE(soa%pos_x, soa%px, soa%py, soa%pz) IF(ALLOCATED(soa%pos_x))
    !$ACC EXIT DATA DELETE(soa%weight, soa%cell_idx, soa%boundary_flag) IF(ALLOCATED(soa%weight))
#endif
    
    IF (ALLOCATED(soa%pos_x)) DEALLOCATE(soa%pos_x)
    IF (ALLOCATED(soa%px)) DEALLOCATE(soa%px)
    IF (ALLOCATED(soa%py)) DEALLOCATE(soa%py)
    IF (ALLOCATED(soa%pz)) DEALLOCATE(soa%pz)
    IF (ALLOCATED(soa%weight)) DEALLOCATE(soa%weight)
    IF (ALLOCATED(soa%cell_idx)) DEALLOCATE(soa%cell_idx)
    IF (ALLOCATED(soa%boundary_flag)) DEALLOCATE(soa%boundary_flag)
    
    soa%count = 0
    soa%capacity = 0
    
  END SUBROUTINE deallocate_particle_soa_1d


  !---------------------------------------------------------------------------
  !> @brief Copy particle data to GPU device
  !---------------------------------------------------------------------------
  SUBROUTINE copy_particles_to_device_1d(soa)
    TYPE(particle_soa_1d), INTENT(INOUT) :: soa
    
    INTEGER(i8) :: n
    
    n = soa%count
    
#ifdef GPU_ENABLED
    !$ACC ENTER DATA COPYIN(soa%pos_x(1:n))
    !$ACC ENTER DATA COPYIN(soa%px(1:n), soa%py(1:n), soa%pz(1:n))
    !$ACC ENTER DATA COPYIN(soa%weight(1:n))
    !$ACC ENTER DATA COPYIN(soa%cell_idx(1:n), soa%boundary_flag(1:n))
#endif
    
  END SUBROUTINE copy_particles_to_device_1d


  !---------------------------------------------------------------------------
  !> @brief Copy particle data from GPU device to host
  !---------------------------------------------------------------------------
  SUBROUTINE copy_particles_to_host_1d(soa)
    TYPE(particle_soa_1d), INTENT(INOUT) :: soa
    
    INTEGER(i8) :: n
    
    n = soa%count
    
#ifdef GPU_ENABLED
    !$ACC UPDATE SELF(soa%pos_x(1:n))
    !$ACC UPDATE SELF(soa%px(1:n), soa%py(1:n), soa%pz(1:n))
    !$ACC UPDATE SELF(soa%weight(1:n))
    !$ACC UPDATE SELF(soa%cell_idx(1:n), soa%boundary_flag(1:n))
#endif
    
  END SUBROUTINE copy_particles_to_host_1d

END MODULE particle_data_gpu

