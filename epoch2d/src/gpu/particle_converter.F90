!> @file particle_converter.F90
!> @brief Particle data structure converter between CPU (AoS) and GPU (SoA)
!>
!> This module provides routines to convert particle data between EPOCH's
!> native linked-list Array of Structures (AoS) format and the GPU-optimized
!> Structure of Arrays (SoA) format.
!>
!> @author EPOCH-GPU Development Team
!> @date December 2024

MODULE particle_converter
  !> @brief Particle data structure converter between CPU (AoS) and GPU (SoA)
  !>
  !> NOTE: This module requires shared_data.mod which is only available
  !> when compiled as part of the full EPOCH build.
  
  USE shared_data
  USE particle_data_gpu

  IMPLICIT NONE

  PRIVATE
  
  PUBLIC :: convert_species_to_soa
  PUBLIC :: convert_soa_to_species
  PUBLIC :: sync_species_to_gpu
  PUBLIC :: sync_species_from_gpu
  PUBLIC :: allocate_soa_for_species
  PUBLIC :: deallocate_soa

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief Allocate SoA arrays for a species based on particle count
  !>
  !> @param[inout] soa       The SoA structure to allocate
  !> @param[in]    n_particles Number of particles to allocate for
  !> @param[in]    species_id Species identifier
  !---------------------------------------------------------------------------
  SUBROUTINE allocate_soa_for_species(soa, n_particles, species_id)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    INTEGER(i8), INTENT(IN) :: n_particles
    INTEGER, INTENT(IN) :: species_id
    
    INTEGER(i8) :: capacity
    
    ! Add 20% buffer for particle creation/injection
    capacity = INT(REAL(n_particles, num) * 1.2_num, i8)
    IF (capacity < 1000) capacity = 1000
    
    ! Deallocate if already allocated with different size
    IF (ALLOCATED(soa%pos_x)) THEN
      IF (soa%capacity < n_particles) THEN
        CALL deallocate_soa(soa)
      ELSE
        ! Already have enough capacity
        soa%count = n_particles
        soa%species_id = species_id
        RETURN
      END IF
    END IF
    
    ! Allocate arrays
    ALLOCATE(soa%pos_x(capacity))
    ALLOCATE(soa%pos_y(capacity))
    ALLOCATE(soa%px(capacity))
    ALLOCATE(soa%py(capacity))
    ALLOCATE(soa%pz(capacity))
    ALLOCATE(soa%weight(capacity))
    ALLOCATE(soa%cell_idx(capacity))
    ALLOCATE(soa%boundary_flag(capacity))
    
    ! Initialize metadata
    soa%count = n_particles
    soa%capacity = capacity
    soa%species_id = species_id
    
    ! Initialize arrays to zero
    soa%pos_x = 0.0_num
    soa%pos_y = 0.0_num
    soa%px = 0.0_num
    soa%py = 0.0_num
    soa%pz = 0.0_num
    soa%weight = 0.0_num
    soa%cell_idx = 0
    soa%boundary_flag = 0
    
  END SUBROUTINE allocate_soa_for_species


  !---------------------------------------------------------------------------
  !> @brief Deallocate SoA arrays
  !>
  !> @param[inout] soa The SoA structure to deallocate
  !---------------------------------------------------------------------------
  SUBROUTINE deallocate_soa(soa)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    
    IF (ALLOCATED(soa%pos_x)) DEALLOCATE(soa%pos_x)
    IF (ALLOCATED(soa%pos_y)) DEALLOCATE(soa%pos_y)
    IF (ALLOCATED(soa%px)) DEALLOCATE(soa%px)
    IF (ALLOCATED(soa%py)) DEALLOCATE(soa%py)
    IF (ALLOCATED(soa%pz)) DEALLOCATE(soa%pz)
    IF (ALLOCATED(soa%weight)) DEALLOCATE(soa%weight)
    IF (ALLOCATED(soa%cell_idx)) DEALLOCATE(soa%cell_idx)
    IF (ALLOCATED(soa%boundary_flag)) DEALLOCATE(soa%boundary_flag)
    
    soa%count = 0
    soa%capacity = 0
    soa%species_id = 0
    
  END SUBROUTINE deallocate_soa


  !---------------------------------------------------------------------------
  !> @brief Convert a particle species from AoS (linked list) to SoA format
  !>
  !> Traverses the linked list of particles and copies data to contiguous
  !> arrays optimized for GPU access.
  !>
  !> @param[in]    species   The particle species (with attached_list)
  !> @param[inout] soa       The SoA structure to fill
  !---------------------------------------------------------------------------
  SUBROUTINE convert_species_to_soa(species, soa)
    TYPE(particle_species), INTENT(IN) :: species
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    
    TYPE(particle), POINTER :: current
    INTEGER(i8) :: ipart, n_particles
    
    ! Get particle count from the species
    n_particles = species%attached_list%count
    
    IF (n_particles <= 0) THEN
      soa%count = 0
      RETURN
    END IF
    
    ! Allocate SoA if needed
    CALL allocate_soa_for_species(soa, n_particles, species%id)
    
    ! Traverse linked list and copy to SoA
    current => species%attached_list%head
    ipart = 0
    
    DO WHILE (ASSOCIATED(current))
      ipart = ipart + 1
      
      IF (ipart > soa%capacity) EXIT
      
      ! Copy position (2D)
      soa%pos_x(ipart) = current%part_pos(1)
      soa%pos_y(ipart) = current%part_pos(2)
      
      ! Copy momentum (3D)
      soa%px(ipart) = current%part_p(1)
      soa%py(ipart) = current%part_p(2)
      soa%pz(ipart) = current%part_p(3)
      
      ! Copy weight
#if !defined(PER_SPECIES_WEIGHT)
      soa%weight(ipart) = current%weight
#else
      soa%weight(ipart) = species%weight
#endif
      
      ! Initialize cell index and boundary flag
      soa%cell_idx(ipart) = 0
      soa%boundary_flag(ipart) = 0
      
      current => current%next
    END DO
    
    soa%count = ipart
    
  END SUBROUTINE convert_species_to_soa


  !---------------------------------------------------------------------------
  !> @brief Convert SoA data back to AoS (linked list) format
  !>
  !> Updates the particle data in the linked list from the SoA arrays.
  !> Assumes the linked list structure is unchanged (same particles).
  !>
  !> @param[in]    soa       The SoA structure with updated data
  !> @param[inout] species   The particle species to update
  !---------------------------------------------------------------------------
  SUBROUTINE convert_soa_to_species(soa, species)
    TYPE(particle_soa_t), INTENT(IN) :: soa
    TYPE(particle_species), INTENT(INOUT) :: species
    
    TYPE(particle), POINTER :: current
    INTEGER(i8) :: ipart
    
    IF (soa%count <= 0) RETURN
    
    ! Traverse linked list and copy from SoA
    current => species%attached_list%head
    ipart = 0
    
    DO WHILE (ASSOCIATED(current))
      ipart = ipart + 1
      
      IF (ipart > soa%count) EXIT
      
      ! Copy position back (2D)
      current%part_pos(1) = soa%pos_x(ipart)
      current%part_pos(2) = soa%pos_y(ipart)
      
      ! Copy momentum back (3D)
      current%part_p(1) = soa%px(ipart)
      current%part_p(2) = soa%py(ipart)
      current%part_p(3) = soa%pz(ipart)
      
      ! Copy weight back (only if per-particle weight)
#if !defined(PER_SPECIES_WEIGHT)
      current%weight = soa%weight(ipart)
#endif
      
      current => current%next
    END DO
    
  END SUBROUTINE convert_soa_to_species


  !---------------------------------------------------------------------------
  !> @brief Sync species data to GPU (copy and transfer)
  !>
  !> Converts species to SoA and copies to GPU device memory.
  !>
  !> @param[in]    species   The particle species
  !> @param[inout] soa       The SoA structure
  !---------------------------------------------------------------------------
  SUBROUTINE sync_species_to_gpu(species, soa)
    TYPE(particle_species), INTENT(IN) :: species
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    
    ! Convert to SoA format
    CALL convert_species_to_soa(species, soa)
    
    IF (soa%count <= 0) RETURN
    
#ifdef GPU_ENABLED
    ! Copy to GPU device memory
    !$ACC ENTER DATA COPYIN(soa%pos_x(1:soa%count))
    !$ACC ENTER DATA COPYIN(soa%pos_y(1:soa%count))
    !$ACC ENTER DATA COPYIN(soa%px(1:soa%count))
    !$ACC ENTER DATA COPYIN(soa%py(1:soa%count))
    !$ACC ENTER DATA COPYIN(soa%pz(1:soa%count))
    !$ACC ENTER DATA COPYIN(soa%weight(1:soa%count))
#endif
    
  END SUBROUTINE sync_species_to_gpu


  !---------------------------------------------------------------------------
  !> @brief Sync species data from GPU (transfer and convert)
  !>
  !> Copies data from GPU device memory and converts back to AoS.
  !>
  !> @param[inout] soa       The SoA structure
  !> @param[inout] species   The particle species to update
  !---------------------------------------------------------------------------
  SUBROUTINE sync_species_from_gpu(soa, species)
    TYPE(particle_soa_t), INTENT(INOUT) :: soa
    TYPE(particle_species), INTENT(INOUT) :: species
    
    IF (soa%count <= 0) RETURN
    
#ifdef GPU_ENABLED
    ! Copy from GPU device memory
    !$ACC UPDATE HOST(soa%pos_x(1:soa%count))
    !$ACC UPDATE HOST(soa%pos_y(1:soa%count))
    !$ACC UPDATE HOST(soa%px(1:soa%count))
    !$ACC UPDATE HOST(soa%py(1:soa%count))
    !$ACC UPDATE HOST(soa%pz(1:soa%count))
    !$ACC UPDATE HOST(soa%weight(1:soa%count))
    
    ! Free GPU memory
    !$ACC EXIT DATA DELETE(soa%pos_x, soa%pos_y)
    !$ACC EXIT DATA DELETE(soa%px, soa%py, soa%pz)
    !$ACC EXIT DATA DELETE(soa%weight)
#endif
    
    ! Convert back to AoS format
    CALL convert_soa_to_species(soa, species)
    
  END SUBROUTINE sync_species_from_gpu

END MODULE particle_converter

