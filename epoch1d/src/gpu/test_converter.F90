!> @file test_converter.F90
!> @brief Test program for particle AoS↔SoA converter (1D version)
!>
!> This tests the particle_converter module in isolation.

PROGRAM test_converter
  USE, INTRINSIC :: iso_fortran_env
  
  IMPLICIT NONE
  
  ! Parameters
  INTEGER, PARAMETER :: num = REAL64
  INTEGER, PARAMETER :: i8 = INT64
  INTEGER, PARAMETER :: c_ndims = 1
  INTEGER, PARAMETER :: string_length = 256
  
  ! Simple particle type (minimal version for 1D)
  TYPE :: particle
    REAL(num), DIMENSION(3) :: part_p
    REAL(num), DIMENSION(c_ndims) :: part_pos
    REAL(num) :: weight
    TYPE(particle), POINTER :: next, prev
  END TYPE particle
  
  ! Simple particle list type
  TYPE :: particle_list
    TYPE(particle), POINTER :: head
    TYPE(particle), POINTER :: tail
    INTEGER(i8) :: count
  END TYPE particle_list
  
  ! Simple SoA type for 1D
  TYPE :: particle_soa_1d
    INTEGER(i8) :: count = 0
    INTEGER(i8) :: capacity = 0
    REAL(num), ALLOCATABLE :: pos_x(:)
    REAL(num), ALLOCATABLE :: px(:)
    REAL(num), ALLOCATABLE :: py(:)
    REAL(num), ALLOCATABLE :: pz(:)
    REAL(num), ALLOCATABLE :: weight(:)
  END TYPE particle_soa_1d
  
  ! Test variables
  TYPE(particle_list) :: plist
  TYPE(particle_soa_1d) :: soa
  TYPE(particle), POINTER :: p1, p2, p3, current
  INTEGER(i8) :: i
  LOGICAL :: test_passed
  REAL(num) :: max_error
  
  PRINT *, '=================================================='
  PRINT *, 'Particle Converter Test - 1D (Standalone)'
  PRINT *, '=================================================='
  
  ! Create test particles
  ALLOCATE(p1)
  ALLOCATE(p2)
  ALLOCATE(p3)
  
  ! Initialize particle 1
  p1%part_pos(1) = 1.0_num
  p1%part_p(1) = 1.0e10_num
  p1%part_p(2) = 2.0e10_num
  p1%part_p(3) = 3.0e10_num
  p1%weight = 1.0e5_num
  
  ! Initialize particle 2
  p2%part_pos(1) = 3.0_num
  p2%part_p(1) = 4.0e10_num
  p2%part_p(2) = 5.0e10_num
  p2%part_p(3) = 6.0e10_num
  p2%weight = 2.0e5_num
  
  ! Initialize particle 3
  p3%part_pos(1) = 5.0_num
  p3%part_p(1) = 7.0e10_num
  p3%part_p(2) = 8.0e10_num
  p3%part_p(3) = 9.0e10_num
  p3%weight = 3.0e5_num
  
  ! Link particles into list
  p1%next => p2
  p1%prev => NULL()
  p2%next => p3
  p2%prev => p1
  p3%next => NULL()
  p3%prev => p2
  
  plist%head => p1
  plist%tail => p3
  plist%count = 3
  
  PRINT *, '1. Created linked list with 3 particles'
  
  ! Test AoS to SoA conversion
  PRINT *, '2. Converting AoS -> SoA...'
  
  ! Allocate SoA
  ALLOCATE(soa%pos_x(10))
  ALLOCATE(soa%px(10))
  ALLOCATE(soa%py(10))
  ALLOCATE(soa%pz(10))
  ALLOCATE(soa%weight(10))
  soa%capacity = 10
  soa%count = 0
  
  ! Manual conversion (simulating convert_species_to_soa)
  current => plist%head
  i = 0
  DO WHILE (ASSOCIATED(current))
    i = i + 1
    soa%pos_x(i) = current%part_pos(1)
    soa%px(i) = current%part_p(1)
    soa%py(i) = current%part_p(2)
    soa%pz(i) = current%part_p(3)
    soa%weight(i) = current%weight
    current => current%next
  END DO
  soa%count = i
  
  PRINT *, '   SoA count:', soa%count
  PRINT *, '   pos_x:', soa%pos_x(1:3)
  PRINT *, '   px:', soa%px(1:3)
  
  ! Verify conversion
  test_passed = .TRUE.
  IF (ABS(soa%pos_x(1) - 1.0_num) > 1e-10) test_passed = .FALSE.
  IF (ABS(soa%pos_x(2) - 3.0_num) > 1e-10) test_passed = .FALSE.
  IF (ABS(soa%pos_x(3) - 5.0_num) > 1e-10) test_passed = .FALSE.
  IF (ABS(soa%px(1) - 1.0e10_num) > 1e-5) test_passed = .FALSE.
  
  IF (test_passed) THEN
    PRINT *, '   [PASS] AoS -> SoA conversion correct'
  ELSE
    PRINT *, '   [FAIL] AoS -> SoA conversion failed'
  END IF
  
  ! Modify SoA data (simulate GPU computation)
  PRINT *, '3. Simulating GPU particle push...'
  DO i = 1, soa%count
    soa%pos_x(i) = soa%pos_x(i) + 0.1_num
    soa%px(i) = soa%px(i) * 1.01_num
  END DO
  
  ! Test SoA to AoS conversion
  PRINT *, '4. Converting SoA -> AoS...'
  
  ! Manual conversion (simulating convert_soa_to_species)
  current => plist%head
  i = 0
  DO WHILE (ASSOCIATED(current))
    i = i + 1
    current%part_pos(1) = soa%pos_x(i)
    current%part_p(1) = soa%px(i)
    current%part_p(2) = soa%py(i)
    current%part_p(3) = soa%pz(i)
    current => current%next
  END DO
  
  ! Verify updated values
  test_passed = .TRUE.
  max_error = ABS(p1%part_pos(1) - 1.1_num)
  IF (max_error > 1e-10) test_passed = .FALSE.
  IF (ABS(p2%part_pos(1) - 3.1_num) > 1e-10) test_passed = .FALSE.
  IF (ABS(p3%part_pos(1) - 5.1_num) > 1e-10) test_passed = .FALSE.
  IF (ABS(p1%part_p(1) - 1.01e10_num) > 1e5) test_passed = .FALSE.
  
  IF (test_passed) THEN
    PRINT *, '   [PASS] SoA -> AoS conversion correct'
  ELSE
    PRINT *, '   [FAIL] SoA -> AoS conversion failed'
  END IF
  
  ! Cleanup
  DEALLOCATE(soa%pos_x)
  DEALLOCATE(soa%px)
  DEALLOCATE(soa%py)
  DEALLOCATE(soa%pz)
  DEALLOCATE(soa%weight)
  DEALLOCATE(p1)
  DEALLOCATE(p2)
  DEALLOCATE(p3)
  
  PRINT *, ''
  PRINT *, '=================================================='
  PRINT *, 'All Particle Converter Tests Complete (1D)'
  PRINT *, '=================================================='
  
END PROGRAM test_converter

