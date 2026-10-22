! Copyright (C) 2025 EPOCH-GPU Development Team
!
! @file fields_gpu.F90
! @brief GPU-accelerated electromagnetic field solver for EPOCH1D
! @details Implements FDTD field updates using OpenACC for 1D grids

MODULE fields_gpu

#ifdef GPU_ENABLED
  USE cudafor
#endif
  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT32, INT64, REAL64
  
  IMPLICIT NONE
  PRIVATE
  
  INTEGER, PARAMETER :: num = REAL64
  
  PUBLIC :: update_e_field_gpu_1d
  PUBLIC :: update_b_field_gpu_1d

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated E-field update for 1D (2nd order Yee scheme)
  !> @details Updates electric field using Ampere's law:
  !>          dE/dt = c^2 * curl(B) - J/epsilon_0
  !---------------------------------------------------------------------------
  SUBROUTINE update_e_field_gpu_1d(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                    nx, ng, cnx, fac)
    INTEGER, INTENT(IN) :: nx, ng
    REAL(num), INTENT(INOUT) :: ex(1-ng:nx+ng)
    REAL(num), INTENT(INOUT) :: ey(1-ng:nx+ng)
    REAL(num), INTENT(INOUT) :: ez(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: bx(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: by(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: bz(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: jx(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: jy(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: jz(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: cnx, fac
    
    INTEGER :: ix
    
    !$ACC PARALLEL LOOP PRESENT(ex, ey, ez, bx, by, bz, jx, jy, jz)
    DO ix = 0, nx
      ! Ex update (no curl contribution in 1D for Ex)
      ex(ix) = ex(ix) - fac * jx(ix)
      
      ! Ey update: dEy/dt = -c^2 * dBz/dx - Jy/eps0
      ey(ix) = ey(ix) - cnx * (bz(ix) - bz(ix-1)) - fac * jy(ix)
      
      ! Ez update: dEz/dt = c^2 * dBy/dx - Jz/eps0
      ez(ix) = ez(ix) + cnx * (by(ix) - by(ix-1)) - fac * jz(ix)
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE update_e_field_gpu_1d


  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated B-field update for 1D (2nd order Yee scheme)
  !> @details Updates magnetic field using Faraday's law:
  !>          dB/dt = -curl(E)
  !---------------------------------------------------------------------------
  SUBROUTINE update_b_field_gpu_1d(ex, ey, ez, bx, by, bz, &
                                    nx, ng, hdtx)
    INTEGER, INTENT(IN) :: nx, ng
    REAL(num), INTENT(IN) :: ex(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: ey(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: ez(1-ng:nx+ng)
    REAL(num), INTENT(INOUT) :: bx(1-ng:nx+ng)
    REAL(num), INTENT(INOUT) :: by(1-ng:nx+ng)
    REAL(num), INTENT(INOUT) :: bz(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: hdtx
    
    INTEGER :: ix
    
    !$ACC PARALLEL LOOP PRESENT(ex, ey, ez, bx, by, bz)
    DO ix = 0, nx
      ! Bx: no curl contribution in 1D
      ! bx(ix) = bx(ix)
      
      ! By update: dBy/dt = dEz/dx
      by(ix) = by(ix) + hdtx * (ez(ix+1) - ez(ix))
      
      ! Bz update: dBz/dt = -dEy/dx
      bz(ix) = bz(ix) - hdtx * (ey(ix+1) - ey(ix))
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE update_b_field_gpu_1d

END MODULE fields_gpu

