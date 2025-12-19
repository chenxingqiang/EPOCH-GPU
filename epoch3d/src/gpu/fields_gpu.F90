! Copyright (C) 2024 EPOCH-GPU Development Team
!
! @file fields_gpu.F90
! @brief GPU-accelerated electromagnetic field solver for EPOCH3D
! @details Implements FDTD field updates using OpenACC for 3D grids

MODULE fields_gpu

#ifdef GPU_ENABLED
  USE cudafor
#endif
  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT32, INT64, REAL64
  
  IMPLICIT NONE
  PRIVATE
  
  INTEGER, PARAMETER :: num = REAL64
  
  PUBLIC :: update_e_field_gpu_3d
  PUBLIC :: update_b_field_gpu_3d

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated E-field update for 3D (2nd order Yee scheme)
  !> @details Updates electric field using Ampere's law:
  !>          dE/dt = c^2 * curl(B) - J/epsilon_0
  !---------------------------------------------------------------------------
  SUBROUTINE update_e_field_gpu_3d(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                    nx, ny, nz, ng, cnx, cny, cnz, fac)
    INTEGER, INTENT(IN) :: nx, ny, nz, ng
    REAL(num), INTENT(INOUT) :: ex(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(INOUT) :: ey(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(INOUT) :: ez(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: bx(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: by(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: bz(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: jx(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: jy(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: jz(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: cnx, cny, cnz, fac
    
    INTEGER :: ix, iy, iz
    
    !$ACC PARALLEL LOOP COLLAPSE(3) PRESENT(ex, ey, ez, bx, by, bz, jx, jy, jz)
    DO iz = 1, nz
      DO iy = 1, ny
        DO ix = 1, nx
          ! Ex: dEx/dt = c^2 * (dBz/dy - dBy/dz) - Jx/eps0
          ex(ix, iy, iz) = ex(ix, iy, iz) &
              + cny * (bz(ix, iy, iz) - bz(ix, iy-1, iz)) &
              - cnz * (by(ix, iy, iz) - by(ix, iy, iz-1)) &
              - fac * jx(ix, iy, iz)
          
          ! Ey: dEy/dt = c^2 * (dBx/dz - dBz/dx) - Jy/eps0
          ey(ix, iy, iz) = ey(ix, iy, iz) &
              + cnz * (bx(ix, iy, iz) - bx(ix, iy, iz-1)) &
              - cnx * (bz(ix, iy, iz) - bz(ix-1, iy, iz)) &
              - fac * jy(ix, iy, iz)
          
          ! Ez: dEz/dt = c^2 * (dBy/dx - dBx/dy) - Jz/eps0
          ez(ix, iy, iz) = ez(ix, iy, iz) &
              + cnx * (by(ix, iy, iz) - by(ix-1, iy, iz)) &
              - cny * (bx(ix, iy, iz) - bx(ix, iy-1, iz)) &
              - fac * jz(ix, iy, iz)
        END DO
      END DO
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE update_e_field_gpu_3d


  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated B-field update for 3D (2nd order Yee scheme)
  !> @details Updates magnetic field using Faraday's law:
  !>          dB/dt = -curl(E)
  !---------------------------------------------------------------------------
  SUBROUTINE update_b_field_gpu_3d(ex, ey, ez, bx, by, bz, &
                                    nx, ny, nz, ng, hdtx, hdty, hdtz)
    INTEGER, INTENT(IN) :: nx, ny, nz, ng
    REAL(num), INTENT(IN) :: ex(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: ey(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: ez(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(INOUT) :: bx(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(INOUT) :: by(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(INOUT) :: bz(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: hdtx, hdty, hdtz
    
    INTEGER :: ix, iy, iz
    
    !$ACC PARALLEL LOOP COLLAPSE(3) PRESENT(ex, ey, ez, bx, by, bz)
    DO iz = 1, nz
      DO iy = 1, ny
        DO ix = 1, nx
          ! Bx: dBx/dt = dEy/dz - dEz/dy
          bx(ix, iy, iz) = bx(ix, iy, iz) &
              + hdtz * (ey(ix, iy, iz+1) - ey(ix, iy, iz)) &
              - hdty * (ez(ix, iy+1, iz) - ez(ix, iy, iz))
          
          ! By: dBy/dt = dEz/dx - dEx/dz
          by(ix, iy, iz) = by(ix, iy, iz) &
              + hdtx * (ez(ix+1, iy, iz) - ez(ix, iy, iz)) &
              - hdtz * (ex(ix, iy, iz+1) - ex(ix, iy, iz))
          
          ! Bz: dBz/dt = dEx/dy - dEy/dx
          bz(ix, iy, iz) = bz(ix, iy, iz) &
              + hdty * (ex(ix, iy+1, iz) - ex(ix, iy, iz)) &
              - hdtx * (ey(ix+1, iy, iz) - ey(ix, iy, iz))
        END DO
      END DO
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE update_b_field_gpu_3d

END MODULE fields_gpu

