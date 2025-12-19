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

!> @file fields_gpu.F90
!> @brief GPU-accelerated electromagnetic field solver
!> @details This module provides GPU implementations of the FDTD Maxwell
!>          solver using OpenACC directives for portability.

MODULE fields_gpu

  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT32, INT64, REAL64
  USE gpu_config
  
  IMPLICIT NONE
  PRIVATE
  
  !---------------------------------------------------------------------------
  ! Public interfaces
  !---------------------------------------------------------------------------
  PUBLIC :: update_e_field_gpu, update_b_field_gpu
  PUBLIC :: fields_copy_to_device, fields_copy_to_host
  PUBLIC :: fields_gpu_init, fields_gpu_finalize
  
  !---------------------------------------------------------------------------
  ! Precision parameter
  !---------------------------------------------------------------------------
  INTEGER, PARAMETER :: num = REAL64
  
  !---------------------------------------------------------------------------
  ! Module state
  !---------------------------------------------------------------------------
  LOGICAL, SAVE :: fields_on_device = .FALSE.

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief Initialize GPU field arrays
  !> @details Copy field arrays to GPU device memory
  !---------------------------------------------------------------------------
  SUBROUTINE fields_gpu_init(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                             nx, ny, ng)
    REAL(num), INTENT(INOUT) :: ex(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ey(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ez(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: by(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jy(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jz(1-ng:nx+ng, 1-ng:ny+ng)
    INTEGER, INTENT(IN) :: nx, ny, ng
    
    IF (.NOT. gpu_enabled) RETURN
    
    !$ACC ENTER DATA COPYIN(ex, ey, ez, bx, by, bz, jx, jy, jz)
    
    fields_on_device = .TRUE.
    
  END SUBROUTINE fields_gpu_init


  !---------------------------------------------------------------------------
  !> @brief Finalize GPU field arrays
  !> @details Copy field arrays back to host and deallocate device memory
  !---------------------------------------------------------------------------
  SUBROUTINE fields_gpu_finalize(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                 nx, ny, ng)
    REAL(num), INTENT(INOUT) :: ex(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ey(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ez(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: by(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jy(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jz(1-ng:nx+ng, 1-ng:ny+ng)
    INTEGER, INTENT(IN) :: nx, ny, ng
    
    IF (.NOT. gpu_enabled) RETURN
    IF (.NOT. fields_on_device) RETURN
    
    !$ACC EXIT DATA COPYOUT(ex, ey, ez, bx, by, bz) DELETE(jx, jy, jz)
    
    fields_on_device = .FALSE.
    
  END SUBROUTINE fields_gpu_finalize


  !---------------------------------------------------------------------------
  !> @brief Copy field arrays to GPU device
  !---------------------------------------------------------------------------
  SUBROUTINE fields_copy_to_device(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                   nx, ny, ng)
    REAL(num), INTENT(IN) :: ex(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: ey(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: ez(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: bx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: by(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: bz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: jx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: jy(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: jz(1-ng:nx+ng, 1-ng:ny+ng)
    INTEGER, INTENT(IN) :: nx, ny, ng
    
    IF (.NOT. gpu_enabled) RETURN
    IF (.NOT. fields_on_device) RETURN
    
    !$ACC UPDATE DEVICE(ex, ey, ez, bx, by, bz, jx, jy, jz)
    
  END SUBROUTINE fields_copy_to_device


  !---------------------------------------------------------------------------
  !> @brief Copy field arrays from GPU device to host
  !---------------------------------------------------------------------------
  SUBROUTINE fields_copy_to_host(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                 nx, ny, ng)
    REAL(num), INTENT(INOUT) :: ex(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ey(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ez(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: by(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jy(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jz(1-ng:nx+ng, 1-ng:ny+ng)
    INTEGER, INTENT(IN) :: nx, ny, ng
    
    IF (.NOT. gpu_enabled) RETURN
    IF (.NOT. fields_on_device) RETURN
    
    !$ACC UPDATE HOST(ex, ey, ez, bx, by, bz, jx, jy, jz)
    
  END SUBROUTINE fields_copy_to_host


  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated E-field update (2nd order Yee scheme)
  !> @details Updates electric field components using Maxwell's equations:
  !>          dE/dt = (1/eps0) * (curl(B) - J)
  !---------------------------------------------------------------------------
  SUBROUTINE update_e_field_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                                nx, ny, ng, cnx, cny, fac)
    REAL(num), INTENT(INOUT) :: ex(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ey(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: ez(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: bx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: by(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: bz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: jx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: jy(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: jz(1-ng:nx+ng, 1-ng:ny+ng)
    INTEGER, INTENT(IN) :: nx, ny, ng
    REAL(num), INTENT(IN) :: cnx, cny, fac
    
    INTEGER :: ix, iy
    REAL(num) :: cx1, cy1
    
    cx1 = cnx
    cy1 = cny
    
    !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(ex, ey, ez, bx, by, bz, jx, jy, jz) &
    !$ACC   PRIVATE(ix, iy)
    DO iy = 0, ny
      DO ix = 0, nx
        ! Ex update: dEx/dt = (1/eps0) * (dBz/dy - Jx)
        ex(ix, iy) = ex(ix, iy) &
            + cy1 * (bz(ix, iy) - bz(ix, iy-1)) &
            - fac * jx(ix, iy)
        
        ! Ey update: dEy/dt = (1/eps0) * (-dBz/dx - Jy)
        ey(ix, iy) = ey(ix, iy) &
            - cx1 * (bz(ix, iy) - bz(ix-1, iy)) &
            - fac * jy(ix, iy)
        
        ! Ez update: dEz/dt = (1/eps0) * (dBy/dx - dBx/dy - Jz)
        ez(ix, iy) = ez(ix, iy) &
            + cx1 * (by(ix, iy) - by(ix-1, iy)) &
            - cy1 * (bx(ix, iy) - bx(ix, iy-1)) &
            - fac * jz(ix, iy)
      END DO
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE update_e_field_gpu


  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated B-field update (2nd order Yee scheme)
  !> @details Updates magnetic field components using Faraday's law:
  !>          dB/dt = -curl(E)
  !---------------------------------------------------------------------------
  SUBROUTINE update_b_field_gpu(ex, ey, ez, bx, by, bz, &
                                nx, ny, ng, hdtx, hdty)
    REAL(num), INTENT(IN) :: ex(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: ey(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: ez(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: by(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: bz(1-ng:nx+ng, 1-ng:ny+ng)
    INTEGER, INTENT(IN) :: nx, ny, ng
    REAL(num), INTENT(IN) :: hdtx, hdty
    
    INTEGER :: ix, iy
    REAL(num) :: cx1, cy1
    
    cx1 = hdtx
    cy1 = hdty
    
    !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(ex, ey, ez, bx, by, bz) &
    !$ACC   PRIVATE(ix, iy)
    DO iy = 0, ny
      DO ix = 0, nx
        ! Bx update: dBx/dt = -dEz/dy
        bx(ix, iy) = bx(ix, iy) &
            - cy1 * (ez(ix, iy+1) - ez(ix, iy))
        
        ! By update: dBy/dt = dEz/dx
        by(ix, iy) = by(ix, iy) &
            + cx1 * (ez(ix+1, iy) - ez(ix, iy))
        
        ! Bz update: dBz/dt = -dEy/dx + dEx/dy
        bz(ix, iy) = bz(ix, iy) &
            - cx1 * (ey(ix+1, iy) - ey(ix, iy)) &
            + cy1 * (ex(ix, iy+1) - ex(ix, iy))
      END DO
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE update_b_field_gpu

END MODULE fields_gpu

