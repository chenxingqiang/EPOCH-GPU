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

!> @file particles_gpu.F90
!> @brief GPU-accelerated particle pusher using OpenACC
!> @details Implements the Boris particle push algorithm on GPU using
!>          Structure of Arrays (SoA) data layout for coalesced memory access.

MODULE particles_gpu

  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT32, INT64, REAL64
  USE gpu_config
  
  IMPLICIT NONE
  PRIVATE
  
  !---------------------------------------------------------------------------
  ! Public interfaces
  !---------------------------------------------------------------------------
  PUBLIC :: push_particles_gpu
  PUBLIC :: interpolate_fields_to_particles_gpu
  PUBLIC :: deposit_current_gpu
  
  !---------------------------------------------------------------------------
  ! Precision parameters
  !---------------------------------------------------------------------------
  INTEGER, PARAMETER :: num = REAL64
  INTEGER, PARAMETER :: i8 = INT64
  
  !---------------------------------------------------------------------------
  ! Physical constants (normalized)
  !---------------------------------------------------------------------------
  REAL(num), PARAMETER :: c = 1.0_num          !< Speed of light (normalized)
  REAL(num), PARAMETER :: c2 = 1.0_num         !< c^2

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated Boris particle pusher
  !> @details Pushes particles using the Boris algorithm with relativistic
  !>          momentum update. Uses SoA data layout for GPU efficiency.
  !>
  !> @param[in] n_particles Number of particles
  !> @param[inout] pos_x, pos_y Particle positions
  !> @param[inout] px, py, pz Particle momenta
  !> @param[in] ex_p, ey_p, ez_p Electric field at particle positions
  !> @param[in] bx_p, by_p, bz_p Magnetic field at particle positions
  !> @param[in] charge Particle charge
  !> @param[in] mass Particle mass
  !> @param[in] dt Time step
  !---------------------------------------------------------------------------
  SUBROUTINE push_particles_gpu(n_particles, pos_x, pos_y, px, py, pz, &
                                ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                                charge, mass, dt)
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(INOUT) :: pos_x(:), pos_y(:)
    REAL(num), INTENT(INOUT) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), INTENT(IN) :: bx_p(:), by_p(:), bz_p(:)
    REAL(num), INTENT(IN) :: charge, mass, dt
    
    INTEGER(i8) :: ip
    REAL(num) :: part_ux, part_uy, part_uz
    REAL(num) :: uxm, uym, uzm, uxp, uyp, uzp
    REAL(num) :: gamma_rel, igamma, root
    REAL(num) :: taux, tauy, tauz, tau
    REAL(num) :: taux2, tauy2, tauz2
    REAL(num) :: cmratio, ccmratio
    REAL(num) :: part_mc, ipart_mc
    REAL(num) :: dto2, dtco2, dtfac
    
    ! Precompute constants
    part_mc = c * mass
    ipart_mc = 1.0_num / part_mc
    dtfac = 0.5_num * dt
    cmratio = charge * dtfac * ipart_mc
    ccmratio = c * cmratio
    dto2 = dt / 2.0_num
    dtco2 = c * dto2
    
    !$ACC PARALLEL LOOP PRESENT(pos_x, pos_y, px, py, pz) &
    !$ACC   PRESENT(ex_p, ey_p, ez_p, bx_p, by_p, bz_p) &
    !$ACC   PRIVATE(part_ux, part_uy, part_uz) &
    !$ACC   PRIVATE(uxm, uym, uzm, uxp, uyp, uzp) &
    !$ACC   PRIVATE(gamma_rel, igamma, root) &
    !$ACC   PRIVATE(taux, tauy, tauz, tau, taux2, tauy2, tauz2)
    DO ip = 1, n_particles
      ! Convert momentum to velocity-like quantity (u = p/mc)
      part_ux = px(ip) * ipart_mc
      part_uy = py(ip) * ipart_mc
      part_uz = pz(ip) * ipart_mc
      
      ! Calculate initial gamma
      gamma_rel = SQRT(part_ux**2 + part_uy**2 + part_uz**2 + 1.0_num)
      root = dtco2 / gamma_rel
      
      ! First half of position update (drift)
      pos_x(ip) = pos_x(ip) + part_ux * root
      pos_y(ip) = pos_y(ip) + part_uy * root
      
      !-----------------------------------------------------------------------
      ! Boris push algorithm
      !-----------------------------------------------------------------------
      
      ! First half of electric field acceleration
      uxm = part_ux + cmratio * ex_p(ip)
      uym = part_uy + cmratio * ey_p(ip)
      uzm = part_uz + cmratio * ez_p(ip)
      
      ! Calculate gamma at mid-step
      gamma_rel = SQRT(uxm**2 + uym**2 + uzm**2 + 1.0_num)
      root = ccmratio / gamma_rel
      
      ! Magnetic field rotation coefficients
      taux = bx_p(ip) * root
      tauy = by_p(ip) * root
      tauz = bz_p(ip) * root
      
      taux2 = taux * taux
      tauy2 = tauy * tauy
      tauz2 = tauz * tauz
      
      tau = 1.0_num / (1.0_num + taux2 + tauy2 + tauz2)
      
      ! Magnetic field rotation
      uxp = ((1.0_num + taux2 - tauy2 - tauz2) * uxm &
          + 2.0_num * ((taux * tauy + tauz) * uym &
          + (taux * tauz - tauy) * uzm)) * tau
      
      uyp = ((1.0_num - taux2 + tauy2 - tauz2) * uym &
          + 2.0_num * ((tauy * tauz + taux) * uzm &
          + (tauy * taux - tauz) * uxm)) * tau
      
      uzp = ((1.0_num - taux2 - tauy2 + tauz2) * uzm &
          + 2.0_num * ((tauz * taux + tauy) * uxm &
          + (tauz * tauy - taux) * uym)) * tau
      
      ! Second half of electric field acceleration
      part_ux = uxp + cmratio * ex_p(ip)
      part_uy = uyp + cmratio * ey_p(ip)
      part_uz = uzp + cmratio * ez_p(ip)
      
      ! Calculate final gamma
      gamma_rel = SQRT(part_ux**2 + part_uy**2 + part_uz**2 + 1.0_num)
      igamma = 1.0_num / gamma_rel
      root = dtco2 * igamma
      
      ! Second half of position update
      pos_x(ip) = pos_x(ip) + part_ux * root
      pos_y(ip) = pos_y(ip) + part_uy * root
      
      ! Store updated momenta
      px(ip) = part_ux * part_mc
      py(ip) = part_uy * part_mc
      pz(ip) = part_uz * part_mc
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE push_particles_gpu


  !---------------------------------------------------------------------------
  !> @brief Interpolate electromagnetic fields to particle positions
  !> @details Uses bilinear interpolation (1st order) for speed
  !>
  !> @param[in] n_particles Number of particles
  !> @param[in] pos_x, pos_y Particle positions
  !> @param[in] ex, ey, ez Electric field arrays
  !> @param[in] bx, by, bz Magnetic field arrays
  !> @param[out] ex_p, ey_p, ez_p Electric field at particles
  !> @param[out] bx_p, by_p, bz_p Magnetic field at particles
  !> @param[in] x_min, y_min Grid origin
  !> @param[in] dx, dy Grid spacing
  !> @param[in] nx, ny Grid dimensions
  !> @param[in] ng Number of ghost cells
  !---------------------------------------------------------------------------
  SUBROUTINE interpolate_fields_to_particles_gpu(n_particles, pos_x, pos_y, &
      ex, ey, ez, bx, by, bz, &
      ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
      x_min, y_min, dx, dy, nx, ny, ng)
    
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: pos_x(:), pos_y(:)
    REAL(num), INTENT(IN) :: ex(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: ey(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: ez(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: bx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: by(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: bz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(OUT) :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), INTENT(OUT) :: bx_p(:), by_p(:), bz_p(:)
    REAL(num), INTENT(IN) :: x_min, y_min, dx, dy
    INTEGER, INTENT(IN) :: nx, ny, ng
    
    INTEGER(i8) :: ip
    INTEGER :: ix, iy
    REAL(num) :: idx, idy
    REAL(num) :: rx, ry
    REAL(num) :: wx0, wx1, wy0, wy1
    REAL(num) :: hx0, hx1, hy0, hy1
    
    idx = 1.0_num / dx
    idy = 1.0_num / dy
    
    !$ACC PARALLEL LOOP PRESENT(pos_x, pos_y) &
    !$ACC   PRESENT(ex, ey, ez, bx, by, bz) &
    !$ACC   PRESENT(ex_p, ey_p, ez_p, bx_p, by_p, bz_p) &
    !$ACC   PRIVATE(ix, iy, rx, ry, wx0, wx1, wy0, wy1, hx0, hx1, hy0, hy1)
    DO ip = 1, n_particles
      ! Cell indices (1-based)
      rx = (pos_x(ip) - x_min) * idx
      ry = (pos_y(ip) - y_min) * idy
      
      ix = FLOOR(rx) + 1
      iy = FLOOR(ry) + 1
      
      ! Ensure indices are within bounds
      ix = MAX(1, MIN(ix, nx))
      iy = MAX(1, MIN(iy, ny))
      
      ! Weight factors for cell-centered quantities
      wx1 = rx - REAL(ix - 1, num)
      wy1 = ry - REAL(iy - 1, num)
      wx0 = 1.0_num - wx1
      wy0 = 1.0_num - wy1
      
      ! Weight factors for staggered quantities (shifted by 0.5)
      hx1 = rx - REAL(ix - 1, num) + 0.5_num
      IF (hx1 >= 1.0_num) THEN
        hx1 = hx1 - 1.0_num
      END IF
      hx0 = 1.0_num - hx1
      
      hy1 = ry - REAL(iy - 1, num) + 0.5_num
      IF (hy1 >= 1.0_num) THEN
        hy1 = hy1 - 1.0_num
      END IF
      hy0 = 1.0_num - hy1
      
      ! Bilinear interpolation for E-field (staggered grid)
      ! Ex is staggered in x
      ex_p(ip) = hx0 * (wy0 * ex(ix,   iy) + wy1 * ex(ix,   iy+1)) &
               + hx1 * (wy0 * ex(ix+1, iy) + wy1 * ex(ix+1, iy+1))
      
      ! Ey is staggered in y
      ey_p(ip) = wx0 * (hy0 * ey(ix,   iy) + hy1 * ey(ix,   iy+1)) &
               + wx1 * (hy0 * ey(ix+1, iy) + hy1 * ey(ix+1, iy+1))
      
      ! Ez is cell-centered
      ez_p(ip) = wx0 * (wy0 * ez(ix,   iy) + wy1 * ez(ix,   iy+1)) &
               + wx1 * (wy0 * ez(ix+1, iy) + wy1 * ez(ix+1, iy+1))
      
      ! Bilinear interpolation for B-field (staggered grid)
      ! Bx is staggered in y,z
      bx_p(ip) = wx0 * (hy0 * bx(ix,   iy) + hy1 * bx(ix,   iy+1)) &
               + wx1 * (hy0 * bx(ix+1, iy) + hy1 * bx(ix+1, iy+1))
      
      ! By is staggered in x,z
      by_p(ip) = hx0 * (wy0 * by(ix,   iy) + wy1 * by(ix,   iy+1)) &
               + hx1 * (wy0 * by(ix+1, iy) + wy1 * by(ix+1, iy+1))
      
      ! Bz is staggered in x,y
      bz_p(ip) = hx0 * (hy0 * bz(ix,   iy) + hy1 * bz(ix,   iy+1)) &
               + hx1 * (hy0 * bz(ix+1, iy) + hy1 * bz(ix+1, iy+1))
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE interpolate_fields_to_particles_gpu


  !---------------------------------------------------------------------------
  !> @brief Deposit current from particles to grid using atomic operations
  !> @details Uses first-order (linear) weighting with atomic updates for
  !>          thread safety on GPU.
  !>
  !> @param[in] n_particles Number of particles
  !> @param[in] pos_x, pos_y Current particle positions
  !> @param[in] pos_x_old, pos_y_old Previous particle positions
  !> @param[in] px, py, pz Particle momenta
  !> @param[in] weight Particle weight
  !> @param[in] charge Particle charge
  !> @param[in] mass Particle mass
  !> @param[inout] jx, jy, jz Current density arrays
  !> @param[in] x_min, y_min Grid origin
  !> @param[in] dx, dy, dt Grid spacing and time step
  !> @param[in] nx, ny, ng Grid dimensions
  !---------------------------------------------------------------------------
  SUBROUTINE deposit_current_gpu(n_particles, pos_x, pos_y, &
      pos_x_old, pos_y_old, px, py, pz, weight, charge, mass, &
      jx, jy, jz, x_min, y_min, dx, dy, dt, nx, ny, ng)
    
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: pos_x(:), pos_y(:)
    REAL(num), INTENT(IN) :: pos_x_old(:), pos_y_old(:)
    REAL(num), INTENT(IN) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: weight, charge, mass
    REAL(num), INTENT(INOUT) :: jx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jy(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: x_min, y_min, dx, dy, dt
    INTEGER, INTENT(IN) :: nx, ny, ng
    
    INTEGER(i8) :: ip
    INTEGER :: ix, iy
    REAL(num) :: idx, idy, idt
    REAL(num) :: rx, ry
    REAL(num) :: wx0, wx1, wy0, wy1
    REAL(num) :: part_mc, gamma_rel, vx, vy, vz
    REAL(num) :: qw_dx, qw_dy, qw_dz
    REAL(num) :: factor
    
    idx = 1.0_num / dx
    idy = 1.0_num / dy
    idt = 1.0_num / dt
    part_mc = c * mass
    factor = weight * charge / (dx * dy)
    
    !$ACC PARALLEL LOOP PRESENT(pos_x, pos_y, px, py, pz, jx, jy, jz) &
    !$ACC   PRIVATE(ix, iy, rx, ry, wx0, wx1, wy0, wy1) &
    !$ACC   PRIVATE(gamma_rel, vx, vy, vz, qw_dx, qw_dy, qw_dz)
    DO ip = 1, n_particles
      ! Calculate velocity
      gamma_rel = SQRT((px(ip)/part_mc)**2 + (py(ip)/part_mc)**2 &
                     + (pz(ip)/part_mc)**2 + 1.0_num)
      vx = px(ip) / (gamma_rel * mass)
      vy = py(ip) / (gamma_rel * mass)
      vz = pz(ip) / (gamma_rel * mass)
      
      ! Current contribution from this particle
      qw_dx = factor * vx
      qw_dy = factor * vy
      qw_dz = factor * vz
      
      ! Cell indices for current position
      rx = (pos_x(ip) - x_min) * idx
      ry = (pos_y(ip) - y_min) * idy
      
      ix = FLOOR(rx) + 1
      iy = FLOOR(ry) + 1
      
      ! Clamp to valid range
      ix = MAX(1, MIN(ix, nx))
      iy = MAX(1, MIN(iy, ny))
      
      ! Weight factors
      wx1 = rx - REAL(ix - 1, num)
      wy1 = ry - REAL(iy - 1, num)
      wx0 = 1.0_num - wx1
      wy0 = 1.0_num - wy1
      
      ! Deposit current using atomic operations for thread safety
      !$ACC ATOMIC UPDATE
      jx(ix,   iy)   = jx(ix,   iy)   + wx0 * wy0 * qw_dx
      !$ACC ATOMIC UPDATE
      jx(ix+1, iy)   = jx(ix+1, iy)   + wx1 * wy0 * qw_dx
      !$ACC ATOMIC UPDATE
      jx(ix,   iy+1) = jx(ix,   iy+1) + wx0 * wy1 * qw_dx
      !$ACC ATOMIC UPDATE
      jx(ix+1, iy+1) = jx(ix+1, iy+1) + wx1 * wy1 * qw_dx
      
      !$ACC ATOMIC UPDATE
      jy(ix,   iy)   = jy(ix,   iy)   + wx0 * wy0 * qw_dy
      !$ACC ATOMIC UPDATE
      jy(ix+1, iy)   = jy(ix+1, iy)   + wx1 * wy0 * qw_dy
      !$ACC ATOMIC UPDATE
      jy(ix,   iy+1) = jy(ix,   iy+1) + wx0 * wy1 * qw_dy
      !$ACC ATOMIC UPDATE
      jy(ix+1, iy+1) = jy(ix+1, iy+1) + wx1 * wy1 * qw_dy
      
      !$ACC ATOMIC UPDATE
      jz(ix,   iy)   = jz(ix,   iy)   + wx0 * wy0 * qw_dz
      !$ACC ATOMIC UPDATE
      jz(ix+1, iy)   = jz(ix+1, iy)   + wx1 * wy0 * qw_dz
      !$ACC ATOMIC UPDATE
      jz(ix,   iy+1) = jz(ix,   iy+1) + wx0 * wy1 * qw_dz
      !$ACC ATOMIC UPDATE
      jz(ix+1, iy+1) = jz(ix+1, iy+1) + wx1 * wy1 * qw_dz
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE deposit_current_gpu

END MODULE particles_gpu

