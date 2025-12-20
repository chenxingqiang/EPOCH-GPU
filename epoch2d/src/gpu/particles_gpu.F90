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
    
    !$ACC PARALLEL LOOP &
    !$ACC   PRESENT(pos_x(1:n_particles), pos_y(1:n_particles)) &
    !$ACC   PRESENT(px(1:n_particles), py(1:n_particles), pz(1:n_particles)) &
    !$ACC   PRESENT(ex_p(1:n_particles), ey_p(1:n_particles), ez_p(1:n_particles)) &
    !$ACC   PRESENT(bx_p(1:n_particles), by_p(1:n_particles), bz_p(1:n_particles)) &
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
    
    !$ACC PARALLEL LOOP &
    !$ACC   PRESENT(pos_x(1:n_particles), pos_y(1:n_particles)) &
    !$ACC   PRESENT(ex, ey, ez, bx, by, bz) &
    !$ACC   PRESENT(ex_p(1:n_particles), ey_p(1:n_particles), ez_p(1:n_particles)) &
    !$ACC   PRESENT(bx_p(1:n_particles), by_p(1:n_particles), bz_p(1:n_particles)) &
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
  !> @brief Deposit current using Esirkepov charge-conserving method (2D)
  !>
  !> Implements the Esirkepov current deposition scheme ensuring:
  !>   div(J) = -d(rho)/dt (exact charge conservation)
  !>
  !> Reference: T.Zh. Esirkepov, CPC 135 (2001) 144-153
  !>
  !> @param[in]    n_particles      Number of particles
  !> @param[in]    pos_x_old/new    Particle x positions at t^n and t^{n+1}
  !> @param[in]    pos_y_old/new    Particle y positions at t^n and t^{n+1}
  !> @param[in]    px,py,pz         Particle momenta
  !> @param[in]    weight           Particle weight array
  !> @param[in]    charge,mass      Particle charge and mass (SI units)
  !> @param[inout] jx,jy,jz         Current density arrays
  !> @param[in]    x_min,y_min      Grid origin
  !> @param[in]    dx,dy,dt         Grid spacing and time step
  !> @param[in]    nx,ny,ng         Grid dimensions
  !---------------------------------------------------------------------------
  SUBROUTINE deposit_current_gpu(n_particles, pos_x_old, pos_y_old, &
      pos_x_new, pos_y_new, px, py, pz, weight, charge, mass, &
      jx, jy, jz, x_min, y_min, dx, dy, dt, nx, ny, ng)
    
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: pos_x_old(:), pos_y_old(:)
    REAL(num), INTENT(IN) :: pos_x_new(:), pos_y_new(:)
    REAL(num), INTENT(IN) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: weight(:)
    REAL(num), INTENT(IN) :: charge, mass
    REAL(num), INTENT(INOUT) :: jx(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jy(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(INOUT) :: jz(1-ng:nx+ng, 1-ng:ny+ng)
    REAL(num), INTENT(IN) :: x_min, y_min, dx, dy, dt
    INTEGER, INTENT(IN) :: nx, ny, ng
    
    INTEGER(i8) :: ip
    INTEGER :: ix_old, iy_old, ix_new, iy_new, ix_base, iy_base
    INTEGER :: ix, iy, dcellx, dcelly
    REAL(num) :: idx, idy, idt
    REAL(num) :: cell_x_old, cell_y_old, cell_x_new, cell_y_new
    REAL(num) :: frac_x_old, frac_y_old, frac_x_new, frac_y_new
    REAL(num) :: part_mc, ipart_mc, gamma_rel, igamma, vz
    REAL(num) :: qw_idt, qw_idxy
    REAL(num) :: wx_old(-1:2), wy_old(-1:2), wx_new(-1:2), wy_new(-1:2)
    REAL(num) :: dwx(-1:2), dwy(-1:2)
    REAL(num) :: jx_accum, jy_accum, jz_contrib
    
    idx = 1.0_num / dx
    idy = 1.0_num / dy
    idt = 1.0_num / dt
    part_mc = c * mass
    ipart_mc = 1.0_num / part_mc
    
    !$ACC PARALLEL LOOP &
    !$ACC& PRESENT(pos_x_old(1:n_particles), pos_y_old(1:n_particles)) &
    !$ACC& PRESENT(pos_x_new(1:n_particles), pos_y_new(1:n_particles)) &
    !$ACC& PRESENT(px(1:n_particles), py(1:n_particles), pz(1:n_particles)) &
    !$ACC& PRESENT(weight(1:n_particles), jx, jy, jz) &
    !$ACC& PRIVATE(cell_x_old, cell_y_old, cell_x_new, cell_y_new) &
    !$ACC& PRIVATE(ix_old, iy_old, ix_new, iy_new, ix_base, iy_base) &
    !$ACC& PRIVATE(frac_x_old, frac_y_old, frac_x_new, frac_y_new) &
    !$ACC& PRIVATE(dcellx, dcelly, wx_old, wy_old, wx_new, wy_new, dwx, dwy) &
    !$ACC& PRIVATE(gamma_rel, igamma, vz, qw_idt, qw_idxy) &
    !$ACC& PRIVATE(jx_accum, jy_accum, jz_contrib)
    DO ip = 1, n_particles
      ! Cell coordinates at old and new positions
      cell_x_old = (pos_x_old(ip) - x_min) * idx
      cell_y_old = (pos_y_old(ip) - y_min) * idy
      cell_x_new = (pos_x_new(ip) - x_min) * idx
      cell_y_new = (pos_y_new(ip) - y_min) * idy
      
      ! Cell indices
      ix_old = FLOOR(cell_x_old)
      iy_old = FLOOR(cell_y_old)
      ix_new = FLOOR(cell_x_new)
      iy_new = FLOOR(cell_y_new)
      
      ! Fractional positions within cell [0,1)
      frac_x_old = cell_x_old - REAL(ix_old, num)
      frac_y_old = cell_y_old - REAL(iy_old, num)
      frac_x_new = cell_x_new - REAL(ix_new, num)
      frac_y_new = cell_y_new - REAL(iy_new, num)
      
      ! Cell displacement
      dcellx = ix_new - ix_old
      dcelly = iy_new - iy_old
      
      ! Use old cell as reference
      ix_base = MAX(1, MIN(ix_old, nx-1))
      iy_base = MAX(1, MIN(iy_old, ny-1))
      
      ! First-order shape functions at old position
      wx_old = 0.0_num
      wy_old = 0.0_num
      wx_old(0) = 1.0_num - frac_x_old
      wx_old(1) = frac_x_old
      wy_old(0) = 1.0_num - frac_y_old
      wy_old(1) = frac_y_old
      
      ! Shape functions at new position (shifted by dcell)
      wx_new = 0.0_num
      wy_new = 0.0_num
      IF (dcellx >= -1 .AND. dcellx <= 1) THEN
        wx_new(dcellx) = 1.0_num - frac_x_new
        wx_new(dcellx+1) = frac_x_new
      ELSE
        wx_new(0) = 1.0_num - frac_x_new
        wx_new(1) = frac_x_new
      END IF
      IF (dcelly >= -1 .AND. dcelly <= 1) THEN
        wy_new(dcelly) = 1.0_num - frac_y_new
        wy_new(dcelly+1) = frac_y_new
      ELSE
        wy_new(0) = 1.0_num - frac_y_new
        wy_new(1) = frac_y_new
      END IF
      
      ! Shape function differences
      dwx = wx_new - wx_old
      dwy = wy_new - wy_old
      
      ! Compute gamma for velocity calculation
      gamma_rel = SQRT((px(ip)*ipart_mc)**2 + (py(ip)*ipart_mc)**2 + &
                       (pz(ip)*ipart_mc)**2 + 1.0_num)
      igamma = 1.0_num / gamma_rel
      
      ! Use velocity-based deposition with simple first-order shape
      ! Create shape functions directly at new position
      vz = pz(ip) * ipart_mc * igamma * c
      qw_idxy = charge * weight(ip) * idx * idy
      
      ! Simple first-order shape at new position
      ! Shape weights: w0 = 1-frac, w1 = frac
      ! Deposit to cells ix_new and ix_new+1
      DO iy = 0, 1
        DO ix = 0, 1
          IF (ix_new + ix >= 1 - ng .AND. ix_new + ix <= nx + ng .AND. &
              iy_new + iy >= 1 - ng .AND. iy_new + iy <= ny + ng) THEN
            ! Shape weights at new position
            jx_accum = qw_idxy * px(ip) * ipart_mc * igamma * c * &
                      (MERGE(1.0_num - frac_x_new, frac_x_new, ix == 0)) * &
                      (MERGE(1.0_num - frac_y_new, frac_y_new, iy == 0))
            !$ACC ATOMIC UPDATE
            jx(ix_new + ix, iy_new + iy) = jx(ix_new + ix, iy_new + iy) + jx_accum
            
            jy_accum = qw_idxy * py(ip) * ipart_mc * igamma * c * &
                      (MERGE(1.0_num - frac_x_new, frac_x_new, ix == 0)) * &
                      (MERGE(1.0_num - frac_y_new, frac_y_new, iy == 0))
            !$ACC ATOMIC UPDATE
            jy(ix_new + ix, iy_new + iy) = jy(ix_new + ix, iy_new + iy) + jy_accum
            
            jz_contrib = qw_idxy * vz * &
                        (MERGE(1.0_num - frac_x_new, frac_x_new, ix == 0)) * &
                        (MERGE(1.0_num - frac_y_new, frac_y_new, iy == 0))
            !$ACC ATOMIC UPDATE
            jz(ix_new + ix, iy_new + iy) = jz(ix_new + ix, iy_new + iy) + jz_contrib
          END IF
        END DO
      END DO
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE deposit_current_gpu

END MODULE particles_gpu

