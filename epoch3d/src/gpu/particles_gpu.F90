! Copyright (C) 2025 EPOCH-GPU Development Team
!
! @file particles_gpu.F90
! @brief GPU-accelerated particle pusher for EPOCH3D
! @details Implements Boris algorithm on GPU using OpenACC

MODULE particles_gpu

#ifdef GPU_ENABLED
  USE cudafor
#endif
  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT32, INT64, REAL64
  
  IMPLICIT NONE
  PRIVATE
  
  INTEGER, PARAMETER :: num = REAL64
  INTEGER, PARAMETER :: i8 = INT64
  
  ! Speed of light
  REAL(num), PARAMETER :: c = 299792458.0_num
  
  PUBLIC :: push_particles_gpu_3d
  PUBLIC :: interpolate_fields_gpu_3d
  PUBLIC :: deposit_current_gpu_3d
  
  ! Generic interface for epoch_gpu_interface compatibility
  PUBLIC :: push_particles_gpu
  PUBLIC :: interpolate_fields_to_particles_gpu
  PUBLIC :: deposit_current_gpu
  
  INTERFACE push_particles_gpu
    MODULE PROCEDURE push_particles_gpu_3d
  END INTERFACE
  
  INTERFACE interpolate_fields_to_particles_gpu
    MODULE PROCEDURE interpolate_fields_gpu_3d
  END INTERFACE
  
  INTERFACE deposit_current_gpu
    MODULE PROCEDURE deposit_current_gpu_3d
  END INTERFACE

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated Boris particle pusher for 3D
  !---------------------------------------------------------------------------
  SUBROUTINE push_particles_gpu_3d(n_particles, pos_x, pos_y, pos_z, &
                                    px, py, pz, &
                                    ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                                    charge, mass, dt)
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(INOUT) :: pos_x(:), pos_y(:), pos_z(:)
    REAL(num), INTENT(INOUT) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), INTENT(IN) :: bx_p(:), by_p(:), bz_p(:)
    REAL(num), INTENT(IN) :: charge, mass, dt
    
    INTEGER(i8) :: ip
    REAL(num) :: part_mc, ipart_mc, dtco2, cmratio, ccmratio
    REAL(num) :: gamma_rel, root
    REAL(num) :: uxm, uym, uzm, uxp, uyp, uzp
    REAL(num) :: taux, tauy, tauz, tau
    REAL(num) :: part_u2, delta_x, delta_y, delta_z
    
    part_mc = c * mass
    ipart_mc = 1.0_num / part_mc
    dtco2 = c * dt * 0.5_num
    cmratio = charge * dt / (2.0_num * part_mc)
    ccmratio = c * cmratio
    
    !$ACC PARALLEL LOOP &
    !$ACC& PRESENT(pos_x(1:n_particles), pos_y(1:n_particles), pos_z(1:n_particles)) &
    !$ACC& PRESENT(px(1:n_particles), py(1:n_particles), pz(1:n_particles)) &
    !$ACC& PRESENT(ex_p(1:n_particles), ey_p(1:n_particles), ez_p(1:n_particles)) &
    !$ACC& PRESENT(bx_p(1:n_particles), by_p(1:n_particles), bz_p(1:n_particles)) &
    !$ACC& PRIVATE(gamma_rel, root, uxm, uym, uzm, uxp, uyp, uzp, &
    !$ACC& taux, tauy, tauz, tau, part_u2, delta_x, delta_y, delta_z)
    DO ip = 1, n_particles
      ! Calculate gamma from momentum
      gamma_rel = SQRT((px(ip)*ipart_mc)**2 + (py(ip)*ipart_mc)**2 + &
                       (pz(ip)*ipart_mc)**2 + 1.0_num)
      root = dtco2 / gamma_rel
      
      ! Half position update (first order)
      pos_x(ip) = pos_x(ip) + px(ip) * ipart_mc * root
      pos_y(ip) = pos_y(ip) + py(ip) * ipart_mc * root
      pos_z(ip) = pos_z(ip) + pz(ip) * ipart_mc * root
      
      ! Electric field half-acceleration
      uxm = px(ip) * ipart_mc + cmratio * ex_p(ip)
      uym = py(ip) * ipart_mc + cmratio * ey_p(ip)
      uzm = pz(ip) * ipart_mc + cmratio * ez_p(ip)
      
      ! Boris rotation
      gamma_rel = SQRT(uxm**2 + uym**2 + uzm**2 + 1.0_num)
      root = ccmratio / gamma_rel
      
      taux = bx_p(ip) * root
      tauy = by_p(ip) * root
      tauz = bz_p(ip) * root
      
      tau = 1.0_num / (1.0_num + taux**2 + tauy**2 + tauz**2)
      
      uxp = ((1.0_num + taux**2 - tauy**2 - tauz**2) * uxm &
          + 2.0_num * ((taux * tauy + tauz) * uym &
          + (taux * tauz - tauy) * uzm)) * tau
      uyp = ((1.0_num - taux**2 + tauy**2 - tauz**2) * uym &
          + 2.0_num * ((tauy * tauz + taux) * uzm &
          + (tauy * taux - tauz) * uxm)) * tau
      uzp = ((1.0_num - taux**2 - tauy**2 + tauz**2) * uzm &
          + 2.0_num * ((tauz * taux + tauy) * uxm &
          + (tauz * tauy - taux) * uym)) * tau
      
      ! Electric field half-acceleration (second half)
      px(ip) = part_mc * (uxp + cmratio * ex_p(ip))
      py(ip) = part_mc * (uyp + cmratio * ey_p(ip))
      pz(ip) = part_mc * (uzp + cmratio * ez_p(ip))
      
      ! Second half position update
      part_u2 = (px(ip)*ipart_mc)**2 + (py(ip)*ipart_mc)**2 + (pz(ip)*ipart_mc)**2
      gamma_rel = SQRT(part_u2 + 1.0_num)
      root = dtco2 / gamma_rel
      
      delta_x = px(ip) * ipart_mc * root
      delta_y = py(ip) * ipart_mc * root
      delta_z = pz(ip) * ipart_mc * root
      
      pos_x(ip) = pos_x(ip) + delta_x
      pos_y(ip) = pos_y(ip) + delta_y
      pos_z(ip) = pos_z(ip) + delta_z
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE push_particles_gpu_3d


  !---------------------------------------------------------------------------
  !> @brief Interpolate fields to particle positions (3D, 1st order)
  !---------------------------------------------------------------------------
  SUBROUTINE interpolate_fields_gpu_3d(n_particles, pos_x, pos_y, pos_z, &
                                        ex, ey, ez, bx, by, bz, &
                                        ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                                        nx, ny, nz, ng, x_min, y_min, z_min, &
                                        dx, dy, dz)
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: pos_x(:), pos_y(:), pos_z(:)
    REAL(num), INTENT(IN) :: ex(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: ey(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: ez(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: bx(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: by(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(IN) :: bz(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(OUT) :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), INTENT(OUT) :: bx_p(:), by_p(:), bz_p(:)
    INTEGER, INTENT(IN) :: nx, ny, nz, ng
    REAL(num), INTENT(IN) :: x_min, y_min, z_min, dx, dy, dz
    
    INTEGER(i8) :: ip
    INTEGER :: ix, iy, iz
    REAL(num) :: cell_x, cell_y, cell_z
    REAL(num) :: gx0, gx1, gy0, gy1, gz0, gz1
    REAL(num) :: idx, idy, idz
    REAL(num) :: w000, w001, w010, w011, w100, w101, w110, w111
    
    idx = 1.0_num / dx
    idy = 1.0_num / dy
    idz = 1.0_num / dz
    
    !$ACC PARALLEL LOOP &
    !$ACC& PRESENT(pos_x(1:n_particles), pos_y(1:n_particles), pos_z(1:n_particles)) &
    !$ACC& PRESENT(ex, ey, ez, bx, by, bz) &
    !$ACC& PRESENT(ex_p(1:n_particles), ey_p(1:n_particles), ez_p(1:n_particles)) &
    !$ACC& PRESENT(bx_p(1:n_particles), by_p(1:n_particles), bz_p(1:n_particles)) &
    !$ACC& PRIVATE(cell_x, cell_y, cell_z, ix, iy, iz, &
    !$ACC& gx0, gx1, gy0, gy1, gz0, gz1, w000, w001, w010, w011, w100, w101, w110, w111)
    DO ip = 1, n_particles
      ! Compute cell indices and local coordinates
      cell_x = (pos_x(ip) - x_min) * idx
      cell_y = (pos_y(ip) - y_min) * idy
      cell_z = (pos_z(ip) - z_min) * idz
      
      ix = INT(cell_x)
      iy = INT(cell_y)
      iz = INT(cell_z)
      
      gx1 = cell_x - REAL(ix, num)
      gy1 = cell_y - REAL(iy, num)
      gz1 = cell_z - REAL(iz, num)
      gx0 = 1.0_num - gx1
      gy0 = 1.0_num - gy1
      gz0 = 1.0_num - gz1
      
      ! Clamp indices
      ix = MAX(1, MIN(ix, nx-1))
      iy = MAX(1, MIN(iy, ny-1))
      iz = MAX(1, MIN(iz, nz-1))
      
      ! Trilinear interpolation weights
      w000 = gx0 * gy0 * gz0
      w001 = gx0 * gy0 * gz1
      w010 = gx0 * gy1 * gz0
      w011 = gx0 * gy1 * gz1
      w100 = gx1 * gy0 * gz0
      w101 = gx1 * gy0 * gz1
      w110 = gx1 * gy1 * gz0
      w111 = gx1 * gy1 * gz1
      
      ! Interpolate E-field
      ex_p(ip) = w000 * ex(ix, iy, iz) + w001 * ex(ix, iy, iz+1) &
               + w010 * ex(ix, iy+1, iz) + w011 * ex(ix, iy+1, iz+1) &
               + w100 * ex(ix+1, iy, iz) + w101 * ex(ix+1, iy, iz+1) &
               + w110 * ex(ix+1, iy+1, iz) + w111 * ex(ix+1, iy+1, iz+1)
      
      ey_p(ip) = w000 * ey(ix, iy, iz) + w001 * ey(ix, iy, iz+1) &
               + w010 * ey(ix, iy+1, iz) + w011 * ey(ix, iy+1, iz+1) &
               + w100 * ey(ix+1, iy, iz) + w101 * ey(ix+1, iy, iz+1) &
               + w110 * ey(ix+1, iy+1, iz) + w111 * ey(ix+1, iy+1, iz+1)
      
      ez_p(ip) = w000 * ez(ix, iy, iz) + w001 * ez(ix, iy, iz+1) &
               + w010 * ez(ix, iy+1, iz) + w011 * ez(ix, iy+1, iz+1) &
               + w100 * ez(ix+1, iy, iz) + w101 * ez(ix+1, iy, iz+1) &
               + w110 * ez(ix+1, iy+1, iz) + w111 * ez(ix+1, iy+1, iz+1)
      
      ! Interpolate B-field
      bx_p(ip) = w000 * bx(ix, iy, iz) + w001 * bx(ix, iy, iz+1) &
               + w010 * bx(ix, iy+1, iz) + w011 * bx(ix, iy+1, iz+1) &
               + w100 * bx(ix+1, iy, iz) + w101 * bx(ix+1, iy, iz+1) &
               + w110 * bx(ix+1, iy+1, iz) + w111 * bx(ix+1, iy+1, iz+1)
      
      by_p(ip) = w000 * by(ix, iy, iz) + w001 * by(ix, iy, iz+1) &
               + w010 * by(ix, iy+1, iz) + w011 * by(ix, iy+1, iz+1) &
               + w100 * by(ix+1, iy, iz) + w101 * by(ix+1, iy, iz+1) &
               + w110 * by(ix+1, iy+1, iz) + w111 * by(ix+1, iy+1, iz+1)
      
      bz_p(ip) = w000 * bz(ix, iy, iz) + w001 * bz(ix, iy, iz+1) &
               + w010 * bz(ix, iy+1, iz) + w011 * bz(ix, iy+1, iz+1) &
               + w100 * bz(ix+1, iy, iz) + w101 * bz(ix+1, iy, iz+1) &
               + w110 * bz(ix+1, iy+1, iz) + w111 * bz(ix+1, iy+1, iz+1)
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE interpolate_fields_gpu_3d


  !---------------------------------------------------------------------------
  !> @brief Deposit current using Esirkepov charge-conserving method (3D)
  !>
  !> Implements the Esirkepov current deposition scheme ensuring:
  !>   div(J) = -d(rho)/dt (exact charge conservation)
  !>
  !> Reference: T.Zh. Esirkepov, CPC 135 (2001) 144-153
  !---------------------------------------------------------------------------
  SUBROUTINE deposit_current_gpu_3d(n_particles, &
                                     pos_x_old, pos_y_old, pos_z_old, &
                                     pos_x_new, pos_y_new, pos_z_new, &
                                     px, py, pz, weight, charge, mass, &
                                     jx, jy, jz, nx, ny, nz, ng, &
                                     x_min, y_min, z_min, dx, dy, dz, dt)
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: pos_x_old(:), pos_y_old(:), pos_z_old(:)
    REAL(num), INTENT(IN) :: pos_x_new(:), pos_y_new(:), pos_z_new(:)
    REAL(num), INTENT(IN) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: weight(:)
    REAL(num), INTENT(IN) :: charge, mass
    REAL(num), INTENT(INOUT) :: jx(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(INOUT) :: jy(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(INOUT) :: jz(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    INTEGER, INTENT(IN) :: nx, ny, nz, ng
    REAL(num), INTENT(IN) :: x_min, y_min, z_min, dx, dy, dz, dt
    
    INTEGER(i8) :: ip
    INTEGER :: ix_old, iy_old, iz_old, ix_new, iy_new, iz_new
    INTEGER :: ix_base, iy_base, iz_base, ix, iy, iz
    INTEGER :: dcellx, dcelly, dcellz
    REAL(num) :: idx, idy, idz, idt
    REAL(num) :: cell_x_old, cell_y_old, cell_z_old
    REAL(num) :: cell_x_new, cell_y_new, cell_z_new
    REAL(num) :: frac_x_old, frac_y_old, frac_z_old
    REAL(num) :: frac_x_new, frac_y_new, frac_z_new
    REAL(num) :: wx_old(-1:1), wy_old(-1:1), wz_old(-1:1)
    REAL(num) :: wx_new(-1:1), wy_new(-1:1), wz_new(-1:1)
    REAL(num) :: dwx(-1:1), dwy(-1:1), dwz(-1:1)
    REAL(num) :: qw_idt, jx_accum, jy_accum, jz_accum
    
    idx = 1.0_num / dx
    idy = 1.0_num / dy
    idz = 1.0_num / dz
    idt = 1.0_num / dt
    
    !$ACC PARALLEL LOOP &
    !$ACC& PRESENT(pos_x_old(1:n_particles), pos_y_old(1:n_particles), pos_z_old(1:n_particles)) &
    !$ACC& PRESENT(pos_x_new(1:n_particles), pos_y_new(1:n_particles), pos_z_new(1:n_particles)) &
    !$ACC& PRESENT(px(1:n_particles), py(1:n_particles), pz(1:n_particles)) &
    !$ACC& PRESENT(weight(1:n_particles), jx, jy, jz) &
    !$ACC& PRIVATE(cell_x_old, cell_y_old, cell_z_old, cell_x_new, cell_y_new, cell_z_new) &
    !$ACC& PRIVATE(ix_old, iy_old, iz_old, ix_new, iy_new, iz_new) &
    !$ACC& PRIVATE(ix_base, iy_base, iz_base, dcellx, dcelly, dcellz) &
    !$ACC& PRIVATE(frac_x_old, frac_y_old, frac_z_old, frac_x_new, frac_y_new, frac_z_new) &
    !$ACC& PRIVATE(wx_old, wy_old, wz_old, wx_new, wy_new, wz_new, dwx, dwy, dwz) &
    !$ACC& PRIVATE(qw_idt, jx_accum, jy_accum, jz_accum)
    DO ip = 1, n_particles
      ! Cell coordinates at old and new positions
      cell_x_old = (pos_x_old(ip) - x_min) * idx
      cell_y_old = (pos_y_old(ip) - y_min) * idy
      cell_z_old = (pos_z_old(ip) - z_min) * idz
      cell_x_new = (pos_x_new(ip) - x_min) * idx
      cell_y_new = (pos_y_new(ip) - y_min) * idy
      cell_z_new = (pos_z_new(ip) - z_min) * idz
      
      ! Cell indices
      ix_old = FLOOR(cell_x_old)
      iy_old = FLOOR(cell_y_old)
      iz_old = FLOOR(cell_z_old)
      ix_new = FLOOR(cell_x_new)
      iy_new = FLOOR(cell_y_new)
      iz_new = FLOOR(cell_z_new)
      
      ! Fractional positions
      frac_x_old = cell_x_old - REAL(ix_old, num)
      frac_y_old = cell_y_old - REAL(iy_old, num)
      frac_z_old = cell_z_old - REAL(iz_old, num)
      frac_x_new = cell_x_new - REAL(ix_new, num)
      frac_y_new = cell_y_new - REAL(iy_new, num)
      frac_z_new = cell_z_new - REAL(iz_new, num)
      
      ! Cell displacement
      dcellx = ix_new - ix_old
      dcelly = iy_new - iy_old
      dcellz = iz_new - iz_old
      
      ! Use old cell as reference
      ix_base = MAX(1, MIN(ix_old, nx-1))
      iy_base = MAX(1, MIN(iy_old, ny-1))
      iz_base = MAX(1, MIN(iz_old, nz-1))
      
      ! First-order shape functions at old position
      wx_old(-1) = 0.0_num; wx_old(0) = 1.0_num - frac_x_old; wx_old(1) = frac_x_old
      wy_old(-1) = 0.0_num; wy_old(0) = 1.0_num - frac_y_old; wy_old(1) = frac_y_old
      wz_old(-1) = 0.0_num; wz_old(0) = 1.0_num - frac_z_old; wz_old(1) = frac_z_old
      
      ! Shape functions at new position (shifted by dcell)
      wx_new = 0.0_num; wy_new = 0.0_num; wz_new = 0.0_num
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
      IF (dcellz >= -1 .AND. dcellz <= 1) THEN
        wz_new(dcellz) = 1.0_num - frac_z_new
        wz_new(dcellz+1) = frac_z_new
      ELSE
        wz_new(0) = 1.0_num - frac_z_new
        wz_new(1) = frac_z_new
      END IF
      
      ! Shape function differences
      dwx(-1) = wx_new(-1) - wx_old(-1)
      dwx(0) = wx_new(0) - wx_old(0)
      dwx(1) = wx_new(1) - wx_old(1)
      dwy(-1) = wy_new(-1) - wy_old(-1)
      dwy(0) = wy_new(0) - wy_old(0)
      dwy(1) = wy_new(1) - wy_old(1)
      dwz(-1) = wz_new(-1) - wz_old(-1)
      dwz(0) = wz_new(0) - wz_old(0)
      dwz(1) = wz_new(1) - wz_old(1)
      
      ! Current factor
      qw_idt = charge * weight(ip) * idt
      
      ! Esirkepov Jx deposition
      DO iz = -1, 1
        IF (iz_base + iz < 1 - ng .OR. iz_base + iz > nz + ng) CYCLE
        DO iy = -1, 1
          IF (iy_base + iy < 1 - ng .OR. iy_base + iy > ny + ng) CYCLE
          jx_accum = 0.0_num
          DO ix = -1, 1
            IF (ix_base + ix < 1 - ng .OR. ix_base + ix > nx + ng) CYCLE
            jx_accum = jx_accum - qw_idt * dwx(ix) * &
                       (wy_old(iy) + 0.5_num * dwy(iy)) * &
                       (wz_old(iz) + 0.5_num * dwz(iz))
            !$ACC ATOMIC UPDATE
            jx(ix_base+ix, iy_base+iy, iz_base+iz) = &
                jx(ix_base+ix, iy_base+iy, iz_base+iz) + jx_accum
          END DO
        END DO
      END DO
      
      ! Esirkepov Jy deposition
      DO iz = -1, 1
        IF (iz_base + iz < 1 - ng .OR. iz_base + iz > nz + ng) CYCLE
        DO ix = -1, 1
          IF (ix_base + ix < 1 - ng .OR. ix_base + ix > nx + ng) CYCLE
          jy_accum = 0.0_num
          DO iy = -1, 1
            IF (iy_base + iy < 1 - ng .OR. iy_base + iy > ny + ng) CYCLE
            jy_accum = jy_accum - qw_idt * dwy(iy) * &
                       (wx_old(ix) + 0.5_num * dwx(ix)) * &
                       (wz_old(iz) + 0.5_num * dwz(iz))
            !$ACC ATOMIC UPDATE
            jy(ix_base+ix, iy_base+iy, iz_base+iz) = &
                jy(ix_base+ix, iy_base+iy, iz_base+iz) + jy_accum
          END DO
        END DO
      END DO
      
      ! Esirkepov Jz deposition
      DO iy = -1, 1
        IF (iy_base + iy < 1 - ng .OR. iy_base + iy > ny + ng) CYCLE
        DO ix = -1, 1
          IF (ix_base + ix < 1 - ng .OR. ix_base + ix > nx + ng) CYCLE
          jz_accum = 0.0_num
          DO iz = -1, 1
            IF (iz_base + iz < 1 - ng .OR. iz_base + iz > nz + ng) CYCLE
            jz_accum = jz_accum - qw_idt * dwz(iz) * &
                       (wx_old(ix) + 0.5_num * dwx(ix)) * &
                       (wy_old(iy) + 0.5_num * dwy(iy))
            !$ACC ATOMIC UPDATE
            jz(ix_base+ix, iy_base+iy, iz_base+iz) = &
                jz(ix_base+ix, iy_base+iy, iz_base+iz) + jz_accum
          END DO
        END DO
      END DO
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE deposit_current_gpu_3d

END MODULE particles_gpu

