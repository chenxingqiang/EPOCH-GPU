! Copyright (C) 2024 EPOCH-GPU Development Team
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
    
    !$ACC PARALLEL LOOP PRESENT(pos_x, pos_y, pos_z, px, py, pz, &
    !$ACC& ex_p, ey_p, ez_p, bx_p, by_p, bz_p) &
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
    
    !$ACC PARALLEL LOOP PRESENT(pos_x, pos_y, pos_z, &
    !$ACC& ex, ey, ez, bx, by, bz, ex_p, ey_p, ez_p, bx_p, by_p, bz_p) &
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
  !> @brief Deposit current to grid (3D, 1st order with atomics)
  !---------------------------------------------------------------------------
  SUBROUTINE deposit_current_gpu_3d(n_particles, pos_x, pos_y, pos_z, &
                                     px, py, pz, weight, &
                                     jx, jy, jz, nx, ny, nz, ng, &
                                     x_min, y_min, z_min, dx, dy, dz, mass)
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: pos_x(:), pos_y(:), pos_z(:)
    REAL(num), INTENT(IN) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: weight(:)
    REAL(num), INTENT(INOUT) :: jx(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(INOUT) :: jy(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    REAL(num), INTENT(INOUT) :: jz(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng)
    INTEGER, INTENT(IN) :: nx, ny, nz, ng
    REAL(num), INTENT(IN) :: x_min, y_min, z_min, dx, dy, dz, mass
    
    INTEGER(i8) :: ip
    INTEGER :: ix, iy, iz, i, j, k
    REAL(num) :: cell_x, cell_y, cell_z
    REAL(num) :: gx0, gx1, gy0, gy1, gz0, gz1
    REAL(num) :: idx, idy, idz
    REAL(num) :: gamma_rel, igamma, part_mc, ipart_mc
    REAL(num) :: vx, vy, vz, factor
    REAL(num) :: wx(0:1), wy(0:1), wz(0:1)
    REAL(num) :: jx_contrib, jy_contrib, jz_contrib
    
    idx = 1.0_num / dx
    idy = 1.0_num / dy
    idz = 1.0_num / dz
    part_mc = c * mass
    ipart_mc = 1.0_num / part_mc
    factor = 1.0_num / (dx * dy * dz)
    
    !$ACC PARALLEL LOOP PRESENT(pos_x, pos_y, pos_z, px, py, pz, weight, jx, jy, jz) &
    !$ACC& PRIVATE(cell_x, cell_y, cell_z, ix, iy, iz, gx0, gx1, gy0, gy1, gz0, gz1, &
    !$ACC& gamma_rel, igamma, vx, vy, vz, wx, wy, wz, jx_contrib, jy_contrib, jz_contrib)
    DO ip = 1, n_particles
      ! Compute velocity
      gamma_rel = SQRT((px(ip)*ipart_mc)**2 + (py(ip)*ipart_mc)**2 + &
                       (pz(ip)*ipart_mc)**2 + 1.0_num)
      igamma = 1.0_num / gamma_rel
      vx = px(ip) * ipart_mc * igamma * c
      vy = py(ip) * ipart_mc * igamma * c
      vz = pz(ip) * ipart_mc * igamma * c
      
      ! Compute cell indices
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
      
      ix = MAX(1, MIN(ix, nx-1))
      iy = MAX(1, MIN(iy, ny-1))
      iz = MAX(1, MIN(iz, nz-1))
      
      wx(0) = gx0; wx(1) = gx1
      wy(0) = gy0; wy(1) = gy1
      wz(0) = gz0; wz(1) = gz1
      
      ! Deposit current with atomics
      DO k = 0, 1
        DO j = 0, 1
          DO i = 0, 1
            jx_contrib = weight(ip) * vx * wx(i) * wy(j) * wz(k) * factor
            jy_contrib = weight(ip) * vy * wx(i) * wy(j) * wz(k) * factor
            jz_contrib = weight(ip) * vz * wx(i) * wy(j) * wz(k) * factor
            
            !$ACC ATOMIC UPDATE
            jx(ix+i, iy+j, iz+k) = jx(ix+i, iy+j, iz+k) + jx_contrib
            !$ACC ATOMIC UPDATE
            jy(ix+i, iy+j, iz+k) = jy(ix+i, iy+j, iz+k) + jy_contrib
            !$ACC ATOMIC UPDATE
            jz(ix+i, iy+j, iz+k) = jz(ix+i, iy+j, iz+k) + jz_contrib
          END DO
        END DO
      END DO
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE deposit_current_gpu_3d

END MODULE particles_gpu

