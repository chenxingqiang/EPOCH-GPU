! Copyright (C) 2025 EPOCH-GPU Development Team
!
! @file particles_gpu.F90
! @brief GPU-accelerated particle pusher for EPOCH1D
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
  
  PUBLIC :: push_particles_gpu_1d
  PUBLIC :: interpolate_fields_gpu_1d
  PUBLIC :: deposit_current_gpu_1d
  
  ! Generic interface for epoch_gpu_interface compatibility
  PUBLIC :: push_particles_gpu
  PUBLIC :: interpolate_fields_to_particles_gpu
  PUBLIC :: deposit_current_gpu
  
  INTERFACE push_particles_gpu
    MODULE PROCEDURE push_particles_gpu_1d
  END INTERFACE
  
  INTERFACE interpolate_fields_to_particles_gpu
    MODULE PROCEDURE interpolate_fields_gpu_1d
  END INTERFACE
  
  INTERFACE deposit_current_gpu
    MODULE PROCEDURE deposit_current_gpu_1d
  END INTERFACE

CONTAINS

  !---------------------------------------------------------------------------
  !> @brief GPU-accelerated Boris particle pusher for 1D
  !---------------------------------------------------------------------------
  SUBROUTINE push_particles_gpu_1d(n_particles, pos_x, px, py, pz, &
                                    ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                                    charge, mass, dt)
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(INOUT) :: pos_x(:)
    REAL(num), INTENT(INOUT) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), INTENT(IN) :: bx_p(:), by_p(:), bz_p(:)
    REAL(num), INTENT(IN) :: charge, mass, dt
    
    INTEGER(i8) :: ip
    REAL(num) :: part_mc, ipart_mc, dtco2, cmratio, ccmratio
    REAL(num) :: gamma_rel, root
    REAL(num) :: uxm, uym, uzm, uxp, uyp, uzp
    REAL(num) :: taux, tauy, tauz, tau
    REAL(num) :: part_u2, delta_x
    
    part_mc = c * mass
    ipart_mc = 1.0_num / part_mc
    dtco2 = c * dt * 0.5_num
    cmratio = charge * dt / (2.0_num * part_mc)
    ccmratio = c * cmratio
    
    !$ACC PARALLEL LOOP &
    !$ACC& PRESENT(pos_x(1:n_particles), px(1:n_particles), py(1:n_particles), pz(1:n_particles)) &
    !$ACC& PRESENT(ex_p(1:n_particles), ey_p(1:n_particles), ez_p(1:n_particles)) &
    !$ACC& PRESENT(bx_p(1:n_particles), by_p(1:n_particles), bz_p(1:n_particles)) &
    !$ACC& PRIVATE(gamma_rel, root, uxm, uym, uzm, uxp, uyp, uzp, taux, tauy, tauz, tau, part_u2, delta_x)
    DO ip = 1, n_particles
      ! Calculate gamma from momentum
      gamma_rel = SQRT((px(ip)*ipart_mc)**2 + (py(ip)*ipart_mc)**2 + &
                       (pz(ip)*ipart_mc)**2 + 1.0_num)
      root = dtco2 / gamma_rel
      
      ! Half position update (first order)
      pos_x(ip) = pos_x(ip) + px(ip) * ipart_mc * root
      
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
      pos_x(ip) = pos_x(ip) + delta_x
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE push_particles_gpu_1d


  !---------------------------------------------------------------------------
  !> @brief Interpolate fields to particle positions (1D, 1st order)
  !---------------------------------------------------------------------------
  SUBROUTINE interpolate_fields_gpu_1d(n_particles, pos_x, &
                                        ex, ey, ez, bx, by, bz, &
                                        ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                                        nx, ng, x_min, dx)
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: pos_x(:)
    REAL(num), INTENT(IN) :: ex(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: ey(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: ez(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: bx(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: by(1-ng:nx+ng)
    REAL(num), INTENT(IN) :: bz(1-ng:nx+ng)
    REAL(num), INTENT(OUT) :: ex_p(:), ey_p(:), ez_p(:)
    REAL(num), INTENT(OUT) :: bx_p(:), by_p(:), bz_p(:)
    INTEGER, INTENT(IN) :: nx, ng
    REAL(num), INTENT(IN) :: x_min, dx
    
    INTEGER(i8) :: ip
    INTEGER :: ix
    REAL(num) :: cell_x, gx0, gx1, idx
    
    idx = 1.0_num / dx
    
    !$ACC PARALLEL LOOP &
    !$ACC& PRESENT(pos_x(1:n_particles), ex, ey, ez, bx, by, bz) &
    !$ACC& PRESENT(ex_p(1:n_particles), ey_p(1:n_particles), ez_p(1:n_particles)) &
    !$ACC& PRESENT(bx_p(1:n_particles), by_p(1:n_particles), bz_p(1:n_particles)) &
    !$ACC& PRIVATE(cell_x, ix, gx0, gx1)
    DO ip = 1, n_particles
      ! Compute cell index and local coordinate
      cell_x = (pos_x(ip) - x_min) * idx
      ix = INT(cell_x)
      gx1 = cell_x - REAL(ix, num)
      gx0 = 1.0_num - gx1
      
      ! Ensure valid index
      ix = MAX(1, MIN(ix, nx-1))
      
      ! Linear interpolation
      ex_p(ip) = gx0 * ex(ix) + gx1 * ex(ix+1)
      ey_p(ip) = gx0 * ey(ix) + gx1 * ey(ix+1)
      ez_p(ip) = gx0 * ez(ix) + gx1 * ez(ix+1)
      bx_p(ip) = gx0 * bx(ix) + gx1 * bx(ix+1)
      by_p(ip) = gx0 * by(ix) + gx1 * by(ix+1)
      bz_p(ip) = gx0 * bz(ix) + gx1 * bz(ix+1)
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE interpolate_fields_gpu_1d


  !---------------------------------------------------------------------------
  !> @brief Deposit current to grid (1D, 1st order with atomics)
  !---------------------------------------------------------------------------
  SUBROUTINE deposit_current_gpu_1d(n_particles, pos_x, px, py, pz, weight, &
                                     jx, jy, jz, nx, ng, x_min, dx, mass, charge)
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: pos_x(:)
    REAL(num), INTENT(IN) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: weight(:)
    REAL(num), INTENT(INOUT) :: jx(1-ng:nx+ng)
    REAL(num), INTENT(INOUT) :: jy(1-ng:nx+ng)
    REAL(num), INTENT(INOUT) :: jz(1-ng:nx+ng)
    INTEGER, INTENT(IN) :: nx, ng
    REAL(num), INTENT(IN) :: x_min, dx, mass, charge
    
    INTEGER(i8) :: ip
    INTEGER :: ix
    REAL(num) :: cell_x, gx0, gx1, idx
    REAL(num) :: gamma_rel, igamma, part_mc, ipart_mc
    REAL(num) :: vx, vy, vz, factor
    REAL(num) :: jx_contrib0, jx_contrib1
    REAL(num) :: jy_contrib0, jy_contrib1
    REAL(num) :: jz_contrib0, jz_contrib1
    
    idx = 1.0_num / dx
    part_mc = c * mass
    ipart_mc = 1.0_num / part_mc
    ! Current = charge * weight * velocity / cell_volume
    factor = charge / dx
    
    !$ACC PARALLEL LOOP &
    !$ACC& PRESENT(pos_x(1:n_particles), px(1:n_particles), py(1:n_particles), pz(1:n_particles)) &
    !$ACC& PRESENT(weight(1:n_particles), jx, jy, jz) &
    !$ACC& PRIVATE(cell_x, ix, gx0, gx1, gamma_rel, igamma, vx, vy, vz, &
    !$ACC& jx_contrib0, jx_contrib1, jy_contrib0, jy_contrib1, jz_contrib0, jz_contrib1)
    DO ip = 1, n_particles
      ! Compute velocity
      gamma_rel = SQRT((px(ip)*ipart_mc)**2 + (py(ip)*ipart_mc)**2 + &
                       (pz(ip)*ipart_mc)**2 + 1.0_num)
      igamma = 1.0_num / gamma_rel
      vx = px(ip) * ipart_mc * igamma * c
      vy = py(ip) * ipart_mc * igamma * c
      vz = pz(ip) * ipart_mc * igamma * c
      
      ! Compute cell index
      cell_x = (pos_x(ip) - x_min) * idx
      ix = INT(cell_x)
      gx1 = cell_x - REAL(ix, num)
      gx0 = 1.0_num - gx1
      
      ix = MAX(1, MIN(ix, nx-1))
      
      ! Compute current contributions
      jx_contrib0 = weight(ip) * vx * gx0 * factor
      jx_contrib1 = weight(ip) * vx * gx1 * factor
      jy_contrib0 = weight(ip) * vy * gx0 * factor
      jy_contrib1 = weight(ip) * vy * gx1 * factor
      jz_contrib0 = weight(ip) * vz * gx0 * factor
      jz_contrib1 = weight(ip) * vz * gx1 * factor
      
      ! Atomic updates
      !$ACC ATOMIC UPDATE
      jx(ix) = jx(ix) + jx_contrib0
      !$ACC ATOMIC UPDATE
      jx(ix+1) = jx(ix+1) + jx_contrib1
      !$ACC ATOMIC UPDATE
      jy(ix) = jy(ix) + jy_contrib0
      !$ACC ATOMIC UPDATE
      jy(ix+1) = jy(ix+1) + jy_contrib1
      !$ACC ATOMIC UPDATE
      jz(ix) = jz(ix) + jz_contrib0
      !$ACC ATOMIC UPDATE
      jz(ix+1) = jz(ix+1) + jz_contrib1
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE deposit_current_gpu_1d

END MODULE particles_gpu

