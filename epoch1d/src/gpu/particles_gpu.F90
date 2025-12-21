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
  !> @brief Deposit current using Esirkepov charge-conserving method (1D)
  !>
  !> Implements the Esirkepov current deposition scheme ensuring:
  !>   div(J) = -d(rho)/dt (exact charge conservation)
  !>
  !> Reference: T.Zh. Esirkepov, CPC 135 (2001) 144-153
  !>
  !> @param[in]    n_particles  Number of particles
  !> @param[in]    pos_x_old    Particle positions at t^n (before push)
  !> @param[in]    pos_x_new    Particle positions at t^{n+1} (after push)
  !> @param[in]    px,py,pz     Particle momenta
  !> @param[in]    weight       Particle weights
  !> @param[inout] jx,jy,jz     Current density arrays
  !> @param[in]    nx,ng        Grid size and ghost cells
  !> @param[in]    x_min,dx,dt  Grid parameters and time step
  !> @param[in]    mass,charge  Particle mass and charge (SI units)
  !---------------------------------------------------------------------------
  SUBROUTINE deposit_current_gpu_1d(n_particles, pos_x_old, pos_x_new, &
                                     px, py, pz, weight, &
                                     jx, jy, jz, nx, ng, x_min, dx, dt, mass, charge)
    INTEGER(i8), INTENT(IN) :: n_particles
    REAL(num), INTENT(IN) :: pos_x_old(:), pos_x_new(:)
    REAL(num), INTENT(IN) :: px(:), py(:), pz(:)
    REAL(num), INTENT(IN) :: weight(:)
    REAL(num), INTENT(INOUT) :: jx(1-ng:nx+ng)
    REAL(num), INTENT(INOUT) :: jy(1-ng:nx+ng)
    REAL(num), INTENT(INOUT) :: jz(1-ng:nx+ng)
    INTEGER, INTENT(IN) :: nx, ng
    REAL(num), INTENT(IN) :: x_min, dx, dt, mass, charge
    
    INTEGER(i8) :: ip
    INTEGER :: ix_old, ix_new, ix_base, ix
    REAL(num) :: cell_x_old, cell_x_new, idx, idt
    REAL(num) :: frac_old, frac_new
    REAL(num) :: gamma_rel, igamma, part_mc, ipart_mc
    REAL(num) :: vy, vz
    REAL(num) :: wold(-1:2), wnew(-1:2), dw(-1:2)
    REAL(num) :: jx_accum, jy_contrib, jz_contrib
    REAL(num) :: qw_idt, qw_idx
    INTEGER :: dcell
    
    idx = 1.0_num / dx
    idt = 1.0_num / dt
    part_mc = c * mass
    ipart_mc = 1.0_num / part_mc
    
    !$ACC PARALLEL LOOP &
    !$ACC& PRESENT(pos_x_old(1:n_particles), pos_x_new(1:n_particles)) &
    !$ACC& PRESENT(px(1:n_particles), py(1:n_particles), pz(1:n_particles)) &
    !$ACC& PRESENT(weight(1:n_particles), jx, jy, jz) &
    !$ACC& PRIVATE(cell_x_old, cell_x_new, ix_old, ix_new, frac_old, frac_new) &
    !$ACC& PRIVATE(dcell, ix_base, wold, wnew, dw) &
    !$ACC& PRIVATE(gamma_rel, igamma, vy, vz, qw_idt, qw_idx) &
    !$ACC& PRIVATE(jx_accum, jy_contrib, jz_contrib)
    DO ip = 1, n_particles
      ! Cell coordinates at old and new positions
      cell_x_old = (pos_x_old(ip) - x_min) * idx
      cell_x_new = (pos_x_new(ip) - x_min) * idx
      
      ! Cell indices (using 1-based indexing like interpolation)
      ix_old = INT(cell_x_old) + 1
      ix_new = INT(cell_x_new) + 1
      
      ! Fractional positions within cell [0,1)
      frac_old = cell_x_old - REAL(ix_old - 1, num)
      frac_new = cell_x_new - REAL(ix_new - 1, num)
      
      ! Cell displacement (should be -1, 0, or +1 for CFL-stable simulations)
      dcell = ix_new - ix_old
      
      ! Use old cell as reference (1-based, clamped to physics cells)
      ix_base = MAX(1, MIN(ix_old, nx))
      
      ! First-order (linear) shape functions at old position
      ! W(-1)=0, W(0)=1-frac, W(1)=frac for cell-centered
      wold = 0.0_num
      wold(0) = 1.0_num - frac_old
      wold(1) = frac_old
      
      ! Shape functions at new position (shifted by dcell)
      wnew = 0.0_num
      IF (dcell >= -1 .AND. dcell <= 1) THEN
        wnew(dcell) = 1.0_num - frac_new
        wnew(dcell+1) = frac_new
      ELSE
        ! Particle moved too far - use simple deposition at new position
        wnew(0) = 1.0_num - frac_new
        wnew(1) = frac_new
      END IF
      
      ! Shape function difference
      dw = wnew - wold
      
      ! Compute transverse velocity for Jy, Jz
      gamma_rel = SQRT((px(ip)*ipart_mc)**2 + (py(ip)*ipart_mc)**2 + &
                       (pz(ip)*ipart_mc)**2 + 1.0_num)
      igamma = 1.0_num / gamma_rel
      vy = py(ip) * ipart_mc * igamma * c
      vz = pz(ip) * ipart_mc * igamma * c
      
      ! Current factors
      ! Note: 1D requires fac=0.5 because there's no transverse shape function
      ! In 2D/3D, the transverse shape function provides this normalization
      qw_idt = 0.5_num * charge * weight(ip) * idt  ! For Jx (charge conservation)
      qw_idx = 0.5_num * charge * weight(ip) * idx  ! For Jy, Jz (velocity-based)
      
      ! Esirkepov Jx deposition: cumulative sum of -dw
      ! This ensures exact charge conservation
      jx_accum = 0.0_num
      DO ix = -1, 2
        ! Check bounds
        IF (ix_base + ix >= 1 - ng .AND. ix_base + ix <= nx + ng) THEN
          ! Jx from charge displacement
          jx_accum = jx_accum - qw_idt * dw(ix)
          !$ACC ATOMIC UPDATE
          jx(ix_base + ix) = jx(ix_base + ix) + jx_accum
          
          ! Jy, Jz from velocity with averaged shape
          jy_contrib = qw_idx * vy * (wold(ix) + 0.5_num * dw(ix))
          jz_contrib = qw_idx * vz * (wold(ix) + 0.5_num * dw(ix))
          
          !$ACC ATOMIC UPDATE
          jy(ix_base + ix) = jy(ix_base + ix) + jy_contrib
          !$ACC ATOMIC UPDATE
          jz(ix_base + ix) = jz(ix_base + ix) + jz_contrib
        END IF
      END DO
    END DO
    !$ACC END PARALLEL LOOP
    
  END SUBROUTINE deposit_current_gpu_1d

END MODULE particles_gpu

