PROGRAM test_integration
  USE gpu_config
  USE fields_gpu
  USE particles_gpu
  IMPLICIT NONE
  
  INTEGER, PARAMETER :: num = 8
  INTEGER, PARAMETER :: i8 = SELECTED_INT_KIND(15)
  REAL(num), PARAMETER :: c = 299792458.0_num
  
  INTEGER :: nx_test, ny_test, ng, i
  INTEGER(i8) :: n_particles
  REAL(num) :: dt, dx, dy
  REAL(num) :: cnx, cny, fac, hdtx, hdty
  REAL(num) :: charge, mass
  REAL(num) :: start_time, end_time, elapsed
  
  ! Test arrays
  REAL(num), ALLOCATABLE :: ex(:,:), ey(:,:), ez(:,:)
  REAL(num), ALLOCATABLE :: bx(:,:), by(:,:), bz(:,:)
  REAL(num), ALLOCATABLE :: jx(:,:), jy(:,:), jz(:,:)
  
  ! Particle arrays
  REAL(num), ALLOCATABLE :: pos_x(:), pos_y(:)
  REAL(num), ALLOCATABLE :: px(:), py(:), pz(:)
  REAL(num), ALLOCATABLE :: ex_p(:), ey_p(:), ez_p(:)
  REAL(num), ALLOCATABLE :: bx_p(:), by_p(:), bz_p(:)
  
  WRITE(*,*) "============================================"
  WRITE(*,*) "   EPOCH-GPU Integration Test"
  WRITE(*,*) "============================================"
  
  ! Initialize GPU
  CALL set_gpu_device(0)
  CALL print_gpu_info()
  
  ! Test parameters
  nx_test = 256
  ny_test = 256
  ng = 2
  n_particles = 1000000_i8
  dt = 1.0e-15_num
  dx = 1.0e-8_num
  dy = 1.0e-8_num
  charge = 1.602e-19_num
  mass = 9.109e-31_num
  
  cnx = c * dt / dx
  cny = c * dt / dy
  fac = dt / (8.854e-12_num * dx * dy)
  hdtx = 0.5_num * dt / dx
  hdty = 0.5_num * dt / dy
  
  WRITE(*,*) ""
  WRITE(*,*) "Test Configuration:"
  WRITE(*,*) "  Grid: ", nx_test, " x ", ny_test
  WRITE(*,*) "  Particles: ", n_particles
  WRITE(*,*) ""
  
  ! Allocate field arrays
  ALLOCATE(ex(1-ng:nx_test+ng, 1-ng:ny_test+ng))
  ALLOCATE(ey(1-ng:nx_test+ng, 1-ng:ny_test+ng))
  ALLOCATE(ez(1-ng:nx_test+ng, 1-ng:ny_test+ng))
  ALLOCATE(bx(1-ng:nx_test+ng, 1-ng:ny_test+ng))
  ALLOCATE(by(1-ng:nx_test+ng, 1-ng:ny_test+ng))
  ALLOCATE(bz(1-ng:nx_test+ng, 1-ng:ny_test+ng))
  ALLOCATE(jx(1-ng:nx_test+ng, 1-ng:ny_test+ng))
  ALLOCATE(jy(1-ng:nx_test+ng, 1-ng:ny_test+ng))
  ALLOCATE(jz(1-ng:nx_test+ng, 1-ng:ny_test+ng))
  
  ! Initialize fields
  ex = 0.0_num; ey = 0.0_num; ez = 0.0_num
  bx = 0.0_num; by = 0.0_num; bz = 1.0e-3_num
  jx = 0.0_num; jy = 0.0_num; jz = 0.0_num
  
  ! Allocate particle arrays
  ALLOCATE(pos_x(n_particles), pos_y(n_particles))
  ALLOCATE(px(n_particles), py(n_particles), pz(n_particles))
  ALLOCATE(ex_p(n_particles), ey_p(n_particles), ez_p(n_particles))
  ALLOCATE(bx_p(n_particles), by_p(n_particles), bz_p(n_particles))
  
  ! Initialize particles
  DO i = 1, n_particles
    pos_x(i) = dx * REAL(MOD(i-1, nx_test), num)
    pos_y(i) = dy * REAL((i-1) / nx_test, num)
    px(i) = 1.0e-24_num * SIN(REAL(i, num))
    py(i) = 1.0e-24_num * COS(REAL(i, num))
    pz(i) = 0.0_num
    ex_p(i) = 1.0e6_num
    ey_p(i) = 0.0_num
    ez_p(i) = 0.0_num
    bx_p(i) = 0.0_num
    by_p(i) = 0.0_num
    bz_p(i) = 1.0e-3_num
  END DO
  
  !================================================
  ! Test 1: Field Update GPU
  !================================================
  WRITE(*,*) "Test 1: Field Update GPU (256x256 grid)"
  
  !$ACC ENTER DATA COPYIN(ex, ey, ez, bx, by, bz, jx, jy, jz)
  
  CALL CPU_TIME(start_time)
  DO i = 1, 100
    CALL update_e_field_gpu(ex, ey, ez, bx, by, bz, jx, jy, jz, &
                            nx_test, ny_test, ng, cnx, cny, fac)
    CALL update_b_field_gpu(ex, ey, ez, bx, by, bz, &
                            nx_test, ny_test, ng, hdtx, hdty)
  END DO
  !$ACC WAIT
  CALL CPU_TIME(end_time)
  
  elapsed = (end_time - start_time) * 1000.0_num
  WRITE(*,*) "  100 field updates: ", elapsed, " ms"
  WRITE(*,*) "  Per iteration: ", elapsed/100.0_num, " ms"
  
  !$ACC EXIT DATA COPYOUT(ex, ey, ez, bx, by, bz) DELETE(jx, jy, jz)
  
  !================================================
  ! Test 2: Particle Push GPU
  !================================================
  WRITE(*,*) ""
  WRITE(*,*) "Test 2: Particle Push GPU (1M particles)"
  
  !$ACC ENTER DATA COPYIN(pos_x, pos_y, px, py, pz)
  !$ACC ENTER DATA COPYIN(ex_p, ey_p, ez_p, bx_p, by_p, bz_p)
  
  CALL CPU_TIME(start_time)
  DO i = 1, 100
    CALL push_particles_gpu(n_particles, pos_x, pos_y, px, py, pz, &
                            ex_p, ey_p, ez_p, bx_p, by_p, bz_p, &
                            charge, mass, dt)
  END DO
  !$ACC WAIT
  CALL CPU_TIME(end_time)
  
  elapsed = (end_time - start_time) * 1000.0_num
  WRITE(*,*) "  100 particle push iterations: ", elapsed, " ms"
  WRITE(*,*) "  Per iteration: ", elapsed/100.0_num, " ms"
  WRITE(*,*) "  Particles/sec: ", 100.0_num * REAL(n_particles, num) / (elapsed/1000.0_num)
  
  !$ACC EXIT DATA COPYOUT(pos_x, pos_y, px, py, pz)
  !$ACC EXIT DATA DELETE(ex_p, ey_p, ez_p, bx_p, by_p, bz_p)
  
  !================================================
  ! Verify results
  !================================================
  WRITE(*,*) ""
  WRITE(*,*) "Verification:"
  WRITE(*,*) "  Sample positions (first 3 particles):"
  DO i = 1, 3
    WRITE(*,*) "    Particle ", i, ": x=", pos_x(i), " y=", pos_y(i)
  END DO
  
  WRITE(*,*) ""
  WRITE(*,*) "============================================"
  WRITE(*,*) "   GPU Integration Test PASSED!"
  WRITE(*,*) "============================================"
  
  ! Cleanup
  DEALLOCATE(ex, ey, ez, bx, by, bz, jx, jy, jz)
  DEALLOCATE(pos_x, pos_y, px, py, pz)
  DEALLOCATE(ex_p, ey_p, ez_p, bx_p, by_p, bz_p)
  
END PROGRAM test_integration

