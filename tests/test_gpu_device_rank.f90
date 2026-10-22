PROGRAM test_gpu_device_rank
  IMPLICIT NONE
  INTEGER :: nfail

  nfail = 0
  IF (gpu_device_for_rank(0, 4) /= 0) nfail = nfail + 1
  IF (gpu_device_for_rank(1, 4) /= 1) nfail = nfail + 1
  IF (gpu_device_for_rank(3, 4) /= 3) nfail = nfail + 1
  IF (gpu_device_for_rank(4, 4) /= 0) nfail = nfail + 1
  IF (gpu_device_for_rank(5, 4) /= 1) nfail = nfail + 1
  IF (gpu_device_for_rank(7, 1) /= 0) nfail = nfail + 1
  IF (gpu_device_for_rank(2, 0) /= 0) nfail = nfail + 1
  IF (gpu_device_for_rank(-1, 4) /= 3) nfail = nfail + 1

  IF (nfail /= 0) THEN
    WRITE(*,*) 'FAIL', nfail
    STOP 1
  END IF
  WRITE(*,*) 'PASS gpu_device_for_rank'
CONTAINS
  INCLUDE 'gpu_device_for_rank.inc'
END PROGRAM test_gpu_device_rank
