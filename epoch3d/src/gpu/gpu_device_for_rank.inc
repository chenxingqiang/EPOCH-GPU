  PURE FUNCTION gpu_device_for_rank(local_rank, device_count) RESULT(device_id)
    INTEGER, INTENT(IN) :: local_rank, device_count
    INTEGER :: device_id

    IF (device_count <= 0) THEN
      device_id = 0
    ELSE
      device_id = MOD(local_rank, device_count)
      IF (device_id < 0) device_id = device_id + device_count
    END IF
  END FUNCTION gpu_device_for_rank
