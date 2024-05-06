// set tuning macros in this file

/*
  sets the tile size for matmul kernels
  best value varies from device to device
*/
#define MATMUL_TILE_SIZE 16

/*
  sets the local memory padding size for matmul kernels
  set it to 0 to disable padding
*/
#define MATMUL_LOCAL_MEM_PADDING_SIZE 1

/*
  vload size in matmul kernels
  possible values: 4, 8, 16
  set it to 0 to disable
  note: MATMUL_TILE_SIZE must be multiple of this value
*/
#define MATMUL_VLOAD_SIZE 8

/*
  set it to 1 to enable preload
  set it to 0 to disable
*/
#define MATMUL_DO_PRELOAD 1
