#include <cstdint>

typedef uint16_t GM_qHash_type;  // qhash
typedef uint16_t GM_kHash_type;  // khash
typedef uint16_t GM_idx_type;    // topk_index

const double GM_BW = 1420;
const double PEAK_FLOPS = 294.912;
const int VEC_NUM = 40;
const int CUBE_NUM = 20;

const int COMPRESS_NUM = 16;  // 16 bits for int16_t --> 1 int16_t represents 16 bools
