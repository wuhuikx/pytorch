#ifndef ATOMIC_ADD_FLOAT
#define ATOMIC_ADD_FLOAT

static inline void cpu_atomic_add_float(float* dst, float fvalue)
{
  typedef union {
    uint32_t intV;
    float floatV;
  } uf32_t;

  uf32_t new_value, old_value;
  uint32_t* dst_intV = (uint32_t*)(dst);
  do {
    old_value.floatV = *dst;
    new_value.floatV = old_value.floatV + fvalue;
  } while (!__sync_bool_compare_and_swap(dst_intV, old_value.intV, new_value.intV));
}

#endif
