#include <climits>

#include "THStorage.hpp"

#include "generic/THStorage.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THStorage.cpp"
#include "THGenerateHalfType.h"

#include "generic/THStorageCopy.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THStorageCopy.cpp"
#include "THGenerateHalfType.h"

void THStorage_free(THStorage *storage) {
  AT_ASSERT(storage->backend == at::kCPU);

  if(!storage)
    return;

  if((storage->flag & TH_STORAGE_REFCOUNTED) && (storage->refcount.load() > 0))
  {
    if(--storage->refcount == 0)
    {
      if(storage->flag & TH_STORAGE_FREEMEM) {
        static_cast<THAllocator*>(storage->allocatorVoidPtr)->free(storage->allocatorContext, storage->data_ptr);
      }
      if(storage->flag & TH_STORAGE_VIEW) {
        THStorage_free(storage->view);
      }
      storage->refcount.~atomic<int>();
      THFree(storage);
    }
  }
}

THDescBuff THLongStorage_sizeDesc(const THLongStorage *size) {
  return _THSizeDesc(THLongStorage_data(size), size->size);
}

THLongStorage *THLongStorage_newInferSize(THLongStorage *size, ptrdiff_t nElement)
{
  ptrdiff_t total_size = (size->size > 0 ? 1 : 0);
  ptrdiff_t dim_infer = -1;
  ptrdiff_t i;
  for (i = 0; i < size->size; i++) {
    if (THLongStorage_data(size)[i] == -1) {
      THArgCheck(dim_infer == -1, 1, "only one dimension can be inferred");
      dim_infer = i;
    } else {
      total_size *= THLongStorage_data(size)[i];
    }
  }
  if (dim_infer != -1) {
    THDescBuff buf = THLongStorage_sizeDesc(size);
    THArgCheck(total_size > 0 && nElement % total_size == 0, 2,
        "size '%s' is invalid for input with %td elements", buf.str, nElement);
  } else {
    THDescBuff buf = THLongStorage_sizeDesc(size);
    THArgCheck(nElement == total_size, 2,
        "size '%s' is invalid for input with %td elements", buf.str, nElement);
  }
  THLongStorage* copy = THLongStorage_newWithSize(size->size);
  THLongStorage_copy(copy, size);
  if (dim_infer != -1) {
    THLongStorage_data(copy)[dim_infer] = nElement / total_size;
  }
  return copy;
}

THStorage* THStorage_new(at::ScalarType scalar_type)
{
  return THStorage_newWithSize(scalar_type, 0);
}

THStorage* THStorage_newWithSize(at::ScalarType scalar_type, ptrdiff_t size)
{
  return THStorage_newWithAllocator(scalar_type, size, &THDefaultAllocator, nullptr);
}

THStorage* THStorage_newWithAllocator(at::ScalarType scalar_type, ptrdiff_t size,
                                      THAllocator *allocator,
                                      void *allocatorContext)
{
  THStorage *storage = static_cast<THStorage*>(THAlloc(sizeof(THStorage)));
  storage->backend = at::kCPU;
  storage->scalar_type = scalar_type;
  storage->data_ptr = allocator->malloc(allocatorContext, at::elementSize(scalar_type)*size);
  storage->size = size;
  new (&storage->refcount) std::atomic<int>(1);
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocatorVoidPtr = allocator;
  storage->allocatorContext = allocatorContext;
  storage->device = INT_MIN;  // device is not meaningful on CPU
  return storage;
}
