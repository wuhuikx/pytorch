#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.cpp"
#else

#include <new>

real* THStorage_(data)(const THStorage *self)
{
  return self->data<real>();
}

ptrdiff_t THStorage_(size)(const THStorage *self)
{
  return self->size;
}

size_t THStorage_(elementSize)()
{
  return sizeof(real);
}

THStorage* THStorage_(new)(void)
{
  return THStorage_new(at::CTypeToScalarType<th::from_type<real>>::to());
}

THStorage* THStorage_(newWithSize)(ptrdiff_t size)
{
  return THStorage_newWithSize(at::CTypeToScalarType<th::from_type<real>>::to(), size);
}

THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
                                        THAllocator *allocator,
                                        void *allocatorContext)
{
  return THStorage_newWithAllocator(at::CTypeToScalarType<th::from_type<real>>::to(), size, allocator, allocatorContext);
}


THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags)
{
  THMapAllocatorContext *ctx = THMapAllocatorContext_new(filename, flags);

  THStorage *storage = THStorage_(newWithAllocator)(size,
                                                    &THMapAllocator,
                                                    ctx);

  if(size <= 0)
    storage->size = THMapAllocatorContext_size(ctx)/sizeof(real);

  THStorage_(clearFlag)(storage, TH_STORAGE_RESIZABLE);

  return storage;
}

THStorage* THStorage_(newWithSize1)(real data0)
{
  THStorage *self = THStorage_(newWithSize)(1);
  real *data = THStorage_(data)(self);
  data[0] = data0;
  return self;
}

THStorage* THStorage_(newWithSize2)(real data0, real data1)
{
  THStorage *self = THStorage_(newWithSize)(2);
  real *data = THStorage_(data)(self);
  data[0] = data0;
  data[1] = data1;
  return self;
}

THStorage* THStorage_(newWithSize3)(real data0, real data1, real data2)
{
  THStorage *self = THStorage_(newWithSize)(3);
  real *data = THStorage_(data)(self);
  data[0] = data0;
  data[1] = data1;
  data[2] = data2;
  return self;
}

THStorage* THStorage_(newWithSize4)(real data0, real data1, real data2, real data3)
{
  THStorage *self = THStorage_(newWithSize)(4);
  real *data = THStorage_(data)(self);
  data[0] = data0;
  data[1] = data1;
  data[2] = data2;
  data[3] = data3;
  return self;
}

void THStorage_(setFlag)(THStorage *storage, const char flag)
{
  storage->flag |= flag;
}

void THStorage_(clearFlag)(THStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}

void THStorage_(retain)(THStorage *storage)
{
  if(storage && (storage->flag & TH_STORAGE_REFCOUNTED))
    ++storage->refcount;
}

int THStorage_(retainIfLive)(THStorage *storage)
{
  // TODO: Check if TH_STORAGE_REFCOUNTED?
  int refcount = storage->refcount.load();
  while (refcount > 0) {
    if (storage->refcount.compare_exchange_strong(refcount, refcount + 1)) {
      return 1;
    }
    refcount = storage->refcount.load();
  }
  return 0;
}

void THStorage_(free)(THStorage *storage)
{
  THStorage_free(storage);
}

THStorage* THStorage_(newWithData)(real *data, ptrdiff_t size)
{
  return THStorage_(newWithDataAndAllocator)(data, size,
                                             &THDefaultAllocator, NULL);
}

THStorage* THStorage_(newWithDataAndAllocator)(real* data, ptrdiff_t size,
                                               THAllocator* allocator,
                                               void* allocatorContext) {
  THStorage *storage = static_cast<THStorage*>(THAlloc(sizeof(THStorage)));
  storage->backend = at::kCPU;
  storage->scalar_type = at::CTypeToScalarType<th::from_type<real>>::to();
  storage->data_ptr = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocatorVoidPtr = allocator;
  storage->allocatorContext = allocatorContext;
  storage->device = 0;
  return storage;
}

void THStorage_(resize)(THStorage *storage, ptrdiff_t size)
{
  AT_ASSERT(storage->backend == at::kCPU);

  auto* th_allocator = static_cast<THAllocator*>(storage->allocatorVoidPtr);

  if(storage->flag & TH_STORAGE_RESIZABLE)
  {
    if(th_allocator->realloc == NULL) {
      /* case when the allocator does not have a realloc defined */
      real *old_data = THStorage_(data)(storage);
      ptrdiff_t old_size = storage->size;
      if (size == 0) {
        storage->data_ptr = NULL;
      } else {
        storage->data_ptr = th_allocator->malloc(
            storage->allocatorContext,
            sizeof(real)*size);
      }
      storage->size = size;
      if (old_data != NULL) {
        ptrdiff_t copy_size = old_size;
        if (storage->size < copy_size) {
          copy_size = storage->size;
        }
        if (copy_size > 0) {
          memcpy(THStorage_(data)(storage), old_data, sizeof(real)*copy_size);
        }
        th_allocator->free(storage->allocatorContext, old_data);
      }
    } else {
      storage->data_ptr = th_allocator->realloc(
              storage->allocatorContext,
              THStorage_(data)(storage),
              sizeof(real)*size);
      storage->size = size;
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }
}

void THStorage_(fill)(THStorage *storage, real value)
{
  ptrdiff_t i;
  for(i = 0; i < storage->size; i++)
    THStorage_(data)(storage)[i] = value;
}

void THStorage_(set)(THStorage *self, ptrdiff_t idx, real value)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  THStorage_(data)(self)[idx] = value;
}

real THStorage_(get)(const THStorage *self, ptrdiff_t idx)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  return THStorage_(data)(self)[idx];
}

void THStorage_(swap)(THStorage *storage1, THStorage *storage2)
{
#define SWAP(val) { val = storage1->val; storage1->val = storage2->val; storage2->val = val; }
    void *data_ptr;
    ptrdiff_t size;
    char flag;
    void *allocatorVoidPtr;
    void *allocatorContext;
    struct THStorage *view;
    int device;

    SWAP(data_ptr);
    SWAP(size);
    SWAP(flag);
    // don't swap refcount!
    SWAP(allocatorVoidPtr);
    SWAP(allocatorContext);
    SWAP(view);
    SWAP(device);
#undef SWAP
}

#endif
