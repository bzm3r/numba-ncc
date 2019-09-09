# -*- coding: utf-8 -*-
import contextlib
import os
import time
import numpy as np
import gzip
import h5py
import shutil

num_timepoints = 18000
num_vertices = 16
num_info = 38
output_size = 12

data = np.empty((num_timepoints, num_vertices, num_info))

z = 0
for i in range(num_timepoints):
    for j in range(num_vertices):
        for k in range(num_info):
            data[i, j, k] = z
            z += 1


t0 = "A:\\test-compression\\t0.hdf5"

with contextlib.suppress(FileNotFoundError):
    os.remove(t0)

st = time.time()

with h5py.File(t0, "a") as f:
    dset = f.create_dataset(
        "test",
        (0, num_vertices, num_info),
        maxshape=(None, num_vertices, num_info),
        compression="gzip",
        compression_opts=9,
    )

for ix in np.arange(int(num_timepoints / output_size)):
    chunk = np.copy(
        np.copy(data[(ix * output_size) : (ix * output_size + output_size)])
    )
    with h5py.File(t0, "a") as f:
        dset = f["test"]

        orig_index = dset.shape[0]

        dset.resize(dset.shape[0] + chunk.shape[0], axis=0)
        dset[orig_index:, :, :] = chunk

et = time.time()

print(
    "test0: time taken: {} s, size: {} kB".format(
        np.round(et - st, 2), int(os.path.getsize(t0)) / 1000
    )
)

st = time.time()
with h5py.File(t0, "r") as f:
    dset = f["test"]
    on_ram_dset = np.empty(dset.shape, dtype=np.float64)
    dset.read_direct(on_ram_dset)

    print("Verification result: ", np.all(on_ram_dset[:] == data[:]))

et = time.time()

print("test0: time taken to read: {} s".format(np.round(et - st, 2)))

del on_ram_dset

# =====================================================================

# t1 = "A:\\test-compression\\t1.npy"
# t1_compressed = "A:\\test-compression\\t1.npy.gzip"
#
# with contextlib.suppress(FileNotFoundError):
#    os.remove(t1)
#
# st = time.time()
#
# fp = np.memmap(t1, dtype=data.dtype, mode='w+', shape=(num_timepoints, num_vertices, num_info))
#
# for ix in np.arange(int(num_timepoints/output_size)):
#    fp[(ix*output_size):(ix*output_size + output_size)] = np.copy(data[(ix*output_size):(ix*output_size + output_size)])
#
# del fp
#
# et = time.time()
#
# print("test1: time taken to write: {} s, size: {} kB".format(np.round(et - st, 2), int(os.path.getsize(t1))/1000))
#
# st = time.time()
#
# with open(t1, 'rb') as f_in:
#    with gzip.open(t1_compressed, 'wb') as f_out:
#        shutil.copyfileobj(f_in, f_out)
#
# with contextlib.suppress(FileNotFoundError):
#    os.remove(t1)
# et = time.time()
# print("test1_compressed: time taken to write: {} s, size: {} kB".format(np.round(et - st, 2), int(os.path.getsize(t1_compressed))/1000))
#
# st = time.time()
# with gzip.open(t1_compressed, 'rb') as f_in:
#    with open(t1, 'wb') as f_out:
#        shutil.copyfileobj(f_in, f_out)
#
#
# fp = np.memmap(t1, dtype=data.dtype, mode='r', shape=(num_timepoints, num_vertices, num_info))
#
# print("Verification result: ", np.all(fp[:] == data[:]))
# et = time.time()
#
# del fp
#
# print("test1: time taken to read: {} s".format(np.round(et - st, 2)))
