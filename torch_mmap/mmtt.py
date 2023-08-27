# mmapped pytorch tensors

import ctypes
import io
import json
import logging
import math
import mmap
import os
import torch

from collections import defaultdict
from dataclasses import dataclass
from warnings import warn

#tensors_start = 1<<16

r"""

"""


@dataclass
class MemoryMapping:
    mm: mmap.mmap
    start_in_file: int
    end_in_file: int

class MMTT():
    def __init__(self, name, tensors_start=1<<16):
        logging.debug(f"MMTT init name {name} tensors_start {tensors_start}")
        self.name = name
        self.tensors_start = self.round_up_to_allocation_granularity(tensors_start)
        self.metadata = {
            "filename": self.name,

            # The beginning of tensor buffer space in the file.
            # [0, tensors_start) is reserved for metadata
            "tensors_start": self.tensors_start,   # unchanging
            
            # Where in the file the next tensor buffer can start
            "first_free_pos": self.tensors_start,  # grows with each tensor buffer allocation
            "tensor_metadata_by_name": defaultdict(list),
                        }
        # The memory mappings and info. Entries of 
        self.memory_mappings = [] # of MemoryMapping's
        #self.page_size = os.sysconf('SC_PAGE_SIZE')
        
    def _init_new(self):
        with open(self.name, 'wb') as f:
            f.write(json.dumps(self.metadata).encode('utf8'))
            logging.debug(f"_init_new tensors_start {self.tensors_start}, f.tell() {f.tell()}")
            f.write(b'\x00' * (self.tensors_start - f.tell()))
        self.file = open(self.name, 'r+b', buffering=0)
        logging.debug(f"_init_new self.file_size() {self.file_size()}")
        #self.save_metadata()

    def open(self):
        # Open the file unbuffered binary read-write and keep it.
        try:
            self.file = open(self.name, 'r+b', buffering=0)
        except FileNotFoundError:
            self._init_new()
        assert self.file_size() % mmap.ALLOCATIONGRANULARITY == 0, \
                f"file_size()={self.file_size()} is not a multiple of {mmap.ALLOCATIONGRANULARITY}"
        self.last_size_seen = self.file_size()
        self.load_metadata()
        return self
        
    def load_metadata(self):
        # parse the JSON from the beginning, up to and including the rightmost '}'
        self.file.seek(0)
        self.metadata = json.loads(self.read_until_zero())
        #self.metadata = json.loads(self.mm[0:1+self.mm.rfind(b'}', 0, self.tensors_start)])
        assert self.tensors_start == self.metadata['tensors_start'], f"self.tensors_start {self.tensors_start} != self.metadata['tensors_start'] {self.metadata['tensors_start']}"
        
        # JSON doesn't know defaultdict, so fix it thus
        d = defaultdict(list)
        d.update(self.metadata["tensor_metadata_by_name"])
        self.metadata["tensor_metadata_by_name"] = d
        
    def save_metadata(self):
        buf = json.dumps(self.metadata).encode('utf8')
        if len(buf) >= self.tensors_start:
            raise ValueError(f"Metadata length {len(buf)} exceeds space {self.tensors_start}.")
        else:
            self.file.seek(0)
            self.file.write(buf)

    def close(self):
        # if hasattr(self, 'mm'):
        #     if not self.mm.closed:
        #         self.save_metadata()
        #         self.mm.flush()
        #         self.mm.close()
        if hasattr(self, 'file'):
            self.file.close()

    def __del__(self):
        self.close()
        
    
    def get(self, name):
        # If the file has grown, the metadata may have changed
        if self.last_size_seen != os.fstat(self.file.fileno()).st_size:
            self.load_metadata()
            self.last_size_seen = os.fstat(self.file.fileno()).st_size
        # Look for the tensor by name
        d = self.get_tensor_metadata(name)

        mm, map_offset = self.find_mapped_buffer_for(d['start_in_file'], d['n_bytes'])
        t = torch.frombuffer(mm,
                             dtype=d["dtype"],
                             count=d["count"],
                             offset=map_offset)
        t = t.view(*d["size"]) # give it the correct shape
        return t

    def get_tensor_metadata(self, name):
        dlist = self.metadata["tensor_metadata_by_name"][name]
        if dlist == []:
            raise KeyError(f"{name} not found in {self.name}")
        d = dlist[-1]  # Use the last one created    
        rv = {}
        rv.update(d)
        rv["dtype"] = eval(d["dtype_str"])
        return rv

    def _create_mmapped_buffer(self, n_bytes):
        # First version: just grow the file and make a bespoke map every time
        start_in_file = self.file_size() # New buffer will start in new section of file
        logging.debug(f"self.file_size() {self.file_size()}")
        assert start_in_file % mmap.ALLOCATIONGRANULARITY == 0 # mmap requires this boundary
        new_space_length = self.round_up_to_allocation_granularity(n_bytes)
        self.file.seek(0,2)
        self.file.write(b'\x00' * new_space_length) # Init with zeros
        logging.debug(f"self.file_size() {self.file_size()}")
        assert self.file_size() % mmap.ALLOCATIONGRANULARITY == 0
        logging.debug(f"mmap({self.file.fileno()}, {new_space_length}, {start_in_file})")
        mm = mmap.mmap(self.file.fileno(), new_space_length, offset=start_in_file)
        map_offset = 0
        self.memory_mappings.append(MemoryMapping(mm=mm,
                                                  start_in_file=start_in_file,
                                                  end_in_file=start_in_file+new_space_length ))
        return mm, map_offset, start_in_file
    
    def zeros(self, name, *size, dtype=torch.float32):
        # Find the size of the tensor
        if size == () or size == (0,):
            return torch.zeros(0, dtype=dtype) # Has no content
        if isinstance(size[0], torch.Size):
            size = list(size[0])
        if isinstance(size[0], (list, tuple)):
            size = size[0]
        count = 1
        for i in size:
            count *= i
        element_size = torch.tensor([], dtype=dtype).element_size()
        n_bytes = count * element_size
        mm, map_offset, start_in_file = self._create_mmapped_buffer(n_bytes)

        # Make the tensor that uses the new space
        t = torch.frombuffer(mm, dtype=dtype, count=count, offset=map_offset)
        t = t.view(*size)

        # Remember name and where
        d = {"size": size,
             "count": count,
             "start_in_file": start_in_file,
             "dtype_str": str(dtype),
             "n_bytes": n_bytes,
            }
        self.metadata["tensor_metadata_by_name"][name].append(d)
        self.metadata['first_free_pos'] += n_bytes
        self.save_metadata()
        return t

    def zeros_like(self, name, model):
        return self.zeros(name, model.size(), dtype=model.dtype)

    def ones(self, name, *size, dtype=torch.float32):
        if size == () or size == (0,):
            return torch.ones(0, dtype=dtype) # Has no content
        return self.new_copy(name, torch.ones(*size, dtype=dtype))

    def ones_like(self, name, model):
        return self.ones(name, model.size(), dtype=model.dtype)
        
    def new_copy(self, name, model):
        t = self.zeros_like(name, model)
        t.copy_(model)
        return t
            
    def _tensor(self, *args, **kwargs):
        raise UnimplementedError
        t = torch.tensor(*args, **kwargs)
        data_ptr = t.untyped_storage().data_ptr()
        num_bytes = t.numel() * t.element_size()
        buffer = (ctypes.c_byte * num_bytes).from_address(data_ptr)
        extent = len(self.mm)
        if extent < tensors_start:
            extent = tensors_start
        new_extent = extent + num_bytes
        self.mm.resize(new_extent)
        self.mm[extent:extent+num_bytes] = buffer
        t = torch.frombuffer(self.mm[extent:extent+num_bytes], dtype=t.dtype)
        return t
    
    def _flat_yielder(a):
        try:
            for v in a:
                yield from flat(v)
        except:
            yield a

    def read_until_zero(self, chunk_size=4096):
        buffer = bytearray()
        while True:
            chunk = self.file.read(chunk_size)
            if not chunk:
                break
            if 0 in chunk:
                buffer.extend(chunk[:chunk.index(0)])
                break
            buffer.extend(chunk)
        return buffer

    def file_size(self):
        return os.fstat(self.file.fileno()).st_size

    def round_up_to_allocation_granularity(self, x):
        return math.ceil(x/mmap.ALLOCATIONGRANULARITY) * mmap.ALLOCATIONGRANULARITY

    def round_down_to_allocation_granularity(self, x):
        return math.floor(x/mmap.ALLOCATIONGRANULARITY) * mmap.ALLOCATIONGRANULARITY

    def find_mapped_buffer_for(self, fpos, n_bytes):
        # From the interval in the file, return an mmap and offset within it to use in torch.frombuffer
        logging.debug(f"find_mapped_buffer_for({fpos}, {n_bytes})")
        # Look for an existing mapping that contains the requested file extent
        if self.memory_mappings:
            for mapping in sorted(self.memory_mappings,
                                         key=lambda x: x.start_in_file,
                                         reverse=True):
                if mapping.start_in_file <= fpos:
                    break
            if mapping.start_in_file <= fpos and mapping.end_in_file >= fpos + n_bytes:
                # it fits in this mapping
                logging.debug(f"find_mapped_buffer_for returning {mapping.mm}, {fpos - mapping.start_in_file}")  
                return mapping.mm, fpos - mapping.start_in_file
        # Here we need to make a new mapping
        start_in_file = self.round_down_to_allocation_granularity(fpos)
        end_in_file = self.round_up_to_allocation_granularity(fpos + n_bytes)
        map_length = end_in_file - start_in_file
        mm = mmap.mmap(self.file.fileno(), map_length, offset=start_in_file)
        mapping = MemoryMapping(mm=mm, start_in_file=start_in_file, end_in_file=end_in_file)
        self.memory_mappings.append(mapping)
        logging.debug(f"find_mapped_buffer_for({fpos}, {n_bytes}) {mapping}")
        return mapping.mm, fpos - mapping.start_in_file
