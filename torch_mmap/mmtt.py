# mmapped pytorch tensors
# Because https://github.com/pytorch/pytorch/issues/64932
#         [RFC] TorchStore - A Shared-Memory Tensor Store 64932
# while lovely, and the need is real, may never get done
# Perhaps this is ugly and useful enough to create the motivation

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
from typing import Union, Tuple, List, cast
from warnings import warn

#tensors_start = 1<<16

r"""
What is the effect of this string?

"""


@dataclass
class MemoryMapping:
    """The data specifying a memory mapping.
    
    This is not used.
    """
    mm: mmap.mmap
    start_in_file: int
    end_in_file: int

class MMTT():
    """
    Another docstring
    """
    TensorSizeType = Union[int, torch.Size, List[int], Tuple[int, ...]]

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
        """Make the MMTT instance ready to provide tensors.
        """
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
        
    
    def get(self, name: str) -> torch.Tensor:
        """
        Get an existing tensor by name.
    
        Multiple copies in one or more processes can be had, all sharing the same storage.
    
        Parameters
        ----------
        name
            The name given to the tensor when it was created.
    
        Returns
        -------
            A tensor of the same shape and dtype, and sharing the same storage,
            as the tensor created with ``name``.
    
        Raises
        ------
        KeyError
            If no tensor of ``name`` has been created.
        """
        
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

    def _create_mmapped_buffer(self, n_bytes: int) -> Tuple[mmap.mmap, int, int]:
        """Create a ``mmap``ped buffer

        Parameters
        ----------
        n_bytes
            The required size of the buffer in bytes.
    
        Returns
        -------
            A tuple of the ``mmap``, the offset within the ``mmap``, and the start position in the file.
        """
         
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


    def zeros(self, name: str, *size_args: TensorSizeType, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.

        **FIXME:** use a default tensor type https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type.html#torch.set_default_tensor_type
        
        Parameters
        ----------
        name
            The name to give to this new tensor.

        size_args
            The size of the tensor to create.

        dtype
            The data dype of the tensor elements.
    
        Returns
        -------
            A tensor filled with the scalar value 0, with the shape defined by the variable argument size.
    
        Raises
        ------
        TypeError
            If the type of size_args is wrong.

        """
        if not size_args or size_args == (0,):
            return torch.zeros(0, dtype=dtype) # Has no content
        
        # Find the size of the tensor
        size_input = size_args[0]
    
        # Handle torch.Size
        if isinstance(size_input, torch.Size):
            size_list = list(size_input)
        # Handle list or tuple
        elif isinstance(size_input, (list, tuple)):
            size_list = list(size_input)
        # Handle multiple int arguments
        elif all(isinstance(v, int) for v in size_args):
            size_list = cast(List[int], list(size_args))
        else:
            raise TypeError(f"Unsupported type for size_args: {type(size_input)}")
    
        count = 1
        for i in size_list:
            count *= i
        element_size = torch.tensor([], dtype=dtype).element_size()
        n_bytes = count * element_size
        mm, map_offset, start_in_file = self._create_mmapped_buffer(n_bytes)

        # Make the tensor that uses the new space
        t = torch.frombuffer(mm, dtype=dtype, count=count, offset=map_offset)
        t = t.view(*size_list)

        # Remember name and where
        d = {"size": size_list,
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
