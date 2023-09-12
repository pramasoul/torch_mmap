# Test mmtt.py, a memory-mapped pytorch tensor package

import pytest
import torch
import ctypes
import json
import logging

from torch_mmap import MMTT

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])  # StreamHandler logs to console

# Now you can use logging in your tests
def test_example():
    logging.debug("This is a debug message.")


CONTENT = "content"


def test_create_file(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text(CONTENT, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == CONTENT
    assert len(list(tmp_path.iterdir())) == 1


@pytest.fixture
def mt_name(tmp_path):
    return str(tmp_path / "foo")

@pytest.fixture
def mt(mt_name):
    return MMTT(mt_name)
    
def test_create(mt):
    assert type(mt) == MMTT
    
def test_open(mt):
    assert mt.open() is mt # returns itself, for e.g. mt = MMTT(filename).open()
    mt.close()
    mt.close()
    
@pytest.fixture
def opened_mt(mt):
    mt.open()
    return mt

def test_close(opened_mt):
    mt = opened_mt
    mt.close()
    # A second close should silently succeed harmlessly
    mt.close()
    
###############
# tensor creation tests
#

def test_make_empty_zero(opened_mt):
    mt = opened_mt
    t = mt.zeros('t', 0)
    assert isinstance(t, torch.Tensor)
    assert torch.equal(t, torch.zeros(0))

def test_make_empty_zero_2(opened_mt):
    mt = opened_mt
    t = mt.zeros('t')
    assert isinstance(t, torch.Tensor)
    assert torch.equal(t, torch.zeros(0))

def test_make_simple_zeros(opened_mt):
    mt = opened_mt
    t = mt.zeros('t', 1)
    assert isinstance(t, torch.Tensor)
    assert t.sum() == 0.0
    assert len(t) == 1

def test_make_zeros_of_various_dtypes(opened_mt):
    mt = opened_mt
    t = mt.zeros('t', 1)
    assert t.dtype == torch.zeros(1).dtype
    possible_dtypes = [
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.float16,
        torch.bfloat16,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.bool ]
    for dtype in possible_dtypes:
        t = mt.zeros('t', 1, dtype=dtype)
        assert t.dtype == dtype
        assert torch.equal(t, torch.zeros(1, dtype=dtype))
        t = mt.zeros('t', 1,2,3, dtype=dtype)
        assert t.dtype == dtype
        assert torch.equal(t, torch.zeros(1,2,3, dtype=dtype))

def test_make_empty_ones(opened_mt):
    mt = opened_mt
    t = mt.ones('t', 0)
    assert isinstance(t, torch.Tensor)
    assert torch.equal(t, torch.ones(0))

def test_make_simple_ones(opened_mt):
    mt = opened_mt
    t = mt.ones('t', 1)
    assert isinstance(t, torch.Tensor)
    assert t.sum() == 1.0
    assert len(t) == 1

def test_make_ones_of_various_dtypes(opened_mt):
    mt = opened_mt
    t = mt.ones('t', 1)
    assert t.dtype == torch.ones(1).dtype
    possible_dtypes = [
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.float16,
        torch.bfloat16,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.bool ]
    for dtype in possible_dtypes:
        t = mt.ones('t', 1, dtype=dtype)
        assert t.dtype == dtype
        assert torch.equal(t, torch.ones(1, dtype=dtype))
        t = mt.ones('t', 1,2,3, dtype=dtype)
        assert t.dtype == dtype
        assert torch.equal(t, torch.ones(1,2,3, dtype=dtype))

def test_make_zeros_of_various_size_args(opened_mt):
    mt = opened_mt
    t = mt.zeros('t', 1,2,3)
    ref = torch.zeros(1,2,3)
    assert torch.equal(t, ref)
    t2 = mt.zeros('t2', [1,2,3])
    assert torch.equal(t2, ref)
    t3 = mt.zeros('t3', ref.shape)
    assert torch.equal(t3, ref)
    t4 = mt.zeros('t4', (1,2,3))
    assert torch.equal(t4, ref)

@pytest.fixture
def shaped_zeros(opened_mt):
    mt = opened_mt
    return mt.zeros('t', 2, 3, 5)

def test_make_shaped_zeros(shaped_zeros):
    t = shaped_zeros
    assert isinstance(t, torch.Tensor)
    assert t.sum() == 0.0
    assert t.numel() == 2*3*5
    assert list(t.shape) == [2, 3, 5]

def test_zeros_like(opened_mt):
    mt = opened_mt
    t = torch.ones(3,4,5)
    tref = torch.zeros_like(t)
    tmt = mt.zeros_like("tmt", tref)
    assert torch.equal(tmt, tref)
    tmt[1] = 3.14
    assert not torch.equal(tmt, tref)

# fixed: @pytest.mark.xfail(reason="new_copy has problem")
def test_new_copy(mt_with_shaped_zero):
    mt = mt_with_shaped_zero
    tref = torch.arange(7)
    tmt = mt.new_copy('tmt', tref)
    assert torch.equal(tmt, tref)
    tmt[0] = 11
    assert not torch.equal(tmt, tref)
    tref2 = torch.randn(7,3,2)
    tmt2 = mt.new_copy('tmt2', tref2)
    assert torch.equal(tmt2, tref2)

###############################
# Tensor alterations and lookups
#

def test_tensor_is_writeable(shaped_zeros):
    t = shaped_zeros
    t[1,2,3] = 3.14
    assert t[1,2,3] == 3.14
    assert t.sum() == 3.14

@pytest.fixture
def mt_with_shaped_zero(opened_mt):
    mt = opened_mt
    mt.zeros('t', 2, 3, 5)
    return mt
    
def test_tensor_can_be_gotten(mt_with_shaped_zero):
    mt = mt_with_shaped_zero
    t = mt.get('t')
    assert t.sum() == 0.0
    assert t.numel() == 2*3*5
    assert list(t.shape) == [2, 3, 5]

def test_not_present_cannot_be_gotten(opened_mt):
    mt = opened_mt
    #KeyError: 'dne not found in /tmp/pytest-of-soul/pytest-134/test_not_present_cannot_be_got0/foo'
    with pytest.raises(KeyError,
                       match="dne not found in .+"):
        t = mt.get('dne')
    mt.metadata = {} # Testing HACK to make the __del__ close() not complain

def test_gotten_tensor_is_writeable(mt_with_shaped_zero):
    mt = mt_with_shaped_zero
    t = mt.get('t')
    t[1,2,3] = 3.14
    assert t.sum() == 3.14

def test_tensors_track_basic(mt_with_shaped_zero):
    mt = mt_with_shaped_zero
    t = mt.get('t')
    t[1,2,3] = 3.14
    t2 = mt.get('t')
    assert t is not t2
    assert torch.equal(t, t2)
    t2[0, 1, 2] = 1/7
    assert torch.equal(t, t2)

def test_names_are_distinct(opened_mt):
    mt = opened_mt
    t1 = mt.zeros('t1', 3)
    t2 = mt.zeros('t2', 3)
    #logging.debug(json.dumps(mt.metadata))
    assert torch.equal(t1, t2)
    t1[1] = 3.14
    assert not torch.equal(t1, t2)
    t2[1] = 3.14
    assert torch.equal(t1, t2)

###############
# inter-MMTT communication tests
#

@pytest.fixture
def mt_opened_ab(tmp_path):
    mta = MMTT(str(tmp_path / "a"))
    mtb = MMTT(str(tmp_path / "b"))
    mta.open()
    mtb.open()
    return mta, mtb
                
def test_mt_ab(mt_opened_ab):
    # When the two filenames are different ...
    mta, mtb = mt_opened_ab
    assert mta is not mtb
    assert mta.name != mtb.name
    # ... tensors created with the same names in each ...
    ta1 = mta.zeros('t1', 7)
    tb1 = mtb.zeros('t1', 7)
    assert torch.equal(ta1, tb1)
    ta1[3] = 3.14
    # ... are isolated.
    assert not torch.equal(ta1, tb1)
    
@pytest.fixture
def mt_opened_aa(tmp_path):
    mta = MMTT(str(tmp_path / "a"))
    mtb = MMTT(str(tmp_path / "a"))
    mta.open()
    mtb.open()
    return mta, mtb
                
def test_two_of_same_filename(mt_opened_aa):
    # When the two filenames are the same ...
    mta, mtb = mt_opened_aa
    assert mta is not mtb
    assert mta.name == mtb.name
    # ... tensors with the same names in each ...
    ta1 = mta.zeros('t1', 7)
    tb1 = mtb.get('t1')
    assert torch.equal(ta1, tb1)
    # ... reflect each other's values ;-) ...
    ta1[3] = 3.14
    assert torch.equal(ta1, tb1)
    # ... bidirectionally.
    tb1[2] = 2.718
    assert torch.equal(ta1, tb1)
    # NOTE one can create again with same tensor name ...
    tb1c1 = mtb.zeros('t1', 7)
    # ... and get different tensors ...
    #logging.debug(json.dumps(mta.metadata))
    #logging.debug(json.dumps(mtb.metadata))
    assert not torch.equal(ta1, tb1c1)
    # ... until re-gotten in the other(s) ...
    ta1g1 = mta.get('t1')
    assert torch.equal(ta1, tb1)
    

###############
# boundaries tests
#

@pytest.fixture
def tiny_mt(mt_name):
    return MMTT(mt_name, 16)

@pytest.mark.skip(reason='obsolete')
def test_tiny_mt(tiny_mt):
    mt = tiny_mt
    with pytest.raises(json.decoder.JSONDecodeError):
        mt.open()
    mt.metadata = {} # Testing HACK to make the __del__ close() not complain

@pytest.fixture
def small_mt(mt_name):
    return MMTT(mt_name, 256)

@pytest.mark.skip(reason='obsolete')
def test_small_mt(small_mt):
    mt = small_mt
    mt.open()
    mt.close()
    mt.open()
    t = mt.zeros('t', 2,3,4)
    with pytest.raises(ValueError,
                       match="Metadata length [0-9]+ exceeds space 256."):
        for i in range(20):
            t = mt.zeros('t', i+1)
    mt.metadata = {} # Testing HACK to make the __del__ close() not complain


#@pytest.mark.skip(reason="segfaults")
def test_intertensor_isolation(opened_mt):
    mt = opened_mt
    t1 = mt.zeros('t1', 1000)
    t2 = mt.zeros("t2", 3217)
    t3 = mt.zeros('t3', 1000)
    # logging.debug(json.dumps(mt.metadata))
    # t1md = mt.metadata["tensor_metadata_by_name"]['t1'][-1]
    # assert t1md["file_offset"] + t1md["n_bytes"] < mt.metadata["first_free_pos"]
    # logging.debug(f"mt.mm.size() = {mt.mm.size()}")
    # logging.debug(f"mt.file.tell() = {mt.file.tell()}")
    # tmp = mt.file.tell()
    # end = mt.file.seek(0,2)
    # mt.file.seek(tmp)
    # logging.debug(f"mt.file.seek(0,2) = {end} = {end/4096} * 4096")
    # t1_data_ptr = t1.untyped_storage().data_ptr()
    # #logging.debug(f"t1.untyped_storage().data_ptr() {hex(t1_data_ptr)}")
    # address_with_offset = ctypes.addressof(ctypes.c_char.from_buffer(mt.mm, t1md["offset"]))
    # #logging.debug(f"mmap + offset = {hex(address_with_offset)}")
    # logging.debug(f"t1.untyped_storage().data_ptr() = {hex(t1_data_ptr)}, mmap + offset = {hex(address_with_offset)}, difference {hex(t1_data_ptr - address_with_offset)}")
    # logging.debug("Accessing t1[0] now")
    assert t1[0] == 0.0
    logging.debug("Accessing t1 now")
    assert t1.sum() == 0.0
    #assert t1.sum() == t2.sum() == t3.sum() == 0.0
    

###############
# persistence tests
#

@pytest.fixture
def mt_with_t1_t2(opened_mt):
    mt = opened_mt
    mt.zeros('t1', 2, 3, 5)
    mt.zeros("t2", 7)
    return mt
    
def test_different_tensor_names(mt_with_t1_t2):
    mt = mt_with_t1_t2
    t1 = mt.get("t1")
    t2 = mt.get("t2")
    assert not torch.equal(t1, t2)

def test_persistence_across_close(mt_name):
    mt = MMTT(mt_name)
    mt.open()
    t1 = mt.zeros('t1', 2,3,4)
    rn1 = torch.randn(2,1,3)
    t2 = mt.new_copy('t2', rn1)
    mt.close()
    # What if we access tensors from closed? should segvault, no?
    # Indeed this segfaults: assert torch.sum(t1) == 0.0
    mt2 = MMTT(mt_name)
    mt2.open()
    t1g = mt2.get('t1')
    t2g = mt2.get('t2')
    assert torch.equal(t1g, torch.zeros(2,3,4))
    assert torch.equal(t2g, rn1)




# def test_got_a_path_string(session_mt_path):
#     assert type(session_mt_path) == str



#######################
# debugging tests
#

@pytest.mark.skip(reason="peeks into internals that are obsolete")
def test_tensor_memory_allocation(opened_mt):
    mt = opened_mt
    t1 = mt.zeros('t1', 1000)
    #t2 = mt.zeros("t2", 3217)
    #t2 = mt.zeros("t2", 321)
    t2 = mt.zeros("t2", 317)
    #t3 = mt.zeros('t3', 1000)
    logging.debug(json.dumps(mt.metadata))
    t1md = mt.metadata["tensor_metadata_by_name"]['t1'][-1]
    assert t1md["offset"] + t1md["n_bytes"] < mt.metadata["first_free_pos"]
    logging.debug(f"mt.mm.size() = {mt.mm.size()}")
    logging.debug(f"mt.file.tell() = {mt.file.tell()}")
    assert mt.mm.size() >= mt.metadata["first_free_pos"]
    tmp = mt.file.tell()
    end = mt.file.seek(0,2)
    mt.file.seek(tmp)
    logging.debug(f"mt.file.seek(0,2) = {end} = {end/4096} * 4096")
    assert end % 4096 == 0

    t1_data_ptr = t1.untyped_storage().data_ptr()
    address_of_t1_in_mm_per_metadata = ctypes.addressof(ctypes.c_char.from_buffer(mt.mm, t1md["offset"]))
    addr_diff = t1_data_ptr - address_of_t1_in_mm_per_metadata

    logging.debug(f"t1.untyped_storage().data_ptr() = {hex(t1_data_ptr)}, mmap + offset = {hex(address_of_t1_in_mm_per_metadata)}, difference {hex(addr_diff)}")
    assert addr_diff == 0, f"t1 buffer start is {hex(addr_diff)} ahead of address in mm per metadata"
    logging.debug("Accessing t1[0] now")
    assert t1[0] == 0.0
    logging.debug("Accessing t1 now")
    assert t1.sum() == 0.0
    #assert t1.sum() == t2.sum() == t3.sum() == 0.0

def test_wild_case_1(opened_mt):
    # From train_instrumented.py
    mmt = opened_mt
    w_histo_edge_locations = mmt.new_copy("w_histo_edge_locations",
                                          torch.arange(-0.1, 0.1, 0.0004))
    #n_2d_tensors_in_model = sum(1 for _, p in model.named_parameters() if p.ndim == 2)
    n_2d_tensors_in_model = 50
    #print(f"Model has {n_2d_tensors_in_model} 2-d tensors")
    model_2d_tensors_histograms = mmt.zeros("model_2d_tensors_histograms",
                                             n_2d_tensors_in_model,
                                             len(w_histo_edge_locations) - 1)
    #print(f"model_2d_tensors_histograms.shape is {model_2d_tensors_histograms.shape}")
    dummy_model_histo_ix = torch.zeros_like(w_histo_edge_locations)
    mmt_lr = mmt.zeros('lr', 1)
    mmt_loss = mmt.zeros('loss', 1)
    mmt_iter = mmt.zeros('iter', 1, dtype=torch.int32)
    mmt_hy = mmt.zeros('hy', 499)
    hy = torch.zeros(499)
    mmt_hy.copy_(hy)
    for i in range(10):
        hy = torch.randn(499)
        mmt_hy.copy_(hy)
        assert torch.equal(hy, mmt_hy)


    
###############
# stress tests
#

def test_metadata_blowup(opened_mt):
    mt = opened_mt
# ValueError: Metadata length 65604 exceeds space 65536.
    with pytest.raises(ValueError,
                       match="Metadata length [0-9]+ exceeds space 65536."):
        for i in range(1<<20):
            t = mt.zeros('t', 1)
    mt.metadata = {} # Testing HACK to make the __del__ close() not complain

@pytest.mark.slow
def test_mem_blowup(opened_mt):
    mt = opened_mt
    with pytest.raises(ValueError, match="mmap length is greater than file size"):
        for i in range(20): # blows up at 13 with ValueError: mmap length is greater than file size
            logging.debug(f"mt.zeros of shape {list(range(2,i))}")
            t = mt.zeros('t', *list(range(2,i)))

# def test_grind(opened_mt):
#     mt = opened_mt
#     t1 = mt.zeros('t1', 2,3,4,5,6)

            
###############
# experiments about pytest
#

@pytest.fixture
def foo():
    return "foo"

@pytest.fixture
def bar():
    return "bar"

def test_pytest_foobar_fixtures(foo, bar):
    assert foo == "foo"
    assert bar == "bar"

@pytest.fixture
def somestring(s):
    return str(s)
    
# def test_pytest_somestring(somestring(7)):
#     assert soemstring == "7"
    
