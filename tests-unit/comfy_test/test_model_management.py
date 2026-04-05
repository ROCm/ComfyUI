"""
Unit tests for comfy.model_management helpers and small APIs that are safe to run on CPU.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management as mm


class TestDtypeSize(unittest.TestCase):
    def test_float16_bfloat16_float32(self):
        self.assertEqual(mm.dtype_size(torch.float16), 2)
        self.assertEqual(mm.dtype_size(torch.bfloat16), 2)
        self.assertEqual(mm.dtype_size(torch.float32), 4)


class TestDeviceHelpers(unittest.TestCase):
    def test_is_device_type_and_cpu_mps_cuda(self):
        self.assertTrue(mm.is_device_cpu(torch.device("cpu")))
        self.assertFalse(mm.is_device_cpu(torch.device("meta")))
        self.assertTrue(mm.is_device_mps(torch.device("mps")))
        self.assertTrue(mm.is_device_cuda(torch.device("cuda:0")))
        self.assertTrue(mm.is_device_xpu(torch.device("xpu:0")))

    def test_is_device_type_non_device(self):
        self.assertFalse(mm.is_device_type(object(), "cpu"))

    def test_get_autocast_device(self):
        self.assertEqual(mm.get_autocast_device(torch.device("cuda:0")), "cuda")
        self.assertEqual(mm.get_autocast_device(torch.device("cpu")), "cpu")
        self.assertEqual(mm.get_autocast_device(0), "cuda")


class TestSupportsDtype(unittest.TestCase):
    def test_float32_always_true(self):
        self.assertTrue(mm.supports_dtype(torch.device("cpu"), torch.float32))
        self.assertTrue(mm.supports_dtype(torch.device("cuda:0"), torch.float32))

    def test_cpu_non_fp32(self):
        self.assertFalse(mm.supports_dtype(torch.device("cpu"), torch.float16))
        self.assertFalse(mm.supports_dtype(torch.device("cpu"), torch.bfloat16))


class TestPickWeightDtype(unittest.TestCase):
    def test_none_uses_fallback(self):
        d = mm.pick_weight_dtype(None, torch.float16, device=torch.device("cpu"))
        self.assertEqual(d, torch.float16)

    def test_downgrades_when_larger_than_fallback(self):
        d = mm.pick_weight_dtype(torch.float32, torch.float16, device=torch.device("cpu"))
        self.assertEqual(d, torch.float16)

    def test_respects_supports_cast(self):
        with patch.object(mm, "supports_cast", return_value=False):
            d = mm.pick_weight_dtype(torch.float16, torch.float32, device=torch.device("cpu"))
        self.assertEqual(d, torch.float32)


class TestCastTo(unittest.TestCase):
    def test_same_device_no_copy_returns_same_when_dtype_matches(self):
        w = torch.ones(2, 3, dtype=torch.float32)
        out = mm.cast_to(w, dtype=torch.float32, device=w.device, copy=False)
        self.assertIs(out, w)

    def test_dtype_conversion_same_device(self):
        w = torch.ones(2, 3, dtype=torch.float32)
        out = mm.cast_to(w, dtype=torch.float16, device=w.device, copy=False)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.allclose(out.float(), w))


class TestModuleSize(unittest.TestCase):
    def test_sums_parameter_bytes(self):
        m = torch.nn.Linear(4, 8, bias=True)
        expected = m.weight.nbytes + m.bias.nbytes
        self.assertEqual(mm.module_size(m), expected)


class TestArchiveModelDtypes(unittest.TestCase):
    def test_records_param_and_buffer_dtypes(self):
        m = torch.nn.Module()
        m.register_parameter("w", torch.nn.Parameter(torch.zeros(1, dtype=torch.float16)))
        m.register_buffer("b", torch.zeros(1, dtype=torch.bfloat16))
        mm.archive_model_dtypes(m)
        self.assertEqual(m.w_comfy_model_dtype, torch.float16)
        self.assertEqual(m.b_comfy_model_dtype, torch.bfloat16)


class TestPlatformHelpers(unittest.TestCase):
    @patch("comfy.model_management.platform.mac_ver", return_value=("14.2.1", ("", "", ""), ""))
    def test_mac_version_parses(self, _mac_ver):
        self.assertEqual(mm.mac_version(), (14, 2, 1))

    @patch("comfy.model_management.platform.mac_ver", side_effect=ValueError("bad"))
    def test_mac_version_returns_none_on_error(self, _mac_ver):
        self.assertIsNone(mm.mac_version())

    @patch("comfy.model_management.platform.uname")
    def test_is_wsl_microsoft_suffix(self, mock_uname):
        mock_uname.return_value.release = "5.10.0-Microsoft"
        self.assertTrue(mm.is_wsl())

    @patch("comfy.model_management.platform.uname")
    def test_is_wsl_wsl2_suffix(self, mock_uname):
        mock_uname.return_value.release = "5.15.0-microsoft-standard-WSL2"
        self.assertTrue(mm.is_wsl())

    @patch("comfy.model_management.platform.uname")
    def test_is_wsl_false(self, mock_uname):
        mock_uname.return_value.release = "6.8.0-31-generic"
        self.assertFalse(mm.is_wsl())


class TestOomHelpers(unittest.TestCase):
    def test_is_oom_torch_oom_subclass(self):
        if not hasattr(torch.cuda, "OutOfMemoryError"):
            self.skipTest("torch.cuda.OutOfMemoryError not available")
        err = torch.cuda.OutOfMemoryError("OOM")
        self.assertTrue(mm.is_oom(err))

#    def test_is_oom_accelerator_error_code_2(self):
#        class FakeAccel(Exception):
#            error_code = 2
#
#       self.assertTrue(mm.is_oom(FakeAccel()))

#    def test_is_oom_accelerator_error_message(self):
#        class FakeAccel(Exception):
#            error_code = 0
#    
#        self.assertTrue(mm.is_oom(FakeAccel("CUDA out of memory")))

    def test_is_oom_false_for_generic(self):
        self.assertFalse(mm.is_oom(RuntimeError("other")))

    def test_raise_non_oom_raises_non_oom(self):
        with self.assertRaises(RuntimeError):
            mm.raise_non_oom(RuntimeError("x"))

    def test_raise_non_oom_swallows_oom(self):
        if not hasattr(torch.cuda, "OutOfMemoryError"):
            self.skipTest("torch.cuda.OutOfMemoryError not available")
        mm.raise_non_oom(torch.cuda.OutOfMemoryError("OOM"))


class TestInterruptProcessing(unittest.TestCase):
    def setUp(self):
        mm.interrupt_current_processing(False)

    def tearDown(self):
        mm.interrupt_current_processing(False)

    def test_interrupt_toggle_and_query(self):
        self.assertFalse(mm.processing_interrupted())
        mm.interrupt_current_processing(True)
        self.assertTrue(mm.processing_interrupted())

    def test_throw_exception_if_processing_interrupted(self):
        mm.interrupt_current_processing(True)
        with self.assertRaises(mm.InterruptProcessingException):
            mm.throw_exception_if_processing_interrupted()
        self.assertFalse(mm.processing_interrupted())


class TestCpuMpsMode(unittest.TestCase):
    def test_cpu_mode_follows_cpu_state(self):
        with patch.object(mm, "cpu_state", mm.CPUState.CPU):
            self.assertTrue(mm.cpu_mode())
        with patch.object(mm, "cpu_state", mm.CPUState.GPU):
            self.assertFalse(mm.cpu_mode())

    def test_mps_mode(self):
        with patch.object(mm, "cpu_state", mm.CPUState.MPS):
            self.assertTrue(mm.mps_mode())
        with patch.object(mm, "cpu_state", mm.CPUState.CPU):
            self.assertFalse(mm.mps_mode())


class TestMemoryBudgetHelpers(unittest.TestCase):
    def test_extra_reserved_memory_returns_int(self):
        v = mm.extra_reserved_memory()
        self.assertIsInstance(v, int)
        self.assertGreater(v, 0)

    def test_minimum_inference_memory(self):
        v = mm.minimum_inference_memory()
        self.assertGreater(v, mm.extra_reserved_memory())


class TestOffloadHelpers(unittest.TestCase):
    def test_offloaded_memory_sums_matching_device(self):
        dev = torch.device("cpu")
        m1 = MagicMock()
        m1.device = dev
        m1.model_offloaded_memory.return_value = 100
        m2 = MagicMock()
        m2.device = torch.device("cuda:0")
        m2.model_offloaded_memory.return_value = 999
        self.assertEqual(mm.offloaded_memory([m1, m2], dev), 100)

    def test_use_more_memory_stops_when_budget_exhausted(self):
        dev = torch.device("cpu")
        m = MagicMock()
        m.device = dev

        def use_vram(n, **_kwargs):
            m.model_use_more_vram.assert_called_once()
            return n

        m.model_use_more_vram.side_effect = use_vram
        mm.use_more_memory(50, [m], dev)
        m.model_use_more_vram.assert_called_once_with(50)


class TestTextEncoderDtype(unittest.TestCase):
    def tearDown(self):
        for name in (
            "fp8_e4m3fn_text_enc",
            "fp8_e5m2_text_enc",
            "fp16_text_enc",
            "bf16_text_enc",
            "fp32_text_enc",
        ):
            setattr(args, name, False)

    def test_cli_overrides(self):
        args.fp32_text_enc = True
        self.assertEqual(mm.text_encoder_dtype(), torch.float32)
        args.fp32_text_enc = False
        args.bf16_text_enc = True
        self.assertEqual(mm.text_encoder_dtype(), torch.bfloat16)

    def test_default_fp16_for_cpu_device(self):
        for name in (
            "fp8_e4m3fn_text_enc",
            "fp8_e5m2_text_enc",
            "fp16_text_enc",
            "bf16_text_enc",
            "fp32_text_enc",
        ):
            setattr(args, name, False)
        self.assertEqual(mm.text_encoder_dtype(torch.device("cpu")), torch.float16)


class TestIntermediateDtype(unittest.TestCase):
    def tearDown(self):
        args.fp16_intermediates = False

    def test_fp16_intermediates_flag(self):
        args.fp16_intermediates = True
        self.assertEqual(mm.intermediate_dtype(), torch.float16)
        args.fp16_intermediates = False
        self.assertEqual(mm.intermediate_dtype(), torch.float32)


class TestVaeDtype(unittest.TestCase):
    def tearDown(self):
        for name in ("fp16_vae", "bf16_vae", "fp32_vae"):
            setattr(args, name, False)

    def test_cli_overrides(self):
        args.fp16_vae = True
        self.assertEqual(mm.vae_dtype(), torch.float16)
        args.fp16_vae = False
        args.fp32_vae = True
        self.assertEqual(mm.vae_dtype(), torch.float32)


class TestLoraComputeDtype(unittest.TestCase):
    def tearDown(self):
        mm.LORA_COMPUTE_DTYPES.clear()

    def test_caches_per_device(self):
        dev = torch.device("cpu")
        with patch.object(mm, "should_use_fp16", return_value=True):
            d1 = mm.lora_compute_dtype(dev)
        self.assertEqual(d1, torch.float16)
        with patch.object(mm, "should_use_fp16", return_value=False):
            d2 = mm.lora_compute_dtype(dev)
        self.assertEqual(d2, torch.float16)


#class TestGetSupportedFloat8Types(unittest.TestCase):
#    def test_returns_list_of_dtypes(self):
#        types = mm.get_supported_float8_types()
#        self.assertIsInstance(types, list)
#        for t in types:
#            self.assertTrue(issubclass(t, torch.dtype))


class TestLoadedModelEquality(unittest.TestCase):
    def test_eq_same_model_reference(self):
        class Dummy:
            load_device = torch.device("cpu")
            parent = None

        d = Dummy()
        a = mm.LoadedModel(d)
        b = mm.LoadedModel(d)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
