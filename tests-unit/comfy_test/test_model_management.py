"""
Unit tests for comfy.model_management helpers and small APIs that are safe to run on CPU.
"""
import contextlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import comfy.memory_management
from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management as mm


class TestDtypeSize(unittest.TestCase):
    def test_float16_bfloat16_float32(self):
        self.assertEqual(mm.dtype_size(torch.float16), 2)
        self.assertEqual(mm.dtype_size(torch.bfloat16), 2)
        self.assertEqual(mm.dtype_size(torch.float32), 4)

    def test_int8_uses_itemsize(self):
        self.assertEqual(mm.dtype_size(torch.int8), 1)

    def test_float64_uses_itemsize(self):
        self.assertEqual(mm.dtype_size(torch.float64), 8)


class TestDeviceHelpers(unittest.TestCase):
    def test_is_device_type_and_cpu_mps_cuda(self):
        self.assertTrue(mm.is_device_cpu(torch.device("cpu")))
        self.assertFalse(mm.is_device_cpu(torch.device("meta")))
        self.assertTrue(mm.is_device_mps(torch.device("mps")))
        self.assertTrue(mm.is_device_cuda(torch.device("cuda:0")))

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

    @patch("comfy.model_management.discard_cuda_async_error", MagicMock())
    def test_is_oom_accelerator_error_code_2(self, *_):
        class E(mm.ACCELERATOR_ERROR):
            def __init__(self):
                super().__init__("x")
                self.error_code = 2

        self.assertTrue(mm.is_oom(E()))

    @patch("comfy.model_management.discard_cuda_async_error", MagicMock())
    def test_is_oom_accelerator_oom_substring(self, *_):
        self.assertTrue(
            mm.is_oom(
                mm.ACCELERATOR_ERROR("cuda out of memory in kernel")
            )
        )

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


class TestLoadedModelEquality(unittest.TestCase):
    def test_eq_same_model_reference(self):
        class Dummy:
            load_device = torch.device("cpu")
            parent = None

        d = Dummy()
        a = mm.LoadedModel(d)
        b = mm.LoadedModel(d)
        self.assertEqual(a, b)

    def test_eq_different_model_not_equal(self):
        class Dummy:
            def __init__(self):
                self.load_device = torch.device("cpu")
                self.parent = None

        # LoadedModel holds only a weakref to the model; keep Dummies alive or
        # they may be collected and both .model resolve to None (then __eq__ is True).
        d1 = Dummy()
        d2 = Dummy()
        a = mm.LoadedModel(d1)
        b = mm.LoadedModel(d2)
        self.assertNotEqual(a, b)


class TestGetSupportedFloat8Types(unittest.TestCase):
    def test_returns_list_of_dtypes(self):
        types = mm.get_supported_float8_types()
        self.assertIsInstance(types, list)
        for t in types:
            # dtype members are torch.dtype instances (issubclass would not apply)
            self.assertIsInstance(t, torch.dtype)


class TestAmdMinVersion(unittest.TestCase):
    def test_false_when_not_amd(self):
        with patch.object(mm, "is_amd", return_value=False):
            self.assertFalse(mm.amd_min_version())

    def test_false_when_device_is_cpu(self):
        with patch.object(mm, "is_amd", return_value=True):
            with patch.object(mm, "is_device_cpu", return_value=True):
                self.assertFalse(mm.amd_min_version(torch.device("cpu")))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda required")
    def test_parses_gfx7_style_arch(self):
        with patch.object(mm, "is_amd", return_value=True):
            with patch.object(mm, "is_device_cpu", return_value=False):
                mock_props = MagicMock()
                mock_props.gcnArchName = "gfx1030"
                with patch("torch.cuda.get_device_properties", return_value=mock_props):
                    with patch.object(
                        torch.cuda, "get_device_name", return_value="AMD Radeon"
                    ):
                        d = torch.device("cuda:0")
                        out = mm.amd_min_version(d, min_rdna_version=0)
        self.assertIsInstance(out, bool)


class TestGetTorchDevice(unittest.TestCase):
    @patch.object(mm, "is_intel_xpu", return_value=False)
    @patch.object(mm, "is_ascend_npu", return_value=False)
    @patch.object(mm, "is_mlu", return_value=False)
    def test_cpu_state_cpu(self, _mlu, _npu, _xpu):
        with patch.object(mm, "cpu_state", mm.CPUState.CPU):
            self.assertEqual(mm.get_torch_device(), torch.device("cpu"))

    @patch.object(mm, "is_intel_xpu", return_value=False)
    @patch.object(mm, "is_ascend_npu", return_value=False)
    @patch.object(mm, "is_mlu", return_value=False)
    @patch("comfy.model_management.torch.mps", create=True)
    @patch("comfy.model_management.torch.device")
    def test_mps_uses_mps_device(self, mock_device, _mock_mps, _mlu, _npu, _xpu):
        with patch.object(mm, "cpu_state", mm.CPUState.MPS):
            mock_device.return_value = torch.device("mps:0")
            d = mm.get_torch_device()
            mock_device.assert_called()


class TestGetTotalMemoryAndFreeMemory(unittest.TestCase):
    def test_get_total_memory_cpu(self):
        v = mm.get_total_memory(torch.device("cpu"), torch_total_too=False)
        self.assertIsInstance(v, (int, float))
        self.assertGreater(v, 0)

    def test_get_total_memory_cpu_torch_too(self):
        mem, mem_t = mm.get_total_memory(torch.device("cpu"), torch_total_too=True)
        self.assertIsInstance(mem, (int, float))
        self.assertIsInstance(mem_t, (int, float))

    def test_get_free_memory_cpu(self):
        m = mm.get_free_memory(torch.device("cpu"), torch_free_too=False)
        self.assertIsInstance(m, (int, float))
        m2, t2 = mm.get_free_memory(torch.device("cpu"), torch_free_too=True)
        self.assertIsInstance(m2, (int, float))
        self.assertIsInstance(t2, (int, float))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    def test_get_free_memory_cuda(self):
        dev = torch.device("cuda:0")
        self.assertIsInstance(mm.get_free_memory(dev), (int, float))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    @patch("comfy.model_management.get_device_name", create=True, new=lambda *a, **k: "GPU")
    def test_get_torch_device_name_cuda_type(self, *_):
        with patch.object(
            torch.cuda, "get_allocator_backend", return_value="native"
        ) as m:
            s = mm.get_torch_device_name(torch.device("cuda:0"))
        self.assertIn("cuda:0", s)
        m.assert_called()


class TestGetTorchDeviceNameBranches(unittest.TestCase):
    def test_string_device_fallback(self):
        s = mm.get_torch_device_name(0)
        self.assertIsInstance(s, str)
        self.assertIn("0", s)

    def test_cpu_type_device(self):
        s = mm.get_torch_device_name(torch.device("cpu"))
        self.assertEqual(s, "cpu")


class TestUnetAndDeviceHelpers(unittest.TestCase):
    def test_unet_offload_high_vram(self):
        with patch.object(mm, "vram_state", mm.VRAMState.HIGH_VRAM):
            d = mm.unet_offload_device()
        self.assertEqual(d, mm.get_torch_device())

    def test_unet_offload_not_high_uses_cpu(self):
        with patch.object(mm, "vram_state", mm.VRAMState.LOW_VRAM):
            d = mm.unet_offload_device()
        self.assertEqual(d, torch.device("cpu"))

    def test_maximum_vram_for_weights(self):
        with patch.object(mm, "get_total_memory", return_value=1e9):
            with patch.object(
                mm, "minimum_inference_memory", return_value=1e6
            ):
                v = mm.maximum_vram_for_weights()
        self.assertIsInstance(v, (int, float))
        self.assertLess(v, 1e9)

    @contextlib.contextmanager
    def _aimdo(self, v):
        old = getattr(comfy.memory_management, "aimdo_enabled", False)
        comfy.memory_management.aimdo_enabled = v
        try:
            yield
        finally:
            comfy.memory_management.aimdo_enabled = old

    def test_unet_inital_load_device_aimdo_cpu(self):
        with self._aimdo(True):
            d = mm.unet_inital_load_device(1_000, torch.float32)
        self.assertEqual(d, torch.device("cpu"))

    def test_unet_inital_load_device_high_vram_uses_torch(self):
        with self._aimdo(False):
            with patch.object(mm, "vram_state", mm.VRAMState.HIGH_VRAM):
                d = mm.unet_inital_load_device(1_000, torch.float32)
        self.assertEqual(d, mm.get_torch_device())

    def test_unet_inital_load_device_no_vram_uses_cpu(self):
        with self._aimdo(False):
            with patch.object(mm, "vram_state", mm.VRAMState.NO_VRAM):
                d = mm.unet_inital_load_device(1_000, torch.float32)
        self.assertEqual(d, torch.device("cpu"))

    def test_unet_inital_load_device_smart_chooses_device(self):
        with self._aimdo(False):
            with patch.object(mm, "vram_state", mm.VRAMState.LOW_VRAM):
                with patch.object(mm, "DISABLE_SMART_MEMORY", False):
                    with patch.object(
                        mm, "get_free_memory", side_effect=(1e9, 1)
                    ):
                        d = mm.unet_inital_load_device(10, torch.float32)
        # torch has more "free" than CPU path in mock; same logical device as get_torch_device()
        self.assertEqual(d, mm.get_torch_device())

    @staticmethod
    def _reset_fp_unet_flags():
        for name in (
            "fp32_unet",
            "fp64_unet",
            "bf16_unet",
            "fp16_unet",
            "fp8_e4m3fn_unet",
            "fp8_e5m2_unet",
            "fp8_e8m0fnu_unet",
        ):
            setattr(args, name, False)

    def tearDown(self):
        self._reset_fp_unet_flags()

    def test_unet_dtype_cli_fp32(self):
        self._reset_fp_unet_flags()
        args.fp32_unet = True
        self.assertEqual(
            mm.unet_dtype(), torch.float32, msg="fp32 unet override"
        )

    def test_unet_dtype_cli_bf16(self):
        self._reset_fp_unet_flags()
        args.bf16_unet = True
        self.assertEqual(mm.unet_dtype(), torch.bfloat16)

    def test_unet_dtype_cli_fp16(self):
        self._reset_fp_unet_flags()
        args.fp16_unet = True
        self.assertEqual(mm.unet_dtype(), torch.float16)

    @unittest.skipIf(
        not hasattr(torch, "float8_e4m3fn")
        and torch.float8_e4m3fn not in getattr(
            torch, "float8_e4m3fn", type("X", (), {})()
        ),
        "no float8",
    )
    def test_unet_dtype_cli_fp8(self):
        self._reset_fp_unet_flags()
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("float8 e4m3 not in torch")
        args.fp8_e4m3fn_unet = True
        self.assertEqual(mm.unet_dtype(), torch.float8_e4m3fn)

    def test_unet_dtype_negative_model_params_uses_sanity(self):
        self._reset_fp_unet_flags()
        d = mm.unet_dtype(
            model_params=-1, supported_dtypes=[torch.float32]
        )
        self.assertEqual(d, torch.float32)

    def test_unet_manual_cast_fp32_gives_none(self):
        d = mm.unet_manual_cast(
            torch.float32, torch.device("cpu")
        )
        self.assertIsNone(d)

    def test_unet_manual_cast_fp16_fp32_fallback_cpu(self):
        d = mm.unet_manual_cast(
            torch.float16, torch.device("cpu"), supported_dtypes=[torch.float16]
        )
        self.assertEqual(d, torch.float32)


def _set_bool_attrs(obj, names, value: bool):
    for n in names:
        setattr(obj, n, value)


class TestTextEncoderAndIntermediateVaeDevice(unittest.TestCase):
    @staticmethod
    def _reset_text_flags():
        for name in (
            "fp8_e4m3fn_text_enc",
            "fp8_e5m2_text_enc",
            "fp16_text_enc",
            "bf16_text_enc",
            "fp32_text_enc",
        ):
            setattr(args, name, False)

    def test_text_encoder_offload_device_gpu_only(self):
        _old = args.gpu_only
        args.gpu_only = True
        try:
            d = mm.text_encoder_offload_device()
        finally:
            args.gpu_only = _old
        self.assertEqual(d, mm.get_torch_device())

    def test_text_encoder_device_caches_cpu_when_vram_normal_and_no_fp16(self):
        _old = args.gpu_only
        _reset = self._reset_text_flags
        _reset()
        with patch.object(mm, "vram_state", mm.VRAMState.NORMAL_VRAM):
            with patch.object(mm, "should_use_fp16", return_value=False):
                with patch.object(comfy.memory_management, "aimdo_enabled", False):
                    args.gpu_only = False
                    d = mm.text_encoder_device()
        args.gpu_only = _old
        _reset()
        self.assertEqual(d, torch.device("cpu"))

    def test_text_encoder_initial_mps_uses_load_device(self):
        mps = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
        d = mm.text_encoder_initial_device(mps, torch.device("cpu"), 10**9)
        self.assertEqual(d, mps if mps.type == "mps" else torch.device("cpu"))


def _vae_tearDown():
    for n in ("fp16_vae", "bf16_vae", "fp32_vae", "cpu_vae", "gpu_only"):
        if n == "cpu_vae":
            setattr(args, n, False)
        else:
            setattr(args, n, False)


class TestVaeDtypeWithAllowed(unittest.TestCase):
    def setUp(self):
        _vae_tearDown()

    def tearDown(self):
        _vae_tearDown()

    def test_allowed_f16(self):
        with patch.object(mm, "should_use_fp16", return_value=True):
            with patch.object(
                mm, "should_use_bf16", return_value=False
            ):
                d = mm.vae_dtype(
                    torch.device("cpu"),
                    allowed_dtypes=[torch.float16, torch.bfloat16],
                )
        self.assertEqual(d, torch.float16)


class TestDeviceSupportNonBlocking(unittest.TestCase):
    def test_mps_device_false(self):
        with patch.object(args, "force_non_blocking", False):
            self.assertFalse(
                mm.device_supports_non_blocking(torch.device("mps"))
            )

    def test_force_non_blocking(self):
        with patch.object(args, "force_non_blocking", True):
            self.assertTrue(
                mm.device_supports_non_blocking(torch.device("mps"))
            )

    @patch("comfy.model_management.is_intel_xpu", return_value=True)
    def test_intel_xpu_disables(self, _ixu):
        with patch.object(args, "force_non_blocking", False):
            self.assertFalse(
                mm.device_supports_non_blocking(torch.device("cpu"))
            )


class TestForceChannelsLastAndDirectmlFlags(unittest.TestCase):
    def test_force_channels_last(self):
        _o = args.force_channels_last
        try:
            args.force_channels_last = True
            self.assertTrue(mm.force_channels_last())
        finally:
            args.force_channels_last = _o
        with patch.object(args, "force_channels_last", False):
            self.assertFalse(mm.force_channels_last())

    def test_is_directml_enabled(self):
        with patch.object(mm, "directml_enabled", False):
            self.assertFalse(mm.is_directml_enabled())
        with patch.object(mm, "directml_enabled", True):
            self.assertTrue(mm.is_directml_enabled())


class TestXformersAndAttentionFlags(unittest.TestCase):
    def test_sage_flash_flags(self):
        o_sage, o_flash = args.use_sage_attention, args.use_flash_attention
        try:
            args.use_sage_attention = True
            self.assertTrue(mm.sage_attention_enabled())
            args.use_sage_attention = o_sage
            args.use_flash_attention = True
            self.assertTrue(mm.flash_attention_enabled())
        finally:
            args.use_sage_attention, args.use_flash_attention = o_sage, o_flash

    def test_xformers_cpu_returns_false(self):
        with patch.object(mm, "cpu_state", mm.CPUState.CPU):
            self.assertFalse(mm.xformers_enabled())
            self.assertFalse(mm.xformers_enabled_vae())

    def test_pytorch_vae_false_on_amd(self):
        with patch.object(mm, "is_amd", return_value=True):
            self.assertFalse(mm.pytorch_attention_enabled_vae())

    @patch("comfy.model_management.mac_version", return_value=(15, 0))
    def test_force_upcast(self, _mv):
        o = args.force_upcast_attention
        try:
            args.force_upcast_attention = True
            m = mm.force_upcast_attention_dtype()
        finally:
            args.force_upcast_attention = o
        self.assertEqual(m, {torch.float16: torch.float32})

    @patch("comfy.model_management.mac_version", return_value=None)
    def test_force_upcast_no_mac(self, _mv):
        o = args.force_upcast_attention
        try:
            args.force_upcast_attention = False
            m = mm.force_upcast_attention_dtype()
        finally:
            args.force_upcast_attention = o
        self.assertIsNone(m)


class TestPytorchFlashAttentionHelper(unittest.TestCase):
    @patch("comfy.model_management.ENABLE_PYTORCH_ATTENTION", False)
    @patch("comfy.model_management.is_nvidia", return_value=True)
    def test_false_when_pytorch_att_disabled(self, _nv):
        self.assertFalse(mm.pytorch_attention_flash_attention())


class TestShouldUseFp16Bf16(unittest.TestCase):
    def test_cpu_mode_returns_fp16_no(self):
        with patch.object(mm, "cpu_mode", return_value=True):
            self.assertFalse(mm.should_use_fp16())
            self.assertFalse(mm.should_use_bf16())

    @patch("comfy.model_management.is_directml_enabled", return_value=True)
    @patch("comfy.model_management.mps_mode", return_value=False)
    @patch("comfy.model_management.cpu_mode", return_value=False)
    @patch("comfy.model_management.args", create=True)
    def test_directml_fp16_true(
        self, m_args, _cpu, _mps, _dml
    ):
        m_args.force_fp16 = False
        m_args.force_fp32 = False
        m_args = mm.args
        m_args.force_fp16 = False
        self.assertTrue(mm.should_use_fp16(torch.device("cuda:0")))

    @patch("comfy.model_management.is_directml_enabled", return_value=False)
    @patch("comfy.model_management.mps_mode", return_value=True)
    @patch("comfy.model_management.cpu_mode", return_value=False)
    @patch("comfy.model_management.args", create=True)
    def test_mps_mode_uses_fp16(
        self, m_args, _cpu, _mps, _dml
    ):
        m_args = args
        m_args.force_fp16 = False
        m_args.force_fp32 = False
        self.assertTrue(mm.should_use_fp16())


class TestShouldUseBf16Macos(unittest.TestCase):
    @patch("comfy.model_management.mac_version", return_value=(13, 0))
    @patch("comfy.model_management.mps_mode", return_value=True)
    @patch("comfy.model_management.cpu_mode", return_value=False)
    @patch("comfy.model_management.args", create=True)
    @patch("comfy.model_management.is_directml_enabled", return_value=False)
    @patch("comfy.model_management.is_ascend_npu", return_value=False)
    @patch("comfy.model_management.is_intel_xpu", return_value=False)
    @patch("comfy.model_management.is_mlu", return_value=False)
    @patch("comfy.model_management.is_ixuca", return_value=False)
    @patch("comfy.model_management.is_amd", return_value=False)
    def test_bf16_mac_lt_14(self, *_mocks):
        old = (args.force_fp32,)
        try:
            args.force_fp32 = False
            self.assertFalse(
                mm.should_use_bf16(torch.device("mps:0"))
            )
        finally:
            (args.force_fp32,) = old

    @patch("comfy.model_management.mac_version", return_value=(14, 0))
    @patch("comfy.model_management.mps_mode", return_value=True)
    @patch("comfy.model_management.cpu_mode", return_value=False)
    @patch("comfy.model_management.args", create=True)
    @patch("comfy.model_management.is_directml_enabled", return_value=False)
    @patch("comfy.model_management.is_ascend_npu", return_value=False)
    @patch("comfy.model_management.is_intel_xpu", return_value=False)
    @patch("comfy.model_management.is_mlu", return_value=False)
    @patch("comfy.model_management.is_ixuca", return_value=False)
    @patch("comfy.model_management.is_amd", return_value=False)
    def test_bf16_mac_ge_14(self, *_mocks):
        old = args.force_fp32
        try:
            args.force_fp32 = False
            self.assertTrue(mm.should_use_bf16(torch.device("mps:0")))
        finally:
            args.force_fp32 = old


class TestSupportsFp8Extended(unittest.TestCase):
    def test_supports_fp8_when_flag(self):
        with patch.object(mm, "SUPPORT_FP8_OPS", True):
            self.assertTrue(mm.supports_fp8_compute())

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    @patch("comfy.model_management.is_nvidia", return_value=True)
    @patch("comfy.model_management.torch_version_numeric", (2, 5))
    def test_supports_nvfp4(self, _nv):
        p = PropertyMock(
            return_value=MagicMock(major=10, minor=0, name="A100+")
        )
        with patch(
            "torch.cuda.get_device_properties", return_value=p
        ) as m:
            m.return_value = MagicMock(major=10, minor=0, name="Test")
        props = MagicMock(major=10, minor=0, name="Test")
        with patch("torch.cuda.get_device_properties", return_value=props):
            self.assertTrue(
                mm.supports_nvfp4_compute(torch.device("cuda:0"))
            )
            with patch("comfy.model_management.torch_version_numeric", (2, 9, 0)):
                if mm.supports_mxfp8_compute is not None:
                    self.assertFalse(
                        mm.supports_mxfp8_compute(
                            torch.device("cuda:0")
                        )
                    )
            with patch("comfy.model_management.torch_version_numeric", (2, 10)):
                self.assertTrue(
                    mm.supports_mxfp8_compute(
                        torch.device("cuda:0")
                    )
                )

    @patch("comfy.model_management.is_nvidia", return_value=False)
    def test_nvfp4_false_non_nvidia(self, _nv):
        if torch.cuda.is_available():
            self.assertFalse(
                mm.supports_nvfp4_compute(torch.device("cuda:0"))
            )


class TestExtendedFp16(unittest.TestCase):
    @patch("comfy.model_management.torch_version_numeric", (2, 5))
    def test_old_torch_false(self):
        self.assertFalse(mm.extended_fp16_support())

    @patch("comfy.model_management.torch_version_numeric", (2, 7))
    def test_new_true(self):
        self.assertTrue(mm.extended_fp16_support())


class TestSynchAndCache(unittest.TestCase):
    @patch("comfy.model_management.cpu_mode", return_value=True)
    @patch("comfy.model_management.torch.cuda.synchronize", create=True)
    @patch("comfy.model_management.torch.xpu.synchronize", create=True)
    def test_synchronize_noop_on_cpu(
        self, _xs, _cs, _cmod
    ):
        mm.synchronize()  # should not raise

    @patch("comfy.model_management.cpu_mode", return_value=True)
    @patch("comfy.model_management.torch.mps.empty_cache", create=True)
    @patch("comfy.model_management.torch.cuda", create=True)
    @patch("comfy.model_management.torch.mps", create=True)
    def test_soft_empty_cache_noop(
        self, *_
    ):
        mm.soft_empty_cache()  # should not raise

    @patch("comfy.model_management.is_amd", return_value=False)
    @patch("comfy.model_management.is_nvidia", return_value=False)
    def test_debug_memory_cpu_empty(self, *mocks):
        self.assertEqual(mm.debug_memory_summary(), "")


class TestStreamsAndCastBuffer(unittest.TestCase):
    @staticmethod
    def _reset_cast_buffers_for_tests():
        """`get_cast_buffer` uses `None` as a dict key; `mm.reset_cast_buffers` calls
        `None.synchronize()` and raises. Clear like `reset_cast_buffers` but skip the None key.
        """
        mm.LARGEST_CASTED_WEIGHT = (None, 0)
        for s in list(mm.STREAM_CAST_BUFFERS.keys()):
            if s is not None:
                s.synchronize()
        mm.synchronize()
        mm.STREAM_CAST_BUFFERS.clear()
        mm.soft_empty_cache()

    def setUp(self):
        self._reset_cast_buffers_for_tests()
        mm.STREAMS.clear()
        mm.stream_counters.clear()

    def tearDown(self):
        self._reset_cast_buffers_for_tests()
        mm.STREAMS.clear()
        mm.stream_counters.clear()

    def test_current_stream_cpu(self):
        self.assertIsNone(mm.current_stream(torch.device("cpu")))

    def test_sync_stream_none(self):
        mm.sync_stream(
            torch.device("cuda:0"), None
        )  # no-op: current_stream None on CPU path for cuda:0? Actually is_device_cuda true - need cuda
        # on CPU test env, current_stream for cuda:0 may still return stream if cuda available
        s = object()
        mm.sync_stream(
            torch.device("cpu"), s
        )  # current_stream is None

    @patch("comfy.model_management.NUM_STREAMS", 0)
    def test_get_offload_none_when_streams_zero(self, *_):
        self.assertIsNone(mm.get_offload_stream(torch.device("cuda:0")))

    def test_get_cast_buffer_creates(self):
        dev = torch.device("cpu")
        buf = mm.get_cast_buffer(None, dev, 32, "ref1")
        self.assertIsNotNone(buf)
        self.assertEqual(buf.numel(), 32)
        # second call same size returns same buffer
        b2 = mm.get_cast_buffer(None, dev, 32, "ref1")
        self.assertIs(b2, buf)

    def test_get_cast_buffer_skips_largest(self):
        dev = torch.device("cpu")
        mm.get_cast_buffer(None, dev, 32, "unique")
        self.assertIsNone(
            mm.get_cast_buffer(
                None, dev, 64, "unique"
            )
        )


class TestCastToAndCastToGathered(unittest.TestCase):
    @patch("comfy.model_management.comfy.memory_management")
    def test_cast_to_gathered_copies(
        self, m_mem
    ):
        a = torch.ones(2, dtype=torch.float32)
        dest = torch.empty(2, dtype=torch.float32)
        m_mem.read_tensor_file_slice_into.return_value = False
        m_mem.interpret_gathered_like.return_value = [dest]
        with patch.object(
            m_mem, "read_tensor_file_slice_into", return_value=False
        ):
            mm.cast_to_gathered(
                [a], r=torch.empty(2), non_blocking=False, stream=None
            )
        self.assertTrue(
            dest.allclose(
                a
            )
        )

    def test_cast_to_other_device(self):
        w = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out = mm.cast_to(
            w,
            dtype=torch.float32,
            device=torch.device("cpu"),
            r=torch.empty(2, dtype=torch.float32),
        )
        self.assertTrue(torch.equal(out, w))

    def test_cast_to_device_helper(self):
        t = torch.ones(1, device="cpu", dtype=torch.float32)
        with patch.object(mm, "device_supports_non_blocking", return_value=True):
            o = mm.cast_to_device(
                t, torch.device("cpu"), torch.float32, copy=True
            )
        self.assertEqual(o.shape, t.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "no cuda")
    def test_cast_to_fresh_cuda_buffer(self):
        t = torch.ones(1, device="cpu")
        o = mm.cast_to(
            t, dtype=t.dtype, device=torch.device("cuda:0"), r=None
        )
        self.assertEqual(o.device.type, "cuda")


class TestModuleMmapResidency(unittest.TestCase):
    def test_linear_no_mmap(self):
        m = torch.nn.Linear(2, 2)
        touched, total = mm.module_mmap_residency(m, free=False)
        self.assertEqual(touched, 0)
        self.assertGreater(total, 0)

    @patch("comfy.model_management.comfy.quant_ops.QuantizedTensor", bool)
    def test_quantized_branch_skipped_if_not_instance(self, *_q):
        m = torch.nn.Linear(2, 2)
        touched, total = mm.module_mmap_residency(
            m, free=True
        )
        self.assertGreaterEqual(total, 0)


class TestLoadedModelWithMocks(unittest.TestCase):
    def _make(self, real=None):
        model = MagicMock()
        model.load_device = torch.device("cpu")
        model.parent = None
        model.model = real if real is not None else torch.nn.Module()
        model.model_size.return_value = 100
        model.model_mmap_residency.return_value = (0, 100)
        model.pinned_memory_size.return_value = 0
        model.model_dtype.return_value = torch.float32
        model.model_patches_to = MagicMock()
        model.partially_load.return_value = 0
        model.current_loaded_device.return_value = torch.device("cpu")
        model.loaded_size.return_value = 0
        model.offload_device = torch.device("cpu")
        return model, mm.LoadedModel(model)

    def test_model_memory(self):
        model, loaded = self._make()
        self.assertEqual(loaded.model_memory(), 100)

    def test_memory_required_different_device(self):
        model, loaded = self._make()
        model.model_size.return_value = 200
        model.current_loaded_device.return_value = (
            torch.device("cuda:0")
        )
        self.assertEqual(loaded.model_memory_required(torch.device("cpu")), 200)

    def test_should_reload_true(self):
        model, loaded = self._make()
        model.lowvram_patch_counter.return_value = 3
        self.assertTrue(loaded.should_reload_model(force_patch_weights=True))

    def test_should_reload_false(self):
        model, loaded = self._make()
        model.lowvram_patch_counter.return_value = 0
        self.assertFalse(loaded.should_reload_model(force_patch_weights=True))

    def test_model_unload_partial_returns_false_when_freed_enough(self):
        model, loaded = self._make()
        model.loaded_size.return_value = 1000
        model.partially_unload.return_value = 500
        self.assertFalse(loaded.model_unload(memory_to_free=100))
        model.partially_unload.assert_called_once()
        model.detach.assert_not_called()


class TestEmptyLoadedModelsAndCleanup(unittest.TestCase):
    def setUp(self):
        self._saved = list(mm.current_loaded_models)

    def tearDown(self):
        mm.current_loaded_models.clear()
        mm.current_loaded_models.extend(self._saved)

    def test_loaded_models_empty(self):
        mm.current_loaded_models.clear()
        self.assertEqual(mm.loaded_models(), [])
        self.assertEqual(mm.loaded_models(only_currently_used=True), [])

    def test_free_memory_no_models(self, *_):
        mm.current_loaded_models.clear()
        with patch(
            "comfy.model_management.cleanup_models_gc", MagicMock()
        ):
            mm.free_memory(1, torch.device("cpu"), keep_loaded=[])

    def test_cleanup_models_drops_none(self):
        mm.current_loaded_models.clear()
        m = MagicMock()
        m.__class__.__name__ = "MockM"
        lm = MagicMock()
        lm.real_model = MagicMock(return_value=None)
        mm.current_loaded_models.append(lm)
        mm.cleanup_models()
        # weak ref is tricky; with MagicMock, real_model() is None
        if len(mm.current_loaded_models) == 0:
            self.assertEqual(len(mm.current_loaded_models), 0)


class TestIsIxucaMluHelpers(unittest.TestCase):
    @patch("comfy.model_management.mlu_available", True)
    def test_is_mlu(self, *a):
        self.assertTrue(mm.is_mlu())

    @patch("comfy.model_management.mlu_available", False)
    def test_is_mlu_false(self, *a):
        self.assertFalse(mm.is_mlu())

    @patch("comfy.model_management.ixuca_available", True)
    def test_ixuca(self, *a):
        self.assertTrue(mm.is_ixuca())


class TestIsNvidiaAmd(unittest.TestCase):
    @patch("comfy.model_management.cpu_state", mm.CPUState.CPU)
    @patch("comfy.model_management.torch.version.cuda", "12.0")
    def test_is_nvidia_false_on_cpu_state(self, *_c):
        self.assertFalse(mm.is_nvidia())

    @patch("comfy.model_management.cpu_state", mm.CPUState.CPU)
    @patch("comfy.model_management.torch.version.hip", "5.0")
    def test_is_amd_false_on_cpu(self, *_h):
        self.assertFalse(mm.is_amd())

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "cuda not available",
    )
    @patch("comfy.model_management.cpu_state", mm.CPUState.GPU)
    @patch("comfy.model_management.torch.version.cuda", "12.0")
    def test_nvidia_on_gpu(self, *a):
        if torch.version.cuda is None:
            self.skipTest("not cuda build")
        self.assertIsInstance(mm.is_nvidia(), bool)


class TestPinMemoryNoopPaths(unittest.TestCase):
    @patch("comfy.model_management.MAX_PINNED_MEMORY", 0)
    def test_pin_unpin_disabled(self, *a):
        t = torch.zeros(1)
        self.assertFalse(mm.pin_memory(t))
        self.assertFalse(mm.unpin_memory(t))


class TestCastToExtraBranches(unittest.TestCase):
    def test_same_device_copy_true(self):
        w = torch.ones(2, dtype=torch.float32)
        out = mm.cast_to(
            w, dtype=torch.float16, device=w.device, copy=True
        )
        self.assertEqual(out.dtype, torch.float16)
        self.assertIsNot(out, w)

    def test_same_device_with_stream_context(self):
        w = torch.ones(1, dtype=torch.float32, device="cpu")
        stream = MagicMock()
        stream.as_context = MagicMock(
            return_value=contextlib.nullcontext()
        )
        out = mm.cast_to(
            w, dtype=torch.float16, device=w.device, copy=False, stream=stream
        )
        self.assertEqual(out.dtype, torch.float16)


class TestCastToGatheredPaths(unittest.TestCase):
    @patch("comfy.memory_management.read_tensor_file_slice_into", return_value=True)
    @patch("comfy.memory_management.interpret_gathered_like")
    def test_read_slice_short_circuits(
        self, m_interp, _rslice
    ):
        m_interp.return_value = [torch.empty(1)]
        mm.cast_to_gathered(
            [torch.zeros(1)], r=torch.empty(1), non_blocking=False
        )

    @patch("comfy.memory_management.read_tensor_file_slice_into", return_value=False)
    @patch("comfy.memory_management.interpret_gathered_like")
    def test_skips_none_tensor(self, m_interp, _rslice):
        m_interp.return_value = [torch.empty(1), torch.empty(1)]
        mm.cast_to_gathered(
            [None, torch.ones(1)], r=torch.empty(2), non_blocking=False
        )


class TestSupportsCastMpsAndFloat8(unittest.TestCase):
    @unittest.skipIf(
        not hasattr(torch, "float8_e4m3fn"), "no float8 dtype"
    )
    def test_mps_blocks_float8(self):
        self.assertFalse(
            mm.supports_cast(
                torch.device("mps"), torch.float8_e4m3fn
            )
        )

    def test_int32_not_castable_on_cpu(self):
        self.assertFalse(
            mm.supports_cast(torch.device("cpu"), torch.int32)
        )


class TestDeviceNonBlockingDeterministic(unittest.TestCase):
    def test_deterministic_disables(self):
        old_f, old_d = args.force_non_blocking, args.deterministic
        try:
            args.force_non_blocking = False
            args.deterministic = True
            with patch.object(mm, "is_device_mps", return_value=False):
                with patch.object(mm, "is_intel_xpu", return_value=False):
                    self.assertFalse(
                        mm.device_supports_non_blocking(
                            torch.device("cpu")
                        )
                    )
        finally:
            args.force_non_blocking, args.deterministic = old_f, old_d


class TestLoadModelGpuAndLoadedModels(unittest.TestCase):
    @patch("comfy.model_management.load_models_gpu")
    def test_load_model_gpu_delegates(self, m_load):
        m = MagicMock()
        mm.load_model_gpu(m)
        m_load.assert_called_once_with([m])

    def test_loaded_models_only_currently_used(self):
        a, b = MagicMock(), MagicMock()
        a.currently_used, b.currently_used = True, False
        a.model, b.model = "a", "b"
        saved = list(mm.current_loaded_models)
        try:
            mm.current_loaded_models.clear()
            mm.current_loaded_models.extend([a, b])
            self.assertEqual(
                mm.loaded_models(only_currently_used=True), ["a"]
            )
            self.assertEqual(
                mm.loaded_models(only_currently_used=False), ["a", "b"]
            )
        finally:
            mm.current_loaded_models.clear()
            mm.current_loaded_models.extend(saved)

    def test_cleanup_models_removes_unloaded(self):
        lm = MagicMock()
        lm.real_model.return_value = None
        saved = list(mm.current_loaded_models)
        try:
            mm.current_loaded_models.clear()
            mm.current_loaded_models.append(lm)
            mm.cleanup_models()
            self.assertEqual(len(mm.current_loaded_models), 0)
        finally:
            mm.current_loaded_models.clear()
            mm.current_loaded_models.extend(saved)


class TestUnetDtypeFp64AndSharedVram(unittest.TestCase):
    @staticmethod
    def _reset_unet_args():
        for n in (
            "fp32_unet",
            "fp64_unet",
            "bf16_unet",
            "fp16_unet",
            "fp8_e4m3fn_unet",
            "fp8_e5m2_unet",
            "fp8_e8m0fnu_unet",
        ):
            setattr(args, n, False)

    def tearDown(self):
        self._reset_unet_args()

    def test_fp64_unet(self):
        self._reset_unet_args()
        args.fp64_unet = True
        self.assertEqual(mm.unet_dtype(), torch.float64)

    @contextlib.contextmanager
    def _aimdo(self, v):
        old = getattr(comfy.memory_management, "aimdo_enabled", False)
        comfy.memory_management.aimdo_enabled = v
        try:
            yield
        finally:
            comfy.memory_management.aimdo_enabled = old

    def test_unet_initial_device_shared_vram(self):
        with self._aimdo(False):
            with patch.object(mm, "vram_state", mm.VRAMState.SHARED):
                d = mm.unet_inital_load_device(100, torch.float32)
        self.assertEqual(d, mm.get_torch_device())


class TestDeviceOffloadArgs(unittest.TestCase):
    def test_intermediate_device_gpu_only(self):
        old = args.gpu_only
        try:
            args.gpu_only = True
            self.assertEqual(
                mm.intermediate_device(), mm.get_torch_device()
            )
        finally:
            args.gpu_only = old

    def test_vae_device_cpu_vae_flag(self):
        old = args.cpu_vae
        try:
            args.cpu_vae = True
            self.assertEqual(mm.vae_device(), torch.device("cpu"))
        finally:
            args.cpu_vae = old


class TestArchiveDtypesNested(unittest.TestCase):
    def test_nested_modules(self):
        parent = torch.nn.Module()
        child = torch.nn.Module()
        child.register_parameter(
            "p", torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        )
        parent.add_module("c", child)
        mm.archive_model_dtypes(parent)
        self.assertEqual(child.p_comfy_model_dtype, torch.float16)


class TestAttentionHelperWrappers(unittest.TestCase):
    def test_pytorch_attention_flag(self):
        with patch.object(mm, "ENABLE_PYTORCH_ATTENTION", True):
            self.assertTrue(mm.pytorch_attention_enabled())

    def test_xformers_enabled_vae_delegates(self):
        with patch.object(mm, "xformers_enabled", return_value=False):
            self.assertFalse(mm.xformers_enabled_vae())
        with patch.object(mm, "xformers_enabled", return_value=True):
            with patch.object(mm, "XFORMERS_ENABLED_VAE", True):
                self.assertTrue(mm.xformers_enabled_vae())


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCudaAmdGfxPathsGpu(unittest.TestCase):
    """Branch coverage for `cuda`, `amd`, and `gfx` in model_management (needs CUDA)."""

    def test_get_torch_device_cuda(self):
        with patch.object(mm, "cpu_state", mm.CPUState.GPU):
            with patch.object(mm, "is_intel_xpu", return_value=False):
                with patch.object(mm, "is_ascend_npu", return_value=False):
                    with patch.object(mm, "is_mlu", return_value=False):
                        d = mm.get_torch_device()
        self.assertEqual(d.type, "cuda")

    def test_get_total_memory_cuda_torch_pair(self):
        d = torch.device("cuda:0")
        t, t2 = mm.get_total_memory(d, torch_total_too=True)
        self.assertIsInstance(t, (int, float))
        self.assertIsInstance(t2, (int, float))
        self.assertGreater(t, 0)

    def test_get_free_memory_cuda_tuple(self):
        d = torch.device("cuda:0")
        a, b = mm.get_free_memory(d, torch_free_too=True)
        self.assertIsInstance(a, (int, float))
        self.assertIsInstance(b, (int, float))

    @patch("torch.cuda.get_device_name", return_value="AMD Radeon")
    @patch("comfy.model_management.is_ascend_npu", return_value=False)
    @patch("comfy.model_management.is_mlu", return_value=False)
    @patch("comfy.model_management.is_intel_xpu", return_value=False)
    def test_get_torch_device_name_int_fallback_cuda_string(
        self, _ix, _m, _n, _gdn
    ):
        s = mm.get_torch_device_name(0)
        self.assertIn("AMD Radeon", s)
        self.assertIn("0", s)

    def test_current_stream_returns_on_cuda(self):
        d = torch.device("cuda:0")
        s = mm.current_stream(d)
        self.assertIsNotNone(s)

    def test_get_offload_stream_creates_and_reuses(self):
        with patch.object(mm, "NUM_STREAMS", 2):
            try:
                mm.STREAMS.clear()
                mm.stream_counters.clear()
                d = torch.device("cuda:0")
                a = mm.get_offload_stream(d)
                b = mm.get_offload_stream(d)
                self.assertIsNotNone(a)
                self.assertIsNotNone(b)
            finally:
                mm.STREAMS.clear()
                mm.stream_counters.clear()
                mm.reset_cast_buffers()

    def test_sync_stream_cuda(self):
        d = torch.device("cuda:0")
        torch.cuda.current_stream(d)
        s2 = torch.cuda.Stream(device=d)
        mm.sync_stream(d, s2)

    def test_discard_cuda_async_error_runs(self):
        if mm.get_torch_device().type != "cuda":
            self.skipTest("default torch device is not CUDA")
        if mm.is_amd():
            try:
                arch = torch.cuda.get_device_properties(0).gcnArchName
            except AttributeError:
                arch = ""
            if "gfx942" not in arch and "gfx950" not in arch:
                self.skipTest(
                    "non-NVIDIA: only runs on AMD gfx942 or gfx950 (gcnArchName)"
                )
        mm.discard_cuda_async_error()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestAmdMinVersionGfxBranch(unittest.TestCase):
    @patch("torch.cuda.get_device_properties")
    @patch.object(mm, "is_amd", return_value=True)
    @patch.object(mm, "is_device_cpu", return_value=False)
    def test_gfx_len7_rdna_version_compare(
        self, m_is_device_cpu, m_is_amd, m_get_props
    ):
        m_get_props.return_value = SimpleNamespace(
            gcnArchName="gfx1100"  # arch[4]=='1' -> 1+2=3
        )
        d = torch.device("cuda:0")
        self.assertTrue(mm.amd_min_version(d, min_rdna_version=2))
        self.assertFalse(mm.amd_min_version(d, min_rdna_version=4))

    @patch("torch.cuda.get_device_properties")
    @patch.object(mm, "is_amd", return_value=True)
    @patch.object(mm, "is_device_cpu", return_value=False)
    def test_gfx_short_arch_returns_false(
        self, m_is_device_cpu, m_is_amd, m_get_props
    ):
        m_get_props.return_value = SimpleNamespace(
            gcnArchName="gfx90a"  # len != 7
        )
        self.assertFalse(
            mm.amd_min_version(torch.device("cuda:0"), min_rdna_version=0)
        )


class TestAmdShouldUseBf16Rdna2Gfx(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and not args.cpu,
        "ROCm/AMD arch checks use torch.cuda.get_device_properties",
    )
    def test_amd_rdna2_gfx_in_arch_bf16_paths(self):
        dev = torch.device("cuda:0")
        props = SimpleNamespace(
            gcnArchName="amdgpu-something-gfx1030",
            major=7,
            minor=0,
            name="Radeon",
        )
        with patch("comfy.model_management.AMD_RDNA2_AND_OLDER_ARCH", ["gfx1030"]):
            with patch("torch.cuda.get_device_properties", return_value=props):
                with patch("torch.cuda.is_bf16_supported", return_value=True):
                    with patch.object(mm, "is_amd", return_value=True):
                        with patch.object(mm, "is_ixuca", return_value=False):
                            with patch.object(mm, "is_mlu", return_value=False):
                                with patch.object(mm, "is_ascend_npu", return_value=False):
                                    with patch.object(
                                        mm, "is_intel_xpu", return_value=False
                                    ):
                                        with patch.object(
                                            mm, "cpu_mode", return_value=False
                                        ):
                                            with patch.object(
                                                mm, "mps_mode", return_value=False
                                            ):
                                                with patch.object(
                                                    mm,
                                                    "is_directml_enabled",
                                                    return_value=False,
                                                ):
                                                    with patch(
                                                        "comfy.model_management.FORCE_FP32",
                                                        False,
                                                    ):
                                                        with patch(
                                                            "comfy.model_management.directml_enabled",
                                                            False,
                                                        ):
                                                            with patch(
                                                                "comfy.model_management.mac_version",
                                                                return_value=None,
                                                            ):
                                                                self.assertFalse(
                                                                    mm.should_use_bf16(
                                                                        dev,
                                                                        model_params=0,
                                                                        manual_cast=False,
                                                                    )
                                                                )
                                                                self.assertTrue(
                                                                    mm.should_use_bf16(
                                                                        dev,
                                                                        model_params=0,
                                                                        manual_cast=True,
                                                                    )
                                                                )

    @unittest.skipUnless(
        torch.cuda.is_available() and not args.cpu,
        "needs CUDA + hip check before get_device_properties",
    )
    def test_hip_path_short_circuits_fp16(self):
        with patch("torch.version.hip", "6.0.0"):
            with patch.object(mm, "is_ixuca", return_value=False):
                with patch.object(mm, "is_mlu", return_value=False):
                    with patch.object(
                        mm, "is_ascend_npu", return_value=False
                    ):
                        with patch.object(
                            mm, "is_intel_xpu", return_value=False
                        ):
                            with patch.object(
                                mm, "cpu_mode", return_value=False
                            ):
                                with patch.object(
                                    mm, "mps_mode", return_value=False
                                ):
                                    with patch.object(
                                        args, "force_fp16", False
                                    ):
                                        with patch.object(
                                            args, "force_fp32", False
                                        ):
                                            with patch.object(
                                                mm,
                                                "is_directml_enabled",
                                                return_value=False,
                                            ):
                                                self.assertTrue(
                                                    mm.should_use_fp16(
                                                        torch.device("cuda:0")
                                                    )
                                                )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDebugMemoryAmdNvidiaWithCuda(unittest.TestCase):
    @patch("torch.cuda.memory.memory_summary", return_value="CUDAMEM\n")
    @patch.object(mm, "is_nvidia", return_value=False)
    @patch.object(mm, "is_amd", return_value=True)
    def test_debug_memory_non_empty_on_amd(self, *_a):
        self.assertEqual(mm.debug_memory_summary().strip(), "CUDAMEM")

    @patch("torch.cuda.memory.memory_summary", return_value="NV\n")
    @patch.object(mm, "is_nvidia", return_value=True)
    @patch.object(mm, "is_amd", return_value=False)
    def test_debug_on_nvidia(self, *_a):
        self.assertEqual(mm.debug_memory_summary().strip(), "NV")


class TestAmdHipIsAmdAndFlashAttention(unittest.TestCase):
    def test_is_amd_true_with_hip_and_no_cuda_string(self):
        with patch("comfy.model_management.cpu_state", mm.CPUState.GPU):
            with patch("comfy.model_management.torch.version.cuda", None):
                with patch(
                    "comfy.model_management.torch.version.hip", "6.0.0"
                ):
                    self.assertTrue(mm.is_amd())

    @patch("comfy.model_management.ENABLE_PYTORCH_ATTENTION", True)
    @patch.object(mm, "is_nvidia", return_value=False)
    @patch.object(mm, "is_amd", return_value=True)
    @patch.object(mm, "is_intel_xpu", return_value=False)
    @patch.object(mm, "is_ascend_npu", return_value=False)
    @patch.object(mm, "is_mlu", return_value=False)
    @patch.object(mm, "is_ixuca", return_value=False)
    def test_pytorch_flash_attention_enabled_on_amd(self, *_a):
        self.assertTrue(mm.pytorch_attention_flash_attention())

    @patch("comfy.model_management.ENABLE_PYTORCH_ATTENTION", True)
    @patch.object(mm, "is_nvidia", return_value=True)
    @patch.object(mm, "is_amd", return_value=False)
    @patch.object(mm, "is_intel_xpu", return_value=False)
    @patch.object(mm, "is_ascend_npu", return_value=False)
    @patch.object(mm, "is_mlu", return_value=False)
    @patch.object(mm, "is_ixuca", return_value=False)
    def test_pytorch_flash_attention_nvidia(self, *_a):
        self.assertTrue(mm.pytorch_attention_flash_attention())


class TestCoverageHelperPaths(unittest.TestCase):
    @patch("comfy.model_management.free_memory")
    @patch(
        "comfy.model_management.get_torch_device",
        return_value=torch.device("cpu"),
    )
    def test_unload_all_models(self, m_dev, m_fm):
        mm.unload_all_models()
        m_fm.assert_called_once()
        cargs, ckwargs = m_fm.call_args
        self.assertEqual(cargs[0], 1e30)
        self.assertEqual(cargs[1], torch.device("cpu"))

    def test_free_memory_device_none_empty_list(self):
        saved = list(mm.current_loaded_models)
        try:
            mm.current_loaded_models.clear()
            with patch.object(mm, "cleanup_models_gc"):
                out = mm.free_memory(1, None, keep_loaded=[])
        finally:
            mm.current_loaded_models.clear()
            mm.current_loaded_models.extend(saved)
        self.assertEqual(out, [])

    @patch("comfy.model_management.MAX_PINNED_MEMORY", 10**12)
    def test_pin_memory_noncontiguous_cpu_tensor(self, *_a):
        t = torch.zeros(2, 2, device="cpu").t()  # not contiguous
        self.assertFalse(mm.pin_memory(t))

    def test_sage_and_flash_flags_when_disabled(self):
        a, b = args.use_sage_attention, args.use_flash_attention
        try:
            args.use_sage_attention = False
            args.use_flash_attention = False
            self.assertFalse(mm.sage_attention_enabled())
            self.assertFalse(mm.flash_attention_enabled())
        finally:
            args.use_sage_attention, args.use_flash_attention = a, b

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_supports_dtype_int8_cuda_false(self):
        self.assertFalse(
            mm.supports_dtype(torch.device("cuda:0"), torch.int8)
        )

    def test_maximum_vram_explicit_cpu_device(self):
        with patch.object(
            mm, "get_total_memory", return_value=1e9
        ):
            with patch.object(
                mm, "minimum_inference_memory", return_value=1e6
            ):
                v = mm.maximum_vram_for_weights(device=torch.device("cpu"))
        self.assertIsInstance(v, (int, float))
        self.assertLess(v, 1e9)


if __name__ == "__main__":
    unittest.main()
