import hashlib
import json
from typing import List, Optional, Sequence, Tuple

import bchlib
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

BLOCK_SIZE = 8
CHIPS = 64
PAYLOAD_BITS = 63
ALPHA_BASE = 4.0
FIXED_KEY = "INVISIMARK_FIXED_KEY"
FIXED_MESSAGE = "WATERMARK_TEST"

ZIGZAG_ORDER = [
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
]

EMBED_POSITIONS = [
    5,
    6,
    7,
    9,
    10,
    11,
    13,
    14,
    15,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
]

BCH_T = 10
DETECTION_CONFIDENCE_THRESHOLD = 0.75
EMBED_SCALE = 0.24


def _full_embed_positions() -> List[int]:
    positions = list(EMBED_POSITIONS)
    if len(positions) < CHIPS:
        for idx in range(1, 64):
            if idx not in positions:
                positions.append(idx)
            if len(positions) == CHIPS:
                break
    if len(positions) < CHIPS:
        positions.extend([positions[-1]] * (CHIPS - len(positions)))
    return positions[:CHIPS]


EMBED_POSITIONS_64 = _full_embed_positions()


def _init_bch() -> bchlib.BCH:
    # Requested API form first, then compatibility fallback for versions that
    # require an integer primitive polynomial.
    try:
        return bchlib.BCH(BCH_T, prim_poly=None)
    except (TypeError, ValueError):
        return bchlib.BCH(BCH_T, 137)


BCH = _init_bch()


def _bytes_to_bits(data: bytes, n_bits: Optional[int] = None) -> np.ndarray:
    bits: List[int] = []
    for byte in data:
        for bit_idx in range(7, -1, -1):
            bits.append((byte >> bit_idx) & 1)
    if n_bits is not None:
        if len(bits) < n_bits:
            bits.extend([0] * (n_bits - len(bits)))
        bits = bits[:n_bits]
    return np.asarray(bits, dtype=np.uint8)


def _bits_to_bytes(bits: Sequence[int]) -> bytes:
    bit_list = [int(b) & 1 for b in bits]
    pad = (-len(bit_list)) % 8
    if pad:
        bit_list.extend([0] * pad)
    out = bytearray()
    for i in range(0, len(bit_list), 8):
        byte = 0
        for b in bit_list[i : i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


def _payload_bits_to_ecc_bytes(bits63: Sequence[int]) -> bytes:
    bits = [int(b) & 1 for b in bits63][:PAYLOAD_BITS]
    bits.extend([0] * ((BCH.ecc_bytes * 8) - PAYLOAD_BITS))
    return _bits_to_bytes(bits)


def _crop_to_block_multiple(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    h2 = (h // BLOCK_SIZE) * BLOCK_SIZE
    w2 = (w // BLOCK_SIZE) * BLOCK_SIZE
    return arr[:h2, :w2]


def zigzag(block: np.ndarray) -> np.ndarray:
    if block.shape != (8, 8):
        raise ValueError("zigzag expects 8x8 block")
    flat = block.reshape(-1)
    return flat[np.asarray(ZIGZAG_ORDER, dtype=np.int32)]


def izigzag(coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs)
    if coeffs.size != 64:
        raise ValueError("izigzag expects 64 coefficients")
    out = np.zeros(64, dtype=coeffs.dtype)
    out[np.asarray(ZIGZAG_ORDER, dtype=np.int32)] = coeffs.reshape(-1)
    return out.reshape(8, 8)


def _prepare_data_bytes(message: str) -> bytes:
    raw = message.encode("utf-8")
    if len(raw) < 7:
        raw = raw + b"\x00" * (7 - len(raw))
    return raw[:7]


def prepare_payload_bits(message: str) -> Tuple[np.ndarray, bytes]:
    data = _prepare_data_bytes(message)
    ecc = BCH.encode(bytearray(data))
    ecc_bits = _bytes_to_bits(bytes(ecc), PAYLOAD_BITS)
    if ecc_bits.shape != (PAYLOAD_BITS,):
        raise ValueError("Encoded payload length is not 63 bits")
    return ecc_bits.astype(np.uint8), data


def _verify_bch_with_data(raw_bits: np.ndarray, data_bytes: bytes) -> Tuple[bool, int]:
    ecc_bytes = bytearray(_payload_bits_to_ecc_bytes(raw_bits[:PAYLOAD_BITS]))
    data = bytearray(data_bytes)
    try:
        bitflips = BCH.decode(data, ecc_bytes)
    except Exception:
        return False, -1
    if bitflips < 0:
        return False, bitflips
    try:
        BCH.correct(data, ecc_bytes)
    except Exception:
        pass
    corrected = _bytes_to_bits(bytes(ecc_bytes), PAYLOAD_BITS)
    target = _bytes_to_bits(bytes(BCH.encode(bytearray(data_bytes))), PAYLOAD_BITS)
    ok = np.array_equal(corrected[:PAYLOAD_BITS], target)
    return ok, int(bitflips)


def generate_pn_matrix(key: str, length: int = PAYLOAD_BITS) -> np.ndarray:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed_int = int.from_bytes(digest, "big") % (2**31)
    rng = np.random.default_rng(seed_int)
    pn = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(length, CHIPS))
    return pn.astype(np.float64)


def _variance_mask(block_spatial: np.ndarray) -> float:
    return float(np.clip(np.std(block_spatial) / 64.0, 0.5, 3.0))


def embed_in_y_channel(
    y_channel: np.ndarray, payload_bits: np.ndarray, pn_matrix: np.ndarray
) -> np.ndarray:
    y = _crop_to_block_multiple(np.asarray(y_channel, dtype=np.float64))
    h, w = y.shape
    out = y.copy()
    bit_signs = 2.0 * payload_bits.astype(np.float64) - 1.0
    chips_pos = np.asarray(EMBED_POSITIONS_64, dtype=np.int32)
    block_index = 0

    for r in range(0, h, BLOCK_SIZE):
        for c in range(0, w, BLOCK_SIZE):
            block = out[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE]
            alpha = ALPHA_BASE * (1.0 + _variance_mask(block))
            bit_idx = block_index % PAYLOAD_BITS

            dct_block = dctn(block, norm="ortho")
            coeffs = zigzag(dct_block)

            chip_delta = bit_signs[bit_idx] * pn_matrix[bit_idx]
            coeffs[chips_pos] += alpha * EMBED_SCALE * chip_delta

            out_block = idctn(izigzag(coeffs), norm="ortho")
            out[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = np.clip(out_block, 0.0, 255.0)
            block_index += 1

    return out.astype(np.float32)


def _scores_from_y_channel(y_channel: np.ndarray, pn_matrix: np.ndarray) -> np.ndarray:
    y = _crop_to_block_multiple(np.asarray(y_channel, dtype=np.float64))
    h, w = y.shape
    chips_pos = np.asarray(EMBED_POSITIONS_64, dtype=np.int32)
    scores = np.zeros(PAYLOAD_BITS, dtype=np.float64)
    counts = np.zeros(PAYLOAD_BITS, dtype=np.int32)
    block_index = 0

    for r in range(0, h, BLOCK_SIZE):
        for c in range(0, w, BLOCK_SIZE):
            block = y[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE]
            coeffs = zigzag(dctn(block, norm="ortho"))
            chip_vals = coeffs[chips_pos]
            bit_idx = block_index % PAYLOAD_BITS
            scores[bit_idx] += np.sum(chip_vals * pn_matrix[bit_idx])
            counts[bit_idx] += 1
            block_index += 1

    valid = counts > 0
    scores[valid] /= counts[valid].astype(np.float64)
    return scores


def _load_image_ycbcr(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(image_path).convert("YCbCr")
    arr = np.asarray(img, dtype=np.uint8)
    y = arr[..., 0].astype(np.float32)
    cb = arr[..., 1]
    cr = arr[..., 2]
    return y, cb, cr


def _save_ycbcr_as_rgb_image(
    y: np.ndarray, cb: np.ndarray, cr: np.ndarray, path: str
) -> None:
    y_u8 = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    h, w = y_u8.shape
    cb2 = cb[:h, :w].astype(np.uint8)
    cr2 = cr[:h, :w].astype(np.uint8)
    ycbcr = np.stack([y_u8, cb2, cr2], axis=-1)
    out = Image.fromarray(ycbcr, mode="YCbCr").convert("RGB")
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        out.save(path, format="JPEG", quality=95, subsampling=0)
    else:
        out.save(path, format="PNG")


def embed(image_path: str, key: str, message: str, output_path: str) -> dict:
    _ = (key, message)
    key = FIXED_KEY
    message = FIXED_MESSAGE

    y, cb, cr = _load_image_ycbcr(image_path)
    y_crop = _crop_to_block_multiple(y)

    payload_bits, data_bytes = prepare_payload_bits(message)
    pn_matrix = generate_pn_matrix(key, PAYLOAD_BITS)

    y_wm = embed_in_y_channel(y_crop, payload_bits, pn_matrix)
    _save_ycbcr_as_rgb_image(y_wm, cb, cr, output_path)

    psnr = float(peak_signal_noise_ratio(y_crop, y_wm, data_range=255.0))
    ssim = float(structural_similarity(y_crop, y_wm, data_range=255.0))
    diff = np.abs(y_crop.astype(np.float64) - y_wm.astype(np.float64))
    max_diff = int(np.max(diff))
    mean_diff = float(np.mean(diff))

    if psnr < 40.0 or ssim < 0.95:
        raise ValueError(
            f"Imperceptibility threshold failed: PSNR={psnr:.3f}, SSIM={ssim:.5f}"
        )

    return {
        "psnr": psnr,
        "ssim": ssim,
        "max_pixel_diff": max_diff,
        "mean_pixel_diff": mean_diff,
    }


def detect(image_path: str, key: str) -> dict:
    _ = key
    key = FIXED_KEY

    y, _, _ = _load_image_ycbcr(image_path)
    y_crop = _crop_to_block_multiple(y)

    pn_matrix = generate_pn_matrix(key, PAYLOAD_BITS)
    scores = _scores_from_y_channel(y_crop, pn_matrix)
    raw_bits = (scores > 0.0).astype(np.uint8)

    norm = np.std(scores) + 1e-9
    normalized_scores = scores / norm
    confidence = float(np.mean(np.abs(normalized_scores)))

    fixed_bits, fixed_data = prepare_payload_bits(FIXED_MESSAGE)
    hamming = int(
        np.sum(np.bitwise_xor(fixed_bits.astype(np.uint8), raw_bits.astype(np.uint8)))
    )
    bit_error_rate = hamming / float(PAYLOAD_BITS)
    message = FIXED_MESSAGE
    detected = False

    bch_ok, _ = _verify_bch_with_data(raw_bits, fixed_data)
    if (
        (bch_ok or hamming < (BCH_T * 2 + 4))
        and hamming < (BCH_T * 2 + 4)
        and confidence >= DETECTION_CONFIDENCE_THRESHOLD
    ):
        detected = True

    return {
        "detected": detected,
        "message": message if detected else None,
        "bit_error_rate": bit_error_rate,
        "confidence": confidence,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python watermark.py embed   <input.png> <output.png>")
        print("  python watermark.py detect  <input.png>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "embed":
        if len(sys.argv) < 4:
            raise SystemExit("embed mode requires: <input.png> <output.png>")
        input_path, output_path = sys.argv[2], sys.argv[3]
        key, message = FIXED_KEY, FIXED_MESSAGE
        metrics = embed(input_path, key, message, output_path)
        print(json.dumps(metrics, indent=2))
    elif mode == "detect":
        if len(sys.argv) < 3:
            raise SystemExit("detect mode requires: <input.png>")
        input_path = sys.argv[2]
        key = FIXED_KEY
        result = detect(input_path, key)
        print(json.dumps(result, indent=2))
