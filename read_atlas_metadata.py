"""
read_atlas_metadata.py

Standalone helper to read the SIB_AtlasMeta JSON metadata that Sphere Imposter
Baker embeds in atlas PNGs. Use this in your game pipeline or asset tools.

No external dependencies — uses only Python stdlib (struct, zlib, json).

Usage:
    from read_atlas_metadata import read_atlas_metadata
    meta = read_atlas_metadata('sprite_atlas.png')
    print(meta['cols'], meta['rows'])
    for sprite in meta['sprites']:
        print(sprite['el'], sprite['az'], '→ cell', sprite['col'], sprite['row'])
"""

import struct
import zlib
import json


KEYWORD = b'SIB_AtlasMeta'


def read_atlas_metadata(filepath):
    """Return the embedded sprite metadata dict, or None if not found."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f"{filepath} is not a PNG file")

    pos = 8  # skip signature
    while pos < len(data) - 8:
        length = struct.unpack('>I', data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk_data = data[pos + 8:pos + 8 + length]

        if chunk_type == b'iTXt':
            try:
                null1 = chunk_data.index(b'\x00')
            except ValueError:
                pos += 8 + length + 4
                continue
            kw = chunk_data[:null1]
            if kw == KEYWORD:
                # Skip compression_flag + compression_method (2 bytes)
                rest = chunk_data[null1 + 3:]
                null_lang = rest.index(b'\x00')
                rest = rest[null_lang + 1:]
                null_trans = rest.index(b'\x00')
                text = rest[null_trans + 1:].decode('utf-8')
                return json.loads(text)

        pos += 8 + length + 4

    return None


def find_nearest_cell(meta, elevation_deg, azimuth_deg):
    """Pick the atlas cell whose camera direction is closest to the given angle.
    Returns the sprite dict (with col, row, x, y) for that cell.

    Uses great-circle distance on the unit sphere — handles azimuth wrap correctly.
    """
    import math
    el_rad = math.radians(elevation_deg)
    az_rad = math.radians(azimuth_deg)
    qx = math.cos(el_rad) * math.cos(az_rad)
    qy = math.cos(el_rad) * math.sin(az_rad)
    qz = math.sin(el_rad)

    best = None
    best_dot = -2.0
    for sprite in meta['sprites']:
        sel = math.radians(sprite['el'])
        saz = math.radians(sprite['az'])
        sx = math.cos(sel) * math.cos(saz)
        sy = math.cos(sel) * math.sin(saz)
        sz = math.sin(sel)
        dot = qx * sx + qy * sy + qz * sz
        if dot > best_dot:
            best_dot = dot
            best = sprite
    return best


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: read_atlas_metadata.py <atlas.png> [elevation_deg azimuth_deg]")
        sys.exit(1)
    meta = read_atlas_metadata(sys.argv[1])
    if meta is None:
        print("No SIB_AtlasMeta chunk found")
        sys.exit(1)
    print(f"Atlas: {meta['cols']}×{meta['rows']} cells, "
          f"{meta['sprite_size']}px each, "
          f"{meta['count']} sprites, "
          f"transparency: {meta['transparency']}")
    if len(sys.argv) >= 4:
        el = float(sys.argv[2])
        az = float(sys.argv[3])
        cell = find_nearest_cell(meta, el, az)
        print(f"Nearest cell to (el={el}, az={az}): "
              f"col={cell['col']} row={cell['row']} "
              f"(actual el={cell['el']} az={cell['az']})")
