"""
Sphere Imposter Baker (v5)

Render selected objects from a sphere of cameras arranged in latitude slices.
First and last slices are always single-camera poles.

v5 changes:
    - Camera preview: "Show Cameras" button creates empties at all camera
      positions before baking, so you can see the arrangement and verify
      framing through any of them. "Clear" removes them.
    - Verification sprite fixed:
        * placed with a small offset so it doesn't sit inside the object
        * show_in_front = True so it's always visible in the viewport
        * optional billboard constraint to the active camera
"""

import bpy
import os
import math
import json
import numpy as np
from bpy.types import PropertyGroup, Panel, Operator, UIList
from bpy.props import (
    IntProperty, FloatProperty, StringProperty, BoolProperty,
    EnumProperty, PointerProperty, CollectionProperty,
)
from mathutils import Vector, Matrix


PREVIEW_CAM_COLLECTION = "SIB_Preview_Cameras"


# ============================================================================
# VIEWPORT BILLBOARD TRACKING
# A timer continually rotates registered preview objects to face the
# active 3D viewport's view direction. There is no scene-camera object
# for the viewport view (it lives in RegionView3D), so a polling timer
# is the only reliable mechanism.
# ============================================================================

_SIB_BILLBOARD_OBJECTS = set()  # object names that should track the viewport
_SIB_TIMER_ACTIVE = False


def _viewport_view_position():
    """Return the world-space position of the active 3D viewport's camera, or None."""
    wm = bpy.context.window_manager
    for window in wm.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            if space is None or space.region_3d is None:
                continue
            rv3d = space.region_3d
            try:
                return rv3d.view_matrix.inverted().translation.copy()
            except (AttributeError, ValueError):
                continue
    return None


def _stable_billboard_rotation(direction):
    """Return an Euler rotation for a plane whose -Y axis faces `direction`,
    with its local Z axis anchored to a stable world-up reference.

    Why not direction.to_track_quat('-Y', 'Z')?
        At the poles the direction is parallel to world Z, the up constraint
        becomes degenerate, and Blender's algorithm picks an inconsistent
        rotation as the camera orbits — visually the sprite spins around
        its own normal axis.

    What this does instead:
        Builds the rotation matrix manually. Uses world Z as the up
        reference normally, smoothly blending toward world Y as the
        direction approaches a pole. The up reference depends only on
        |direction.z|, never on direction.x or direction.y — so as the
        camera orbits around the pole the up reference stays constant
        and the plane has no roll component.
    """
    forward = direction.normalized()
    abs_z = abs(forward.z)

    # Smoothly blend up reference from world Z (equator) to world Y (poles).
    # Blend zone is the last 5 degrees or so before the pole.
    if abs_z > 0.95:
        blend = (abs_z - 0.95) / 0.05  # 0..1
        up_ref = Vector((0.0, blend, 1.0 - blend))
        if up_ref.length < 0.0001:
            up_ref = Vector((0.0, 1.0, 0.0))
        up_ref = up_ref.normalized()
    else:
        up_ref = Vector((0.0, 0.0, 1.0))

    right = forward.cross(up_ref)
    if right.length < 0.0001:
        right = forward.cross(Vector((1.0, 0.0, 0.0)))
    right = right.normalized()

    up = right.cross(forward).normalized()

    # Plane's local axes in world coordinates:
    #   plane X = right, plane Y = -forward (since -Y faces camera), plane Z = up
    mat = Matrix((
        (right.x, -forward.x, up.x),
        (right.y, -forward.y, up.y),
        (right.z, -forward.z, up.z),
    ))
    return mat.to_euler()


def _viewport_billboard_tick():
    """Timer: rotate registered preview objects so their front face the viewport."""
    view_pos = _viewport_view_position()
    if view_pos is None:
        return 0.1

    to_remove = []
    for name in _SIB_BILLBOARD_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            to_remove.append(name)
            continue
        direction = view_pos - obj.matrix_world.translation
        if direction.length < 0.0001:
            continue
        # Stable rotation that doesn't spin the sprite at the poles.
        obj.rotation_euler = _stable_billboard_rotation(direction)

    for name in to_remove:
        _SIB_BILLBOARD_OBJECTS.discard(name)

    if not _SIB_BILLBOARD_OBJECTS:
        global _SIB_TIMER_ACTIVE
        _SIB_TIMER_ACTIVE = False
        return None  # stop the timer

    return 1.0 / 30  # 30 fps


def _register_billboard(obj_name):
    global _SIB_TIMER_ACTIVE
    _SIB_BILLBOARD_OBJECTS.add(obj_name)
    if not _SIB_TIMER_ACTIVE:
        try:
            bpy.app.timers.register(_viewport_billboard_tick,
                                     first_interval=0.1, persistent=True)
            _SIB_TIMER_ACTIVE = True
        except (RuntimeError, ValueError):
            pass


def _stop_billboard_timer():
    global _SIB_TIMER_ACTIVE
    _SIB_BILLBOARD_OBJECTS.clear()
    if bpy.app.timers.is_registered(_viewport_billboard_tick):
        try:
            bpy.app.timers.unregister(_viewport_billboard_tick)
        except (RuntimeError, ValueError):
            pass
    _SIB_TIMER_ACTIVE = False


# ============================================================================
# PROPERTY GROUPS
# ============================================================================

class SIB_Slice(PropertyGroup):
    elevation: FloatProperty(
        name="Elevation",
        description="Latitude angle. +90° = north pole, 0° = equator, -90° = south pole",
        default=0.0,
        min=math.radians(-90),
        max=math.radians(90),
        subtype='ANGLE', unit='ROTATION',
    )
    camera_count: IntProperty(
        name="Cameras", default=8, min=1, max=128,
    )
    azimuth_offset: FloatProperty(
        name="Azimuth Offset",
        default=0.0, min=0, max=math.radians(360),
        subtype='ANGLE', unit='ROTATION',
    )


def _redistribute_slices(settings):
    target = settings.slice_count
    while len(settings.slices) < target:
        s = settings.slices.add()
        s.camera_count = 8
    while len(settings.slices) > target:
        settings.slices.remove(len(settings.slices) - 1)
    n = len(settings.slices)
    if n < 2:
        return
    for i, slc in enumerate(settings.slices):
        slc.elevation = math.radians(90.0 - (180.0 * i / (n - 1)))
        if i == 0 or i == n - 1:
            slc.camera_count = 1
            slc.azimuth_offset = 0.0


def _on_slice_count_update(self, context):
    _redistribute_slices(self)


class SIB_Settings(PropertyGroup):
    # ---- Output ----
    output_dir: StringProperty(
        name="Sheet Location", default="//imposter_sprites/", subtype='DIR_PATH',
    )
    output_prefix: StringProperty(name="Prefix", default="sprite")
    sprite_size: IntProperty(
        name="Sheet Resolution",
        description="Pixel size of each sprite cell (square)",
        default=256, min=16, max=4096,
    )

    # ---- Slices ----
    slice_count: IntProperty(
        name="Number of Slices",
        description="Total latitude slices including the two poles (min 3)",
        default=5, min=3, max=33,
        update=_on_slice_count_update,
    )
    slices: CollectionProperty(type=SIB_Slice)
    active_slice_index: IntProperty(default=0)

    # ---- Camera ----
    distance: FloatProperty(
        name="Camera Distance", default=5.0, min=0.001, subtype='DISTANCE',
    )
    camera_type: EnumProperty(
        name="Camera Type",
        items=[
            ('ORTHO', "Orthographic", "Recommended for impostors"),
            ('PERSP', "Perspective", "Standard perspective"),
        ],
        default='ORTHO',
    )
    ortho_scale: FloatProperty(name="Ortho Scale", default=2.5, min=0.001)
    focal_length: FloatProperty(name="Focal Length (mm)", default=50.0, min=1.0)
    auto_fit: BoolProperty(name="Auto-fit Camera", default=True)
    auto_fit_padding: FloatProperty(name="Padding", default=0.1, min=0.0, max=2.0)

    # ---- Render & transparency ----
    render_engine: EnumProperty(
        name="Render Engine",
        description="Engine used for the bake. Try Cycles if EEVEE produces wrong output (it's slower but more reliable with material overrides, transparency, and unusual setups)",
        items=[
            ('SCENE', "Use Scene Engine", "Use whatever's already in render settings (switches off Workbench)"),
            ('EEVEE', "EEVEE", "Force EEVEE — fast"),
            ('CYCLES', "Cycles", "Force Cycles — slower but more reliable"),
        ],
        default='SCENE',
    )
    cycles_samples: IntProperty(
        name="Cycles Samples",
        description="Samples per sprite when rendering with Cycles. Low values are fine for impostor sprites",
        default=16, min=1, max=512,
    )
    hide_non_selected: BoolProperty(
        name="Hide Non-Selected Meshes", default=True,
    )
    lighting_mode: EnumProperty(
        name="Lighting",
        description="How to light the scene during bake. Use a fallback if your scene has no lights",
        items=[
            ('AUTO', "Auto", "Use scene lights. If none, fall back to ambient world emission"),
            ('SCENE', "Scene Only", "Use scene as-is (sprites will be dark if no lights)"),
            ('AMBIENT', "Ambient", "Always override world background with white emission (forces ambient illumination)"),
            ('SHADELESS', "Shadeless", "Replace materials with their base color (true fullbright, no lighting)"),
        ],
        default='AUTO',
    )
    transparency_mode: EnumProperty(
        name="Transparency",
        description="How transparent pixels are encoded in the atlas",
        items=[
            ('ALPHA', "Alpha", "Smooth RGBA alpha (default)"),
            ('DITHER', "Dither", "Binary alpha via 8×8 Bayer dither"),
        ],
        default='ALPHA',
    )
    delete_individual_after_pack: BoolProperty(
        name="Delete Per-Sprite PNGs", default=True,
    )

    # ---- Verification sprite ----
    create_preview: BoolProperty(
        name="Create Verification Sprite",
        description="After baking, create a plane at the selection center with the atlas applied",
        default=True,
    )
    preview_billboard: BoolProperty(
        name="Billboard to Camera",
        description="Add a Track-To constraint so the preview always faces the active camera",
        default=True,
    )
    preview_offset_distance: FloatProperty(
        name="Preview Offset",
        description="Distance from the bake center to place the preview plane. 0 = on top of the object",
        default=1.0, min=0.0,
        subtype='DISTANCE',
    )


# ============================================================================
# UI
# ============================================================================

class SIB_UL_Slices(UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        n = len(data.slices)
        is_pole = (index == 0 or index == n - 1)

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text='', icon='LOCKED' if is_pole else 'DOT')
            elev_deg = math.degrees(item.elevation)
            row.label(text=f"φ {elev_deg:+6.1f}°")
            if is_pole:
                row.label(text="× 1 (pole)")
            else:
                sub = row.row(align=True)
                sub.prop(item, 'camera_count', text='×')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text='')


class SIB_PT_Main(Panel):
    bl_label = "Sphere Imposter Baker"
    bl_idname = "SIB_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Imposter"

    def draw(self, context):
        layout = self.layout
        s = context.scene.sib_settings

        sel = [o for o in context.selected_objects
               if o.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'EMPTY'}]
        info = layout.row()
        if not sel:
            info.label(text="Select object(s) to bake", icon='ERROR')
        else:
            info.label(text=f"Selected: {len(sel)} object(s)", icon='RESTRICT_SELECT_OFF')

        # ---- Output ----
        box = layout.box()
        box.label(text="Output Sheet", icon='OUTPUT')
        box.prop(s, 'output_dir')
        box.prop(s, 'output_prefix')
        box.prop(s, 'sprite_size')
        box.prop(s, 'transparency_mode')

        # ---- Slices ----
        box = layout.box()
        box.label(text="Sphere Slices", icon='MESH_ICOSPHERE')
        box.prop(s, 'slice_count')

        if s.slice_count != len(s.slices):
            box.operator("sib.rebuild_slices", icon='FILE_REFRESH')
        else:
            box.template_list(
                "SIB_UL_Slices", "",
                s, "slices",
                s, "active_slice_index",
                rows=max(3, min(s.slice_count, 9)),
            )
            i = s.active_slice_index
            n = len(s.slices)
            if 0 < i < n - 1:
                sub = box.row()
                sub.prop(s.slices[i], 'azimuth_offset')
            elif n > 0 and (i == 0 or i == n - 1):
                box.label(text="(pole slices locked to 1 camera)", icon='INFO')

            prow = box.row(align=True)
            prow.operator("sib.preset_equator_weighted", text="Equator-Weighted")
            prow.operator("sib.preset_uniform", text="Uniform")

            total = sum(slc.camera_count for slc in s.slices)
            info = box.row()
            info.alignment = 'RIGHT'
            info.label(text=f"Total cameras: {total}", icon='CAMERA_STEREO')

        # ---- Camera ----
        box = layout.box()
        box.label(text="Camera", icon='CAMERA_DATA')
        box.prop(s, 'camera_type')
        box.prop(s, 'distance')
        box.prop(s, 'auto_fit')
        if s.auto_fit:
            box.prop(s, 'auto_fit_padding')
        else:
            if s.camera_type == 'ORTHO':
                box.prop(s, 'ortho_scale')
            else:
                box.prop(s, 'focal_length')

        # ---- Camera preview ----
        box = layout.box()
        box.label(text="Camera Preview", icon='HIDE_OFF')
        prow = box.row(align=True)
        prow.enabled = bool(sel)
        prow.operator("sib.show_cameras", text="Show Cameras", icon='CAMERA_DATA')
        prow.operator("sib.clear_cameras", text="Clear", icon='X')
        exists = (PREVIEW_CAM_COLLECTION in bpy.data.collections)
        if exists:
            box.label(text="Preview cameras are in the scene", icon='INFO')

        # ---- Render options ----
        box = layout.box()
        box.label(text="Render Options", icon='RENDER_STILL')
        box.prop(s, 'render_engine')
        if s.render_engine == 'CYCLES':
            box.prop(s, 'cycles_samples')
        box.prop(s, 'lighting_mode')
        box.prop(s, 'hide_non_selected')
        box.prop(s, 'delete_individual_after_pack')

        # ---- Verification ----
        box = layout.box()
        box.label(text="Verification Sprite", icon='MESH_PLANE')
        box.prop(s, 'create_preview')
        if s.create_preview:
            box.prop(s, 'preview_billboard')
            box.prop(s, 'preview_offset_distance')

        # ---- Bake ----
        layout.separator()
        big = layout.row()
        big.scale_y = 1.6
        big.enabled = bool(sel)
        big.operator("sib.bake", icon='RENDER_STILL', text="Bake from Selection")


# ============================================================================
# SLICE / PRESET OPERATORS
# ============================================================================

class SIB_OT_RebuildSlices(Operator):
    bl_idname = "sib.rebuild_slices"
    bl_label = "Rebuild Slices"

    def execute(self, context):
        _redistribute_slices(context.scene.sib_settings)
        return {'FINISHED'}


class SIB_OT_PresetEquatorWeighted(Operator):
    bl_idname = "sib.preset_equator_weighted"
    bl_label = "Equator-Weighted"

    def execute(self, context):
        s = context.scene.sib_settings
        s.slice_count = 5
        if len(s.slices) == 5:
            s.slices[1].camera_count = 3
            s.slices[2].camera_count = 8
            s.slices[3].camera_count = 3
            s.slices[3].azimuth_offset = math.radians(60)
        return {'FINISHED'}


class SIB_OT_PresetUniform(Operator):
    bl_idname = "sib.preset_uniform"
    bl_label = "Uniform"

    def execute(self, context):
        s = context.scene.sib_settings
        n = len(s.slices)
        for i in range(1, n - 1):
            s.slices[i].camera_count = 8
            s.slices[i].azimuth_offset = 0.0
        return {'FINISHED'}


# ============================================================================
# CAMERA PREVIEW OPERATORS
# ============================================================================

def _gather_selected_meshes(context):
    """Return the selected objects suitable for baking."""
    return [o for o in context.selected_objects
            if o.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'EMPTY'}]


def _compute_center(obj_set):
    coords = []
    for obj in obj_set:
        if obj.type == 'MESH' and obj.data.vertices:
            for v in obj.bound_box:
                coords.append(obj.matrix_world @ Vector(v))
    if not coords:
        if not obj_set:
            return Vector((0, 0, 0))
        return sum((o.matrix_world.translation for o in obj_set),
                   Vector()) / len(obj_set)
    bbox_min = Vector((min(c.x for c in coords), min(c.y for c in coords), min(c.z for c in coords)))
    bbox_max = Vector((max(c.x for c in coords), max(c.y for c in coords), max(c.z for c in coords)))
    return (bbox_min + bbox_max) * 0.5


def _add_descendants(obj, out):
    for child in obj.children:
        if child not in out:
            out.add(child)
            _add_descendants(child, out)


def _iter_camera_positions(settings, center):
    """Yield (slice_idx, cam_idx_in_slice, world_pos, world_dir_to_center) for every camera."""
    n_slices = len(settings.slices)
    for slice_idx, slc in enumerate(settings.slices):
        count = slc.camera_count
        elev = slc.elevation
        az_off = slc.azimuth_offset
        is_pole = (slice_idx == 0 or slice_idx == n_slices - 1)
        for i in range(count):
            az = 0.0 if is_pole else (i * 2.0 * math.pi / count) + az_off
            x = settings.distance * math.cos(elev) * math.cos(az)
            y = settings.distance * math.cos(elev) * math.sin(az)
            z = settings.distance * math.sin(elev)
            pos = center + Vector((x, y, z))
            direction = (center - pos).normalized()
            yield slice_idx, i, az, pos, direction


def _find_layer_collection(layer_coll, name):
    if layer_coll.collection.name == name:
        return layer_coll
    for child in layer_coll.children:
        r = _find_layer_collection(child, name)
        if r is not None:
            return r
    return None


def _ensure_preview_collection(scene):
    # Get existing or create new
    if PREVIEW_CAM_COLLECTION in bpy.data.collections:
        coll = bpy.data.collections[PREVIEW_CAM_COLLECTION]
    else:
        coll = bpy.data.collections.new(PREVIEW_CAM_COLLECTION)

    # Ensure it's linked under the scene's master collection
    if PREVIEW_CAM_COLLECTION not in scene.collection.children:
        try:
            scene.collection.children.link(coll)
        except RuntimeError:
            pass  # already linked somewhere

    # Ensure it's visible in every view layer (not excluded, not hidden)
    for view_layer in scene.view_layers:
        lc = _find_layer_collection(view_layer.layer_collection, PREVIEW_CAM_COLLECTION)
        if lc is not None:
            lc.exclude = False
            lc.hide_viewport = False

    coll.hide_viewport = False
    coll.hide_render = False
    return coll


def _clear_preview_collection():
    if PREVIEW_CAM_COLLECTION not in bpy.data.collections:
        return 0
    coll = bpy.data.collections[PREVIEW_CAM_COLLECTION]
    n = len(coll.objects)

    for obj in list(coll.objects):
        cam_data = obj.data if isinstance(obj.data, bpy.types.Camera) else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if cam_data is not None and cam_data.users == 0:
            bpy.data.cameras.remove(cam_data)

    # Unlink from all scenes before removing
    for scene in bpy.data.scenes:
        if PREVIEW_CAM_COLLECTION in scene.collection.children:
            try:
                scene.collection.children.unlink(coll)
            except RuntimeError:
                pass

    try:
        bpy.data.collections.remove(coll, do_unlink=True)
    except (TypeError, RuntimeError):
        try:
            bpy.data.collections.remove(coll)
        except RuntimeError:
            pass
    return n


class SIB_OT_ShowCameras(Operator):
    bl_idname = "sib.show_cameras"
    bl_label = "Show Cameras"
    bl_description = "Create camera objects at every position the bake will use"

    def execute(self, context):
        s = context.scene.sib_settings
        selected = _gather_selected_meshes(context)
        if not selected:
            self.report({'ERROR'}, "Select at least one object first")
            return {'CANCELLED'}

        if s.slice_count != len(s.slices) or len(s.slices) < 3:
            _redistribute_slices(s)

        # Compute center from selection
        visible_set = set()
        for o in selected:
            visible_set.add(o)
            _add_descendants(o, visible_set)
        center = _compute_center(visible_set)

        # Clear any prior preview
        _clear_preview_collection()
        coll = _ensure_preview_collection(context.scene)

        # Compute auto-fit distance and ortho scale for the preview cameras,
        # matching what bake will use
        preview_distance = s.distance
        ortho_scale = s.ortho_scale
        if s.auto_fit:
            coords = []
            for obj in visible_set:
                if obj.type == 'MESH':
                    for v in obj.bound_box:
                        coords.append(obj.matrix_world @ Vector(v))
            if coords:
                radius = max((c - center).length for c in coords)
                padded_radius = radius * (1.0 + s.auto_fit_padding)
                if s.camera_type == 'ORTHO':
                    ortho_scale = padded_radius * 2.0
                    preview_distance = max(preview_distance, padded_radius * 1.5)
                else:
                    # Perspective: assume default sensor (36mm) and user's focal length
                    sensor = 36.0
                    half_fov = math.atan((sensor / 2.0) / s.focal_length)
                    needed = padded_radius / math.tan(half_fov)
                    preview_distance = max(needed, padded_radius * 1.2)

        # Persist computed distance so the bake uses the same positions
        s.distance = preview_distance

        count = 0
        for slice_idx, cam_idx, az, pos, direction in _iter_camera_positions(s, center):
            cam_data = bpy.data.cameras.new(f"SIB_PrevCam_{count:03d}")
            if s.camera_type == 'ORTHO':
                cam_data.type = 'ORTHO'
                cam_data.ortho_scale = ortho_scale
            else:
                cam_data.type = 'PERSP'
                cam_data.lens = s.focal_length
            cam_data.display_size = max(0.1, preview_distance * 0.05)

            cam_obj = bpy.data.objects.new(f"SIB_PrevCam_{count:03d}", cam_data)
            cam_obj.location = pos
            cam_obj.rotation_euler = (-direction).to_track_quat('Z', 'Y').to_euler()
            coll.objects.link(cam_obj)
            count += 1

        self.report({'INFO'}, f"Created {count} preview cameras at distance {preview_distance:.2f}m")
        return {'FINISHED'}


class SIB_OT_ClearCameras(Operator):
    bl_idname = "sib.clear_cameras"
    bl_label = "Clear Camera Preview"
    bl_description = "Remove all preview camera markers"

    def execute(self, context):
        n = _clear_preview_collection()
        if n:
            self.report({'INFO'}, f"Removed {n} preview cameras")
        return {'FINISHED'}


# ============================================================================
# BAKE OPERATOR
# ============================================================================

_BAYER_8 = np.array([
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) / 64.0


def _scene_has_lights(scene):
    """True if any non-hidden light exists in the scene."""
    return any(o.type == 'LIGHT' and not o.hide_render for o in scene.objects)


def _apply_ambient_fallback(scene):
    """Set world background to white emission so unlit scenes still render."""
    backup = {'mode': 'AMBIENT'}

    if scene.world is None:
        scene.world = bpy.data.worlds.new("SIB_TempWorld")
        backup['created_world'] = True
        backup['world_name'] = scene.world.name
    else:
        backup['created_world'] = False
        backup['world_name'] = scene.world.name

    world = scene.world
    backup['use_nodes_old'] = world.use_nodes
    world.use_nodes = True

    bg = world.node_tree.nodes.get('Background')
    if bg is not None:
        backup['had_bg'] = True
        backup['bg_color'] = tuple(bg.inputs['Color'].default_value)
        backup['bg_strength'] = bg.inputs['Strength'].default_value
        bg.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs['Strength'].default_value = 1.0
    else:
        backup['had_bg'] = False

    return backup


def _apply_shadeless(target_objects):
    """Replace each material's surface output with an Emission of its base color.
    Handles objects with no material, materials with no nodes, and non-Principled shaders."""
    backup = {'mode': 'SHADELESS', 'materials': [], 'objects_temp_mat': []}

    # Step 1: Give a temp material to objects that have none
    for obj in target_objects:
        if obj.type != 'MESH':
            continue
        has_real_mat = any(m is not None for m in obj.data.materials) if obj.data.materials else False
        if not has_real_mat:
            mat = bpy.data.materials.new("SIB_TempMat")
            mat.use_nodes = True
            mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            obj.data.materials.append(mat)
            backup['objects_temp_mat'].append((obj, mat))

    # Step 2: Collect every material referenced by the selection
    materials = set()
    for obj in target_objects:
        if hasattr(obj.data, 'materials'):
            for mat in obj.data.materials:
                if mat is not None:
                    materials.add(mat)

    for mat in materials:
        had_nodes = mat.use_nodes
        if not had_nodes:
            mat.use_nodes = True

        nt = mat.node_tree
        output = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if output is None:
            output = nt.nodes.new('ShaderNodeOutputMaterial')

        # Pick a color source — Principled BSDF base color first, then any image texture
        color_link = None
        color_value = (0.8, 0.8, 0.8, 1.0)
        for n in nt.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                bc = n.inputs.get('Base Color')
                if bc is not None:
                    if bc.is_linked:
                        color_link = bc.links[0].from_socket
                    else:
                        color_value = tuple(bc.default_value)
                break
        if color_link is None:
            # Fallback: any image texture in the tree
            for n in nt.nodes:
                if n.type == 'TEX_IMAGE' and n.image is not None:
                    color_link = n.outputs['Color']
                    break
        if color_link is None:
            # Last resort: material diffuse color
            color_value = tuple(mat.diffuse_color)

        # Snapshot original surface link, then disconnect
        surface_input = output.inputs['Surface']
        original_socket = None
        if surface_input.is_linked:
            original_socket = surface_input.links[0].from_socket
            nt.links.remove(surface_input.links[0])

        emit = nt.nodes.new('ShaderNodeEmission')
        emit.label = "SIB_TempShadeless"
        emit.inputs['Strength'].default_value = 1.0
        if color_link is not None:
            nt.links.new(color_link, emit.inputs['Color'])
        else:
            emit.inputs['Color'].default_value = color_value

        nt.links.new(emit.outputs[0], surface_input)

        backup['materials'].append({
            'material': mat,
            'output': output,
            'original_socket': original_socket,
            'emit': emit,
            'had_nodes': had_nodes,
        })

    return backup


def _restore_lighting(scene, backup):
    if backup is None:
        return

    mode = backup.get('mode')

    if mode == 'AMBIENT':
        if backup.get('created_world'):
            world = scene.world
            if world is not None and world.name == backup['world_name']:
                scene.world = None
                bpy.data.worlds.remove(world)
            return
        world = scene.world
        if world is not None:
            world.use_nodes = backup['use_nodes_old']
            if backup.get('had_bg'):
                bg = world.node_tree.nodes.get('Background')
                if bg is not None:
                    bg.inputs['Color'].default_value = backup['bg_color']
                    bg.inputs['Strength'].default_value = backup['bg_strength']

    elif mode == 'SHADELESS':
        for m in backup['materials']:
            nt = m['material'].node_tree
            for link in list(m['output'].inputs['Surface'].links):
                nt.links.remove(link)
            if m['original_socket'] is not None:
                nt.links.new(m['original_socket'], m['output'].inputs['Surface'])
            nt.nodes.remove(m['emit'])
            if not m['had_nodes']:
                m['material'].use_nodes = False
        # Remove temp materials we created for objects that had none
        for obj, mat in backup.get('objects_temp_mat', []):
            for i in range(len(obj.data.materials) - 1, -1, -1):
                if obj.data.materials[i] == mat:
                    obj.data.materials.pop(index=i)
                    break
            if mat.users == 0:
                bpy.data.materials.remove(mat)


def _dither_alpha(rgba):
    h, w = rgba.shape[:2]
    yi, xi = np.indices((h, w))
    threshold = _BAYER_8[yi % 8, xi % 8]
    alpha = rgba[..., 3]
    binary = (alpha > threshold).astype(np.float32)
    out = rgba.copy()
    out[..., 3] = binary
    return out


def _generate_lut(sprite_meta, cols, rows, lut_w=128, lut_h=64):
    """Build a lookup texture: each LUT pixel (u_az, v_el) stores the atlas
    bottom-left UV (R=u_offset, G=v_offset) of the sprite whose camera direction
    is closest to (azimuth, elevation) at that pixel.

    The shader samples the LUT with the world-space view direction, then uses
    the returned offset + local UV to pick the correct atlas cell.
    """
    n = len(sprite_meta)
    sprite_dirs = np.zeros((n, 3), dtype=np.float32)
    sprite_cells = np.zeros((n, 2), dtype=np.float32)
    for i, m in enumerate(sprite_meta):
        el = math.radians(m['elevation_deg'])
        az = math.radians(m['azimuth_deg'])
        sprite_dirs[i] = (
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        )
        # Bottom-left UV of cell. Blender V is flipped vs PNG row index:
        # PNG row 0 is top of image → V close to 1.
        sprite_cells[i] = (
            m['atlas_col'] / cols,
            1.0 - (m['atlas_row'] + 1) / rows,
        )

    # Query grid covering full sphere of view directions
    az_n = (np.arange(lut_w, dtype=np.float32) + 0.5) / lut_w   # cell centers
    el_n = (np.arange(lut_h, dtype=np.float32) + 0.5) / lut_h
    az = az_n * (2 * np.pi) - np.pi
    el = el_n * np.pi - (np.pi / 2)

    AZ, EL = np.meshgrid(az, el)
    qx = np.cos(EL) * np.cos(AZ)
    qy = np.cos(EL) * np.sin(AZ)
    qz = np.sin(EL)
    queries = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)

    # Pick nearest sprite by dot-product (cosine similarity on unit sphere)
    dots = queries @ sprite_dirs.T
    best_idx = np.argmax(dots, axis=1)
    best_cells = sprite_cells[best_idx]

    lut = np.zeros((lut_h, lut_w, 4), dtype=np.float32)
    lut[..., 0] = best_cells[:, 0].reshape(lut_h, lut_w)
    lut[..., 1] = best_cells[:, 1].reshape(lut_h, lut_w)
    lut[..., 3] = 1.0
    return lut


def _save_lut_png(lut, filepath, name):
    """Save the LUT as a Blender image (Non-Color) then export as PNG.

    Important: the numpy array convention here matches Blender's pixel storage
    (row 0 = bottom = V=0). So writing the numpy array directly without flipping
    gives correct shader sampling:
        numpy row 0    (lowest elevation, south pole)  → V=0
        numpy row max  (highest elevation, north pole) → V=1
    Shader samples LUT at V=el_n where el_n=1 for "looking down from above"
    (high elevation) → correctly hits the north pole data at V=1.
    """
    h, w = lut.shape[:2]
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(lut.ravel())
    img.filepath_raw = filepath
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


def _embed_png_metadata(filepath, sprite_meta, sprite_size, cols, rows, transparency_mode):
    """Insert an iTXt chunk containing JSON sprite metadata into the PNG.
    Runtime can parse this to map (elevation, azimuth) → atlas cell."""
    import struct
    import zlib

    # Build compact metadata payload
    payload = {
        'version': 1,
        'sprite_size': sprite_size,
        'cols': cols,
        'rows': rows,
        'count': len(sprite_meta),
        'transparency': transparency_mode.lower(),
        'sprites': [
            {
                'i': m['index'],
                'el': m['elevation_deg'],
                'az': m['azimuth_deg'],
                'col': m.get('atlas_col', 0),
                'row': m.get('atlas_row', 0),
                'x': m.get('atlas_x', 0),
                'y': m.get('atlas_y', 0),
            }
            for m in sprite_meta
        ],
    }
    text = json.dumps(payload, separators=(',', ':'))

    with open(filepath, 'rb') as f:
        data = f.read()

    # Locate IEND chunk (always last chunk, fixed at end of file)
    iend_type_pos = data.rfind(b'IEND')
    if iend_type_pos < 0:
        return  # not a PNG, give up silently
    # The chunk starts 4 bytes before the type field (the length field)
    iend_chunk_start = iend_type_pos - 4

    # Build iTXt chunk
    # Format: keyword \0 compression_flag(1) compression_method(1)
    #         language_tag \0 translated_keyword \0 text
    keyword = b'SIB_AtlasMeta'
    chunk_data = (
        keyword + b'\x00'
        + b'\x00'   # uncompressed
        + b'\x00'   # compression method (0 = deflate, unused since flag=0)
        + b'\x00'   # empty language tag + null
        + b'\x00'   # empty translated keyword + null
        + text.encode('utf-8')
    )
    chunk_type = b'iTXt'
    length = struct.pack('>I', len(chunk_data))
    crc = struct.pack('>I', zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
    new_chunk = length + chunk_type + chunk_data + crc

    new_data = data[:iend_chunk_start] + new_chunk + data[iend_chunk_start:]
    with open(filepath, 'wb') as f:
        f.write(new_data)


class SIB_OT_Bake(Operator):
    bl_idname = "sib.bake"
    bl_label = "Bake Imposter Sprites"
    bl_description = "Render selected objects from all slice cameras and pack into a sheet"

    def execute(self, context):
        s = context.scene.sib_settings
        scene = context.scene

        selected = _gather_selected_meshes(context)
        if not selected:
            self.report({'ERROR'}, "Select at least one object before baking")
            return {'CANCELLED'}

        if len(s.slices) != s.slice_count or len(s.slices) < 3:
            _redistribute_slices(s)

        # Auto-clear preview cameras so they don't pollute the render
        _clear_preview_collection()

        st = self._snapshot_state(scene)
        view_layer = context.view_layer
        vl_backup = self._snapshot_view_layer(scene, view_layer)

        # ---- Render engine selection ----
        cycles_backup = None
        if s.render_engine == 'EEVEE':
            for engine_id in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
                try:
                    scene.render.engine = engine_id
                    break
                except TypeError:
                    continue
        elif s.render_engine == 'CYCLES':
            try:
                scene.render.engine = 'CYCLES'
                if hasattr(scene, 'cycles'):
                    cycles_backup = {
                        'samples': scene.cycles.samples,
                        'preview_samples': scene.cycles.preview_samples,
                    }
                    if hasattr(scene.cycles, 'use_denoising'):
                        cycles_backup['use_denoising'] = scene.cycles.use_denoising
                        scene.cycles.use_denoising = False
                    scene.cycles.samples = s.cycles_samples
                    scene.cycles.preview_samples = s.cycles_samples
            except TypeError:
                self.report({'WARNING'}, "Cycles not available, falling back to EEVEE")
                for engine_id in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
                    try:
                        scene.render.engine = engine_id
                        break
                    except TypeError:
                        continue
        else:  # SCENE
            # Just switch off Workbench since it ignores film_transparent
            if scene.render.engine == 'BLENDER_WORKBENCH':
                for engine_id in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
                    try:
                        scene.render.engine = engine_id
                        break
                    except TypeError:
                        continue

        # Compositor can rewrite alpha and color — bypass it for the bake
        scene.render.use_compositing = False
        scene.render.use_sequencer = False

        # Render only the active view layer; clear any material override that
        # would shove every material in this layer to a fixed (likely black) one
        scene.render.use_single_layer = True
        view_layer.use = True
        if hasattr(view_layer, 'use_pass_combined'):
            view_layer.use_pass_combined = True
        if hasattr(view_layer, 'material_override'):
            view_layer.material_override = None
        if hasattr(view_layer, 'samples'):
            try:
                view_layer.samples = 0  # 0 = use scene samples
            except (AttributeError, TypeError):
                pass

        # Color management: anything other than Standard with neutral
        # exposure can produce surprising output (filmic crushes blacks,
        # negative exposure makes everything black, etc.)
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0

        scene.render.film_transparent = True
        scene.render.resolution_x = s.sprite_size
        scene.render.resolution_y = s.sprite_size
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'

        visible_set = set()
        for o in selected:
            visible_set.add(o)
            _add_descendants(o, visible_set)

        hidden = []
        if s.hide_non_selected:
            for obj in scene.objects:
                if obj.type == 'MESH' and obj not in visible_set:
                    if not obj.hide_render:
                        obj.hide_render = True
                        hidden.append(obj)

        center = _compute_center(visible_set)

        cam_data = bpy.data.cameras.new(name="SIB_BakeCam")
        if s.camera_type == 'ORTHO':
            cam_data.type = 'ORTHO'
            cam_data.ortho_scale = s.ortho_scale
        else:
            cam_data.type = 'PERSP'
            cam_data.lens = s.focal_length
        cam_obj = bpy.data.objects.new("SIB_BakeCam", cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj

        if s.auto_fit:
            new_distance = self._auto_fit_camera(visible_set, center, cam_data, s)
            if new_distance is not None:
                s.distance = new_distance
            # Sync ortho_scale back to settings so preview uses the same size
            if s.camera_type == 'ORTHO':
                s.ortho_scale = cam_data.ortho_scale

        # Apply lighting fallback before rendering
        lighting_backup = None
        if s.lighting_mode == 'AMBIENT':
            lighting_backup = _apply_ambient_fallback(scene)
        elif s.lighting_mode == 'SHADELESS':
            lighting_backup = _apply_shadeless(visible_set)
        elif s.lighting_mode == 'AUTO':
            if not _scene_has_lights(scene):
                # Shadeless is far more reliable than world emission alone,
                # especially in EEVEE Next which needs irradiance probes for ambient.
                lighting_backup = _apply_shadeless(visible_set)

        output_dir = bpy.path.abspath(s.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        sprite_paths = []
        sprite_meta = []
        idx = 0
        n_slices = len(s.slices)

        try:
            for slice_idx, cam_idx, az, pos, direction in _iter_camera_positions(s, center):
                cam_obj.location = pos
                cam_obj.rotation_euler = (-direction).to_track_quat('Z', 'Y').to_euler()

                filename = f"{s.output_prefix}_{idx:04d}_S{slice_idx:02d}_a{int(math.degrees(az)) % 360:03d}.png"
                filepath = os.path.join(output_dir, filename)
                scene.render.filepath = filepath
                bpy.ops.render.render(write_still=True)

                sprite_paths.append(filepath)
                sprite_meta.append({
                    'index': idx,
                    'slice': slice_idx,
                    'elevation_deg': round(math.degrees(s.slices[slice_idx].elevation), 3),
                    'azimuth_deg': round(math.degrees(az) % 360, 3),
                    'filename': filename,
                })
                idx += 1

                # After the first render, check if it's blank. Catches view-layer
                # material overrides, missing combined pass, etc., before wasting
                # all 16 renders.
                if idx == 1:
                    diag = self._diagnose_first_render(filepath, s.sprite_size)
                    if diag:
                        self.report({'WARNING'}, diag)

            atlas_path, atlas_cols, atlas_rows = self._pack_atlas(
                sprite_paths, sprite_meta, s, output_dir)

            # Generate impostor lookup texture
            lut_data = _generate_lut(sprite_meta, atlas_cols, atlas_rows)
            lut_path = os.path.join(output_dir, f"{s.output_prefix}_lut.png")
            _save_lut_png(lut_data, lut_path, f"{s.output_prefix}_lut")

            manifest = {
                'version': 5,
                'sprite_size': s.sprite_size,
                'total_sprites': len(sprite_meta),
                'transparency_mode': s.transparency_mode,
                'atlas': {
                    'filename': os.path.basename(atlas_path),
                    'columns': atlas_cols,
                    'rows': atlas_rows,
                    'width': atlas_cols * s.sprite_size,
                    'height': atlas_rows * s.sprite_size,
                    'layout': 'DENSE_GRID',
                    'metadata_chunk_keyword': 'SIB_AtlasMeta',
                },
                'camera': {
                    'type': s.camera_type,
                    'distance': s.distance,
                    'ortho_scale': s.ortho_scale if s.camera_type == 'ORTHO' else None,
                    'focal_length': s.focal_length if s.camera_type == 'PERSP' else None,
                },
                'center': [center.x, center.y, center.z],
                'slices': [
                    {
                        'index': i,
                        'elevation_deg': round(math.degrees(slc.elevation), 3),
                        'azimuth_offset_deg': round(math.degrees(slc.azimuth_offset), 3),
                        'camera_count': slc.camera_count,
                        'is_pole': (i == 0 or i == n_slices - 1),
                    }
                    for i, slc in enumerate(s.slices)
                ],
                'sprites': sprite_meta,
            }
            manifest_path = os.path.join(output_dir, f"{s.output_prefix}_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            if s.delete_individual_after_pack:
                for p in sprite_paths:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

            preview_msg = ""
            if s.create_preview:
                preview = self._create_preview_plane(
                    atlas_path, lut_path, center, atlas_cols, atlas_rows, s, context)
                preview_msg = (f" + preview '{preview.name}' "
                               "(switch viewport to Material Preview to see it)")

            self.report({'INFO'},
                f"Baked {idx} sprites → {os.path.basename(atlas_path)} "
                f"({s.transparency_mode}){preview_msg}")

        finally:
            bpy.data.objects.remove(cam_obj, do_unlink=True)
            bpy.data.cameras.remove(cam_data, do_unlink=True)
            for obj in hidden:
                obj.hide_render = False
            _restore_lighting(scene, lighting_backup)
            self._restore_view_layer(scene, view_layer, vl_backup)
            if cycles_backup and hasattr(scene, 'cycles'):
                scene.cycles.samples = cycles_backup['samples']
                scene.cycles.preview_samples = cycles_backup['preview_samples']
                if 'use_denoising' in cycles_backup:
                    scene.cycles.use_denoising = cycles_backup['use_denoising']
            self._restore_state(scene, st)

        return {'FINISHED'}

    # ---- Helpers ----

    def _diagnose_first_render(self, filepath, sprite_size):
        """Load the freshly-rendered PNG and check it has actual content.
        Returns a warning message if it looks blank/wrong, else None."""
        try:
            img = bpy.data.images.load(filepath, check_existing=False)
        except RuntimeError:
            return None
        try:
            w, h = img.size
            buf = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(buf)
            arr = buf.reshape(-1, 4)
            rgb_max = float(arr[:, :3].max())
            alpha_min = float(arr[:, 3].min())
            alpha_max = float(arr[:, 3].max())
        finally:
            bpy.data.images.remove(img)

        # All black RGB + all opaque alpha = render is wrong
        if rgb_max < 1e-4 and alpha_min > 0.99:
            return ("First sprite is solid black & fully opaque. Check the active "
                    "view layer for a Material Override (Properties > View Layer), "
                    "and try Lighting=Shadeless explicitly.")
        # All transparent — geometry didn't render at all
        if alpha_max < 1e-4:
            return ("First sprite is fully transparent — selected objects may "
                    "not be in the active view layer's visible collections.")
        return None

    def _snapshot_state(self, scene):
        return {
            'camera': scene.camera,
            'film_transparent': scene.render.film_transparent,
            'res_x': scene.render.resolution_x,
            'res_y': scene.render.resolution_y,
            'res_pct': scene.render.resolution_percentage,
            'filepath': scene.render.filepath,
            'file_format': scene.render.image_settings.file_format,
            'color_mode': scene.render.image_settings.color_mode,
            'color_depth': scene.render.image_settings.color_depth,
            'engine': scene.render.engine,
            'use_compositing': scene.render.use_compositing,
            'use_sequencer': scene.render.use_sequencer,
            'use_single_layer': scene.render.use_single_layer,
            'view_transform': scene.view_settings.view_transform,
            'look': scene.view_settings.look,
            'exposure': scene.view_settings.exposure,
            'gamma': scene.view_settings.gamma,
        }

    def _snapshot_view_layer(self, scene, view_layer):
        backup = {'name': view_layer.name}
        backup['use'] = view_layer.use
        if hasattr(view_layer, 'use_pass_combined'):
            backup['use_pass_combined'] = view_layer.use_pass_combined
        if hasattr(view_layer, 'material_override'):
            backup['material_override'] = view_layer.material_override
        return backup

    def _restore_view_layer(self, scene, view_layer, backup):
        if backup is None:
            return
        view_layer.use = backup['use']
        if 'use_pass_combined' in backup and hasattr(view_layer, 'use_pass_combined'):
            view_layer.use_pass_combined = backup['use_pass_combined']
        if 'material_override' in backup and hasattr(view_layer, 'material_override'):
            view_layer.material_override = backup['material_override']

    def _restore_state(self, scene, st):
        scene.camera = st['camera']
        scene.render.film_transparent = st['film_transparent']
        scene.render.resolution_x = st['res_x']
        scene.render.resolution_y = st['res_y']
        scene.render.resolution_percentage = st['res_pct']
        scene.render.filepath = st['filepath']
        scene.render.image_settings.file_format = st['file_format']
        scene.render.image_settings.color_mode = st['color_mode']
        scene.render.image_settings.color_depth = st['color_depth']
        try:
            scene.render.engine = st['engine']
        except TypeError:
            pass
        scene.render.use_compositing = st['use_compositing']
        scene.render.use_sequencer = st['use_sequencer']
        scene.render.use_single_layer = st['use_single_layer']
        scene.view_settings.view_transform = st['view_transform']
        scene.view_settings.look = st['look']
        scene.view_settings.exposure = st['exposure']
        scene.view_settings.gamma = st['gamma']

    def _auto_fit_camera(self, obj_set, center, cam_data, settings):
        """Return a new distance (and adjust ortho_scale on cam_data) so that
        the object fits entirely in frame from any sphere position. Returns
        None on failure (e.g., empty selection)."""
        coords = []
        for obj in obj_set:
            if obj.type == 'MESH':
                for v in obj.bound_box:
                    coords.append(obj.matrix_world @ Vector(v))
        if not coords:
            return None

        radius = max((c - center).length for c in coords)
        padded_radius = radius * (1.0 + settings.auto_fit_padding)

        if settings.camera_type == 'ORTHO':
            cam_data.ortho_scale = padded_radius * 2.0
            # Distance must be > radius so the camera is outside the object;
            # ortho doesn't need to fit by distance, just be in front of geometry.
            new_distance = max(settings.distance, padded_radius * 1.5)
        else:
            # Perspective: distance must be large enough for the bounding sphere
            # to fit inside the FOV. Use the smaller of sensor width / height
            # to fit even on non-square renders.
            sensor_w = cam_data.sensor_width
            sensor_h = cam_data.sensor_height
            sensor = min(sensor_w, sensor_h) if sensor_h > 0 else sensor_w
            focal = cam_data.lens
            half_fov = math.atan((sensor / 2.0) / focal)
            # tan(half_fov) = padded_radius / distance  →  distance = r / tan(half_fov)
            needed_distance = padded_radius / math.tan(half_fov)
            # Also guarantee we're outside the bounding sphere
            new_distance = max(needed_distance, padded_radius * 1.2)

        # Make sure camera clip planes can see the object
        cam_data.clip_start = max(0.001, (new_distance - padded_radius) * 0.5)
        cam_data.clip_end = max(cam_data.clip_end, (new_distance + padded_radius) * 2.0)

        return new_distance

    def _pack_atlas(self, sprite_paths, sprite_meta, settings, output_dir):
        """Dense square-ish grid packing: cols = ceil(sqrt(N)), rows = ceil(N/cols).
        16 sprites → 4×4. 17 sprites → 5×4. 9 → 3×3. No wasted rows."""
        s = settings
        sprite_size = s.sprite_size
        n = len(sprite_paths)
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, math.ceil(n / cols))
        atlas_w = cols * sprite_size
        atlas_h = rows * sprite_size

        atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.float32)

        for i, path in enumerate(sprite_paths):
            col_i = i % cols
            row_i = i // cols

            sprite_img = bpy.data.images.load(path, check_existing=False)
            try:
                sw = sprite_img.size[0]
                sh = sprite_img.size[1]
                buf = np.empty(sw * sh * 4, dtype=np.float32)
                sprite_img.pixels.foreach_get(buf)
                sprite_np = buf.reshape(sh, sw, 4)[::-1]

                dst_y = row_i * sprite_size
                dst_x = col_i * sprite_size
                atlas[dst_y:dst_y + sh, dst_x:dst_x + sw] = sprite_np

                sprite_meta[i]['atlas_col'] = col_i
                sprite_meta[i]['atlas_row'] = row_i
                sprite_meta[i]['atlas_x'] = dst_x
                sprite_meta[i]['atlas_y'] = dst_y
            finally:
                bpy.data.images.remove(sprite_img)

        if s.transparency_mode == 'DITHER':
            atlas = _dither_alpha(atlas)

        atlas_flipped = atlas[::-1]
        atlas_name = f"{s.output_prefix}_atlas"
        if atlas_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[atlas_name])
        atlas_img = bpy.data.images.new(atlas_name, atlas_w, atlas_h, alpha=True)
        atlas_img.pixels.foreach_set(atlas_flipped.ravel())

        atlas_path = os.path.join(output_dir, f"{s.output_prefix}_atlas.png")
        atlas_img.filepath_raw = atlas_path
        atlas_img.file_format = 'PNG'
        atlas_img.save()
        bpy.data.images.remove(atlas_img)

        # Embed sprite metadata directly in the PNG as an iTXt chunk
        _embed_png_metadata(atlas_path, sprite_meta, sprite_size, cols, rows,
                            s.transparency_mode)

        return atlas_path, cols, rows

    def _create_preview_plane(self, atlas_path, lut_path, center, atlas_cols,
                              atlas_rows, settings, context):
        prev_name = f"{settings.output_prefix}_Preview"

        # Clean up old preview
        if prev_name in bpy.data.objects:
            old = bpy.data.objects[prev_name]
            old_mesh = old.data
            bpy.data.objects.remove(old, do_unlink=True)
            if old_mesh and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)

        # Plane is square — sized to match ortho frame the bake used.
        # (The impostor shader handles aspect; we don't need atlas aspect on the plane.)
        size = settings.ortho_scale

        mesh = bpy.data.meshes.new(prev_name)
        h = size / 2.0
        # Plane in local XZ, normal points -Y
        verts = [(-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h)]
        faces = [(0, 1, 2, 3)]
        mesh.from_pydata(verts, [], faces)
        mesh.uv_layers.new(name="UVMap")
        uv_data = mesh.uv_layers[0].data
        for i, uv in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
            uv_data[i].uv = uv
        mesh.update()

        obj = bpy.data.objects.new(prev_name, mesh)
        context.scene.collection.objects.link(obj)

        offset = Vector((settings.preview_offset_distance, 0, 0))
        obj.location = center + offset
        obj.show_in_front = True

        # Register for viewport-tracking billboard. Polled by a timer because
        # the viewport's view isn't a scene-camera object — it lives in
        # RegionView3D and can't be referenced by a Track-To constraint.
        if settings.preview_billboard:
            _register_billboard(obj.name)
        else:
            # Clean up any leftover billboard registration for this name
            _SIB_BILLBOARD_OBJECTS.discard(obj.name)

        # ---- Material ----
        mat_name = f"{settings.output_prefix}_PreviewMat"
        if mat_name in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials[mat_name])
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True

        if settings.transparency_mode == 'DITHER':
            for attr, val in (('surface_render_method', 'DITHERED'),
                              ('blend_method', 'CLIP'),
                              ('alpha_threshold', 0.5)):
                try:
                    setattr(mat, attr, val)
                except (AttributeError, TypeError):
                    pass
        else:
            for attr, val in (('surface_render_method', 'BLENDED'),
                              ('blend_method', 'BLEND')):
                try:
                    setattr(mat, attr, val)
                except (AttributeError, TypeError):
                    pass

        mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)

        # Load images
        atlas_img = self._reload_or_load(atlas_path)
        lut_img = self._reload_or_load(lut_path)
        lut_img.colorspace_settings.name = 'Non-Color'

        self._build_impostor_shader(mat, atlas_img, lut_img,
                                     atlas_cols, atlas_rows)

        obj.data.materials.append(mat)

        # Select preview so user sees it immediately
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        return obj

    def _reload_or_load(self, path):
        """Load an image fresh from disk. If a cached image with the same name
        exists but points to a different file (e.g., output dir changed), fix
        its filepath before reloading."""
        name = os.path.basename(path)
        abs_target = bpy.path.abspath(path)

        if name in bpy.data.images:
            existing = bpy.data.images[name]
            existing_abs = bpy.path.abspath(existing.filepath) if existing.filepath else ""

            if existing.users == 0:
                # Orphan — safe to remove and load fresh
                bpy.data.images.remove(existing)
            else:
                # In use elsewhere — fix path if needed, then reload
                if existing_abs != abs_target:
                    existing.filepath = path
                    existing.filepath_raw = path
                try:
                    existing.reload()
                except RuntimeError:
                    pass
                return existing

        return bpy.data.images.load(path, check_existing=False)

    def _build_impostor_shader(self, mat, atlas_img, lut_img, cols, rows):
        """View-direction-aware shader, computing view direction at the
        object's CENTER so every fragment of the plane samples the same
        LUT cell. With Closest interpolation everywhere, transitions
        between cells are instant — no bleed, no tearing.

        How:
            1. Constant (0,0,0) in OBJECT space = the object's origin
            2. Transform it OBJECT → CAMERA as a Point (translation applies)
               → gives the object's origin in camera-space coordinates
            3. Negate → vector from object to camera in camera space
               (camera is at origin in camera space)
            4. Transform CAMERA → WORLD as a Vector (no translation)
               → world-space direction from object to viewer
            5. Normalize → use for atan2/asin

        Because the input (0,0,0) is constant per object, the resulting view
        direction is identical for every fragment of the plane. Every fragment
        computes the same azimuth/elevation, samples the same LUT cell,
        and shows the same sprite cell.
        """
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # ---- Atlas Image Texture (first so Blender solid mode picks it) ----
        atlas_tex = nodes.new('ShaderNodeTexImage')
        atlas_tex.image = atlas_img
        atlas_tex.interpolation = 'Closest'
        atlas_tex.extension = 'CLIP'
        atlas_tex.location = (-100, 0)

        # ---- LUT Image Texture ----
        lut_tex = nodes.new('ShaderNodeTexImage')
        lut_tex.image = lut_img
        lut_tex.interpolation = 'Closest'
        lut_tex.extension = 'REPEAT'
        lut_tex.location = (-500, 200)

        # ---- View direction at object CENTER ----
        # Constant (0,0,0) representing the object's origin in object space
        zero = nodes.new('ShaderNodeCombineXYZ')
        zero.inputs[0].default_value = 0.0
        zero.inputs[1].default_value = 0.0
        zero.inputs[2].default_value = 0.0
        zero.location = (-2400, 100)

        # OBJECT → CAMERA as a Point → object's origin in camera space
        obj_in_cam = nodes.new('ShaderNodeVectorTransform')
        obj_in_cam.vector_type = 'POINT'
        obj_in_cam.convert_from = 'OBJECT'
        obj_in_cam.convert_to = 'CAMERA'
        obj_in_cam.location = (-2200, 100)
        links.new(zero.outputs[0], obj_in_cam.inputs[0])

        # Negate: camera is at origin in camera space, so direction from
        # object to camera = -(object's camera-space position)
        negate = nodes.new('ShaderNodeVectorMath')
        negate.operation = 'MULTIPLY'
        negate.inputs[1].default_value = (-1.0, -1.0, -1.0)
        negate.location = (-2000, 100)
        links.new(obj_in_cam.outputs[0], negate.inputs[0])

        # CAMERA → WORLD as a Vector (no translation) → world direction
        v_transform = nodes.new('ShaderNodeVectorTransform')
        v_transform.vector_type = 'VECTOR'
        v_transform.convert_from = 'CAMERA'
        v_transform.convert_to = 'WORLD'
        v_transform.location = (-1800, 100)
        links.new(negate.outputs[0], v_transform.inputs[0])

        # Normalize
        norm = nodes.new('ShaderNodeVectorMath')
        norm.operation = 'NORMALIZE'
        norm.location = (-1600, 100)
        links.new(v_transform.outputs[0], norm.inputs[0])

        # Split into components
        sep = nodes.new('ShaderNodeSeparateXYZ')
        sep.location = (-1400, 100)
        links.new(norm.outputs[0], sep.inputs[0])

        # ---- Azimuth = atan2(y, x), normalized to [0, 1] ----
        atan2 = nodes.new('ShaderNodeMath')
        atan2.operation = 'ARCTAN2'
        atan2.location = (-1200, 220)
        links.new(sep.outputs['Y'], atan2.inputs[0])
        links.new(sep.outputs['X'], atan2.inputs[1])

        az_n = nodes.new('ShaderNodeMath')
        az_n.operation = 'MULTIPLY_ADD'
        az_n.inputs[1].default_value = 1.0 / (2.0 * math.pi)
        az_n.inputs[2].default_value = 0.5
        az_n.location = (-1000, 220)
        links.new(atan2.outputs[0], az_n.inputs[0])

        # ---- Elevation = asin(z), normalized to [0, 1] ----
        asin = nodes.new('ShaderNodeMath')
        asin.operation = 'ARCSINE'
        asin.location = (-1200, -20)
        links.new(sep.outputs['Z'], asin.inputs[0])

        el_n = nodes.new('ShaderNodeMath')
        el_n.operation = 'MULTIPLY_ADD'
        el_n.inputs[1].default_value = 1.0 / math.pi
        el_n.inputs[2].default_value = 0.5
        el_n.location = (-1000, -20)
        links.new(asin.outputs[0], el_n.inputs[0])

        # Pack into LUT sample UV
        lut_uv = nodes.new('ShaderNodeCombineXYZ')
        lut_uv.location = (-800, 100)
        links.new(az_n.outputs[0], lut_uv.inputs[0])
        links.new(el_n.outputs[0], lut_uv.inputs[1])

        links.new(lut_uv.outputs[0], lut_tex.inputs['Vector'])

        # ---- Local UV (planar) ----
        uvmap = nodes.new('ShaderNodeUVMap')
        uvmap.uv_map = 'UVMap'
        uvmap.location = (-1200, -400)

        cell_size = nodes.new('ShaderNodeCombineXYZ')
        cell_size.inputs[0].default_value = 1.0 / cols
        cell_size.inputs[1].default_value = 1.0 / rows
        cell_size.inputs[2].default_value = 0.0
        cell_size.location = (-1000, -500)

        scaled = nodes.new('ShaderNodeVectorMath')
        scaled.operation = 'MULTIPLY'
        scaled.location = (-600, -400)
        links.new(uvmap.outputs[0], scaled.inputs[0])
        links.new(cell_size.outputs[0], scaled.inputs[1])

        # Final atlas UV = LUT.rg + planar_uv * cell_size
        atlas_uv = nodes.new('ShaderNodeVectorMath')
        atlas_uv.operation = 'ADD'
        atlas_uv.location = (-300, 0)
        links.new(lut_tex.outputs['Color'], atlas_uv.inputs[0])
        links.new(scaled.outputs[0], atlas_uv.inputs[1])

        links.new(atlas_uv.outputs[0], atlas_tex.inputs['Vector'])

        # ---- Emission + transparent mix ----
        emit = nodes.new('ShaderNodeEmission')
        emit.inputs['Strength'].default_value = 1.0
        emit.location = (200, 100)
        links.new(atlas_tex.outputs['Color'], emit.inputs['Color'])

        trans = nodes.new('ShaderNodeBsdfTransparent')
        trans.location = (200, -100)

        mix = nodes.new('ShaderNodeMixShader')
        mix.location = (450, 0)
        links.new(atlas_tex.outputs['Alpha'], mix.inputs['Fac'])
        links.new(trans.outputs[0], mix.inputs[1])
        links.new(emit.outputs[0], mix.inputs[2])

        out = nodes.new('ShaderNodeOutputMaterial')
        out.location = (650, 0)
        links.new(mix.outputs[0], out.inputs['Surface'])


# ============================================================================
# REGISTRATION
# ============================================================================

CLASSES = (
    SIB_Slice,
    SIB_Settings,
    SIB_UL_Slices,
    SIB_PT_Main,
    SIB_OT_RebuildSlices,
    SIB_OT_PresetEquatorWeighted,
    SIB_OT_PresetUniform,
    SIB_OT_ShowCameras,
    SIB_OT_ClearCameras,
    SIB_OT_Bake,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.sib_settings = PointerProperty(type=SIB_Settings)
    bpy.app.timers.register(_init_default_slices, first_interval=0.1)


def _init_default_slices():
    try:
        for scene in bpy.data.scenes:
            s = scene.sib_settings
            if len(s.slices) == 0:
                _redistribute_slices(s)
    except Exception:
        pass
    return None


def unregister():
    _stop_billboard_timer()
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass
    if hasattr(bpy.types.Scene, 'sib_settings'):
        del bpy.types.Scene.sib_settings


if __name__ == "__main__":
    register()
