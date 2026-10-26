"""Small parameterized rigid objects with matching visual/collision boxes."""
import xml.etree.ElementTree as ET
import numpy as np


def target_surface_z(scene):
    return scene['table_top_z'] + scene.get('placement_platform', {}).get('height_m', 0.)


def platform_size(scene):
    paper = scene['paper_object']
    return np.array([paper['pan_width_m'], paper['pan_length_m']]) + scene['placement_platform']['extra_size_xy_m']


def placement_geometry(controller):
    """Actual tray-base corners in the support's local frame; handles overhang."""
    c, scene = controller, controller.config['scene']
    if 'placement_platform' not in scene:
        return {'required': False, 'contained': True}
    body = c.model.body('placement_platform').id
    paper = scene['paper_object']
    corners = np.array([[x, y, -scene['object_half_size'][2]]
                        for x in (-paper['pan_width_m']/2, paper['pan_width_m']/2)
                        for y in (-paper['pan_length_m']/2, paper['pan_length_m']/2)])
    world = corners @ c.data.xmat[c.object_id].reshape(3, 3).T + c.data.xpos[c.object_id]
    local = (world-c.data.xpos[body]) @ c.data.xmat[body].reshape(3, 3)
    margin = platform_size(scene)/2 - np.max(np.abs(local[:, :2]), axis=0)
    return {'required': True, 'contained': bool(np.all(margin >= -scene['placement_platform']['containment_tolerance_m'])),
            'margin_xy_m': margin.tolist(), 'surface_z_m': target_surface_z(scene)}


def numbers(value):
    return " ".join(str(float(x)) for x in value)


def reset_paper_obstacles(model, scene, object_position, yaw):
    """Place S2 barriers once at reset, then leave them fixed to the table.

    Older frozen configurations without the opt-in retain fixed-world barriers.
    Their physical poses are included in ordinary body_pos/body_quat snapshots.
    """
    specification = scene.get('paper_object', {})
    if specification.get('kind') != 'blocked_tray':
        return {}
    follows = scene.get('blockers_follow_initial_pose', False)
    angle = yaw if follows else 0.
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    origin = np.asarray(object_position[:2]) if follows else np.zeros(2)
    poses = {}
    for side, sign in (('left', -1), ('right', 1)):
        body = model.body(f'blocker_{side}').id
        offset = [-.055, sign*(specification['pan_length_m']+specification['handle_length_m'])/2]
        position = np.r_[origin + rotation @ offset, scene['table_top_z']+.06]
        quaternion = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
        model.body_pos[body] = position
        model.body_quat[body] = quaternion
        poses[side] = {'position': position.tolist(), 'quaternion_wxyz': quaternion,
                       'reference': 'initial_object' if follows else 'fixed_world'}
    return poses


def configure_paper_scene(root, scene):
    specification = scene.get("paper_object")
    if not specification:
        return
    if specification["kind"] not in {"tray", "blocked_tray"}:
        raise ValueError("Unsupported Paper A object")
    matte = specification.get("visual_style") == "matte_v1"
    if matte:
        ET.SubElement(root.find("asset"), "material", name="paper_matte", specular="0", shininess="0", reflectance="0")
        root.find(".//geom[@name='table_geom']").set("material", "paper_matte")
    width, length = specification["pan_width_m"], specification["pan_length_m"]
    handle_width, handle_length = specification["handle_width_m"], specification["handle_length_m"]
    height, bottom = 2*scene["object_half_size"][2], specification["bottom_thickness_m"]
    wall = specification["wall_thickness_m"]
    # Each box has its own mass proportional to volume; MuJoCo computes
    # aggregate inertia. No fictitious mesh fills the recessed tray interior.
    parts = [([0, 0, -height/2+bottom/2], [width/2, length/2, bottom/2], [.75, .3, .16, 1])]
    for sign in (-1, 1):
        parts.append(([sign*(width-wall)/2, 0, bottom/2], [wall/2, length/2, (height-bottom)/2], [.85, .38, .2, 1]))
        parts.append(([0, sign*(length-wall)/2, bottom/2], [(width-2*wall)/2, wall/2, (height-bottom)/2], [.85, .38, .2, 1]))
        parts.append(([0, sign*(length+handle_length)/2, 0], [handle_width/2, handle_length/2, height/2], [.95, .7, .18, 1]))
    volume = sum(float(8*np.prod(size)) for _, size, _ in parts)
    body = root.find(".//body[@name='cooperative_object']")
    for geom in list(body.findall("geom")):
        body.remove(geom)
    for i, (pos, size, color) in enumerate(parts):
        part = ET.SubElement(body, "geom", name="object_geom" if i == 0 else f"object_part_{i}", type="box",
                      pos=numbers(pos), size=numbers(size), rgba=numbers(color), condim="6",
                      friction=numbers(scene["object_friction"]), mass=str(scene["object_mass_kg"]*8*np.prod(size)/volume))
        if matte:
            part.set("material", "paper_matte")
    for side, sign in (("left", -1), ("right", 1)):
        site = body.find(f"site[@name='{side}_grasp_site']")
        site.set("pos", numbers([0, sign*(length+handle_length)/2, 0]))
        site.set("rgba", "0 0 0 0")
    target = root.find(".//body[@name='target']")
    target.find("geom").set("size", numbers([width/2+.012, length/2+handle_length+.012, .0004]))
    # Render the footprint outline, allowing the operator to see orientation.
    target.find("geom").set("rgba", "0.2 0.85 0.3 0.2")
    for i, (pos, size, _) in enumerate(parts):
        ET.SubElement(target, "geom", name=f"target_part_{i}", type="box", pos=numbers([*pos[:2], 0]),
                      size=numbers([*size[:2], .0005]), rgba="0.2 0.85 0.3 0.45", contype="0", conaffinity="0")
    if 'placement_platform' in scene:
        height = scene['placement_platform']['height_m']
        platform = ET.SubElement(root.find('worldbody'), 'body', name='placement_platform',
            pos=numbers([*scene['target_position'][:2], scene['table_top_z']+height/2]))
        ET.SubElement(platform, 'geom', name='placement_platform_geom', type='box',
            size=numbers([*(platform_size(scene)/2), height/2]), rgba='0.24 0.48 0.58 1',
            friction=numbers(scene['object_friction']), condim='6')
        # The target overlay must not pretend the overhanging handles are supported.
        target.find('geom').set('size', numbers([*(platform_size(scene)/2), .0004]))
        for i in (3, 6):
            target.find(f"geom[@name='target_part_{i}']").set('rgba', '0 0 0 0')
    if specification["kind"] == "blocked_tray":
        for side, sign in (("left", -1), ("right", 1)):
            blocker = ET.SubElement(root.find("worldbody"), "body", name=f"blocker_{side}",
                                    pos=numbers([-.055, sign*(length+handle_length)/2, scene["table_top_z"]+.06]))
            ET.SubElement(blocker, "geom", name=f"blocker_{side}_geom", type="box", size="0.025 0.05 0.06", rgba="0.25 0.35 0.5 1")


def scenario_config(config, name):
    """Create one full frozen configuration. Callers must not merge it again."""
    import copy
    if name not in {"S0", "S1", "S2", "S3"}:
        raise ValueError("Unknown scene")
    result = copy.deepcopy(config)
    result['scene'].pop('placement_platform', None)
    if name == "S0":
        result["scene"].pop("paper_object", None)
    else:
        result["scene"]["paper_object"] = {"kind": "blocked_tray" if name == "S2" else "tray",
            "visual_style": "matte_v1",
            "pan_width_m": .24, "pan_length_m": .32, "handle_width_m": .036, "handle_length_m": .12,
            "bottom_thickness_m": .008, "wall_thickness_m": .014}
        result["scene"]["grasp_offsets"] = [[0, -.22, 0], [0, .22, 0]]
    if name == 'S3':
        import json
        from pathlib import Path
        # Resolve once at session creation. Replay compiles the frozen config
        # directly and never rereads this mutable authoring file.
        patch = json.loads((Path(__file__).resolve().parents[3] / 'config/scene_raised_placement.json').read_text(encoding='utf-8'))
        result['scene'].update(patch['scene'])
    result["scene"]["scenario"] = name
    return result
