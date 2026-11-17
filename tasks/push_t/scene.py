"""Compile source-frozen Push-T geometry; names are the only robot XML adaptation."""
from pathlib import Path
import copy
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from tasks.push_t.success import make_push_t_sample_points, push_t_coverage

ROOT = Path(__file__).resolve().parents[2]


def footprint_xy(position, yaw=0., *, target=False):
    """Outline of the frozen source T boxes, including the enlarged goal."""
    outer, stem, join = (.155, .050, .055) if target else (.150, .045, .060)
    outline = np.array([[-outer, -stem], [join, -stem], [join, -outer],
                        [outer, -outer], [outer, outer], [join, outer],
                        [join, stem], [-outer, stem]])
    c, s = math.cos(yaw), math.sin(yaw)
    return outline @ np.array([[c, s], [-s, c]]) + np.asarray(position)[:2]


def initial_state(config, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    p = config['scene']['push_t']; rng = np.random.default_rng(seed)
    margin = p['push_clearance_m']
    scene = config['scene']
    workspace = np.asarray(scene['left_workspace'])[:2]
    table = np.asarray(scene['table_position'])[:2]
    half = np.asarray(scene['table_half_size_xy'])
    lower = np.maximum(workspace[:, 0], table-half)
    upper = np.minimum(workspace[:, 1], table+half)
    def safe(outline):
        # A centre inside a radius is insufficient for this 300 mm T. Keep
        # every edge and an extra pusher approach margin inside the safe disk
        # and the configured workspace/table. Convex bounds cover the edges.
        return (np.all(np.linalg.norm(outline, axis=1)+margin <= p['safe_reach_radius_m'])
                and np.all(outline.min(axis=0)-margin >= lower)
                and np.all(outline.max(axis=0)+margin <= upper))
    def clears_home(obj, yaw):
        # Distance from the parked pusher to each of the two source boxes.
        # Using their union avoids rejecting empty space beside the T stem.
        c, s = math.cos(yaw), math.sin(yaw)
        local = (np.asarray(scene['left_home_ee'])[:2]-obj) @ np.array([[c, -s], [s, c]])
        lower = np.array([[.060, -.150], [-.150, -.045]])
        upper = np.array([[.150, .150], [.060, .045]])
        distance = np.linalg.norm(np.maximum(np.maximum(lower-local, local-upper), 0.), axis=1)
        return np.min(distance) >= p['home_pusher_clearance_m']
    target_position = np.asarray(scene['target_position'], dtype=float)
    target_yaw = float(scene['target_yaw_rad'])
    if target_position.shape != (3,) or not np.isfinite(target_position).all() or not math.isfinite(target_yaw):
        raise ValueError('Invalid fixed Push-T target pose')
    target = target_position[:2]
    if not safe(footprint_xy(target, target_yaw, target=True)):
        raise ValueError('Fixed Push-T target footprint and push clearance exceed reachable workspace')
    c, s = math.cos(target_yaw), math.sin(target_yaw)
    target_rotation = np.array([[c, -s], [s, c]])
    samples = make_push_t_sample_points(p['coverage_resolution'])
    for _ in range(10000):
        obj = rng.uniform([p['object_x_range'][0], p['object_y_range'][0]], [p['object_x_range'][1], p['object_y_range'][1]])
        yaw = float(np.deg2rad(rng.uniform(*p['yaw_degrees'])))
        object_outline = footprint_xy(obj, yaw)
        if (safe(object_outline) and np.linalg.norm(obj-target) >= p['object_min_separation_m']
                and clears_home(obj, yaw)):
            c, s = math.cos(yaw), math.sin(yaw)
            coverage = push_t_coverage(obj, np.array([[c, -s], [s, c]]), target, target_rotation, samples)
            # The visual goal may overlap the start, as in standard Push-T.
            # Reject starts close to completion without forcing a fixed side.
            if coverage <= p['initial_max_coverage']:
                break
    else:
        raise ValueError('Push-T randomization cannot satisfy footprint, push-clearance and reach constraints')
    return {'seed': seed, 'object_position': [*obj, .415 + p['initial_object_clearance_m']],
            'object_yaw_rad': yaw, 'object_quaternion_wxyz': [math.cos(yaw/2), 0, 0, math.sin(yaw/2)],
            'target_position': target_position.tolist(), 'target_yaw_rad': target_yaw}


def configure_xml(root, config):
    """Add aliases and auxiliary cameras without changing source physics."""
    world = root.find('worldbody')
    cube = world.find("body[@name='cube']")
    if cube is not None:
        cube.set('name', 'cooperative_object')
        cube.find('freejoint').set('name', 'object_freejoint')
        cube.find("geom[@name='push_t_bar']").set('name', 'object_geom')
    target = world.find("body[@name='target_plate']")
    if target is not None: target.set('name', 'target')
    world.find("body[@name='table']/geom").set('name', 'table_geom')
    hand = root.find(".//body[@name='left_hand']")
    if hand is not None and hand.find("site[@name='left_ee']") is None:
        ET.SubElement(hand, 'site', name='left_ee', pos='0 0 0', size='.002', rgba='0 0 0 0')
    for name in config['scene']['cameras']:
        if world.find(f"camera[@name='{name}']") is None:
            ET.SubElement(world, 'camera', name=name, pos='0 0 2', fovy='45')


def create_model(config):
    scene = ET.parse(ROOT/'tasks/push_t/assets/push_t_scene.xml').getroot()
    panda = ET.parse(ROOT/'tasks/push_t/assets/panda.xml').getroot()
    # Namespace body/joint/actuator/tendon names and every corresponding reference.
    names = {node.get('name') for tag in ('body', 'joint', 'general', 'position', 'fixed')
             for node in panda.iter(tag) if node.get('name')}
    refs = {'name', 'joint', 'joint1', 'joint2', 'body', 'body1', 'body2', 'tendon', 'target'}
    for node in panda.iter():
        for key in refs:
            if node.get(key) in names: node.set(key, 'left_'+node.get(key))
    root = ET.Element('mujoco', model='single Panda Push-T')
    merge = {'asset', 'worldbody', 'visual', 'default', 'contact', 'equality', 'tendon', 'actuator'}
    for source in (panda, scene):
        for child in source:
            if child.tag in {'include', 'keyframe'}: continue
            existing = root.find(child.tag)
            if existing is not None and child.tag in merge:
                for entry in child: existing.append(copy.deepcopy(entry))
            elif existing is not None:
                existing.attrib.update(child.attrib)
            else: root.append(copy.deepcopy(child))
    root.find('compiler').set('meshdir', '')
    visual = root.find('visual')
    global_node = visual.find('global')
    if global_node is None: global_node = ET.SubElement(visual, 'global')
    global_node.set('offwidth', str(config['recording']['width']))
    global_node.set('offheight', str(config['recording']['height']))
    configure_xml(root, config)
    mesh_dir = ROOT/'simulation/mujoco/assets/franka_emika_panda/assets'
    assets = {node.get('file'): (mesh_dir/node.get('file')).read_bytes() for node in root.findall('asset/mesh')}
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets=assets)
    # Convert MuJoCo's own free-camera convention into fixed auxiliary cameras.
    # This avoids guessing azimuth signs or using a different perspective model.
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    view = mujoco.MjvScene(model, maxgeom=2000); option = mujoco.MjvOption()
    for name, spec in config['scene']['cameras'].items():
        camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = spec['lookat']; camera.distance = spec['distance']
        camera.azimuth = spec['azimuth']; camera.elevation = spec['elevation']
        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, view)
        gl = view.camera[0]; cid = model.camera(name).id
        position = .5*(view.camera[0].pos + view.camera[1].pos)
        forward = np.asarray(gl.forward); up = np.asarray(gl.up)
        right = np.cross(forward, up); rotation = np.column_stack((right, up, -forward))
        model.cam_pos[cid] = position
        mujoco.mju_mat2Quat(model.cam_quat[cid], rotation.reshape(-1))
        model.cam_fovy[cid] = model.vis.global_.fovy
    return model


def reset(controller, object_position, yaw):
    return []


def preflight(config, seeds, demonstration_seeds=(), engineering_seeds=()):
    from simulation.mujoco.single_arm.task_controller import SingleArmTaskController
    controller = SingleArmTaskController(config)
    cases=[]
    for seed in seeds:
        controller.reset(int(seed)); controller.step(.4)
        status=controller.get_status()
        cases.append({'seed': int(seed), 'finite': bool(np.isfinite(controller.data.qpos).all()),
                      'initial_state': controller.initial_state, 'coverage': status['push_t_coverage']})
    return {'cases': cases, 'summary': {'count': len(cases), 'all_finite': all(c['finite'] for c in cases)}}
