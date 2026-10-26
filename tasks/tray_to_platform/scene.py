"""High-wall empty tray and 10 cm support; shared Panda physics stays unchanged."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from tasks.adapters.tray import initial_state as sample_tray_state, reset, dataset_adapter
from simulation.mujoco.dual_arm.task_scenes import configure_paper_scene, numbers, target_surface_z


def validate_geometry(config):
    s = config['scene']; p = s['paper_object']; t = s['platform_task']
    for key in ('side_wall_height_above_floor_m', 'end_wall_height_above_floor_m',
                'end_wall_thickness_m', 'grip_height_m', 'minimum_lift_m',
                'lift_stable_seconds', 'transport_clearance_m'):
        if type(t[key]) not in (int, float) or not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError('Invalid platform_task.'+key)
    height = 2*s['object_half_size'][2]
    if not np.isclose(height, p['bottom_thickness_m']+t['end_wall_height_above_floor_m']):
        raise ValueError('Reference half-height must match the actual highest end wall')
    if t['side_wall_height_above_floor_m'] > t['end_wall_height_above_floor_m']:
        raise ValueError('Grasp ends must be at least as tall as side walls')
    if not 0 < p['handle_width_m'] < .08 or not 0 < t['minimum_open_fraction'] <= 1:
        raise ValueError('Grip must fit unmodified 80 mm Panda aperture')
    if abs(t['grip_center_z_m'])+t['grip_height_m']/2 > height/2:
        raise ValueError('Grasp block extends beyond reference vertical bounds')
    if not np.allclose(s['grasp_offsets'], [[0.,-(p['pan_length_m']+p['handle_length_m'])/2,t['grip_center_z_m']],
                                          [0., (p['pan_length_m']+p['handle_length_m'])/2,t['grip_center_z_m']]]):
        raise ValueError('Grasp offsets disagree with the physical grip blocks')


def configure_xml(root, config):
    validate_geometry(config)
    s = config['scene']; p = s['paper_object']; t = s['platform_task']
    configure_paper_scene(root, s)
    # The shared XML has fixed legacy pedestal coordinates. This task moves
    # the robots, so move their physical supports with them (top = base Z).
    for side in ('left', 'right'):
        base = s[f'{side}_base_position']
        pedestal = root.find(f".//body[@name='{side}_pedestal']")
        geom = pedestal.find('geom')
        half_size = np.fromstring(geom.get('size'), sep=' ')
        half_size[2] = base[2] / 2
        pedestal.set('pos', numbers([base[0], base[1], base[2] / 2]))
        geom.set('size', numbers(half_size))
    platform = s.get('platform_appearance', {})
    if 'rgba' in platform:
        color = np.asarray(platform['rgba'], dtype=float)
        if color.shape != (4,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
            raise ValueError('platform_appearance.rgba must contain four values in [0, 1]')
        # Appearance only: retain the same solid support box and friction.
        ET.SubElement(root.find('asset'), 'material', name='platform_matte_wood',
                      rgba=numbers(color), specular='0', shininess='0', reflectance='0')
        geom = root.find(".//geom[@name='placement_platform_geom']")
        geom.set('rgba', numbers(color))
        geom.set('material', 'platform_matte_wood')
    show_marker = platform.get('show_target_marker', True)
    if type(show_marker) is not bool:
        raise ValueError('platform_appearance.show_target_marker must be boolean')
    if not show_marker:
        for geom in root.find(".//body[@name='target']").findall('geom'):
            geom.set('rgba', '0 0 0 0')
    body = root.find(".//body[@name='cooperative_object']")
    bottom_z = -s['object_half_size'][2]
    for index, sign in ((1,-1), (4,1)):
        geom = body.find(f"geom[@name='object_part_{index}']")
        geom.set('pos', numbers([sign*(p['pan_width_m']-p['wall_thickness_m'])/2,0,
                                bottom_z+p['bottom_thickness_m']+t['side_wall_height_above_floor_m']/2]))
        geom.set('size', numbers([p['wall_thickness_m']/2,p['pan_length_m']/2,t['side_wall_height_above_floor_m']/2]))
    for index, sign in ((2,-1), (5,1)):
        geom = body.find(f"geom[@name='object_part_{index}']")
        geom.set('pos', numbers([0,sign*(p['pan_length_m']-t['end_wall_thickness_m'])/2,
                                bottom_z+p['bottom_thickness_m']+t['end_wall_height_above_floor_m']/2]))
        geom.set('size', numbers([(p['pan_width_m']-2*p['wall_thickness_m'])/2,t['end_wall_thickness_m']/2,t['end_wall_height_above_floor_m']/2]))
    for index, sign in ((3,-1), (6,1)):
        geom = body.find(f"geom[@name='object_part_{index}']")
        geom.set('pos', numbers([0,sign*(p['pan_length_m']+p['handle_length_m'])/2,t['grip_center_z_m']]))
        geom.set('size', numbers([p['handle_width_m']/2,p['handle_length_m']/2,t['grip_height_m']/2]))
    # Non-overlapping box volumes allocate a uniform effective density. MuJoCo
    # derives the combined center of mass and principal inertia from these masses.
    parts = body.findall('geom')
    volumes = [float(8*np.prod(np.fromstring(g.get('size'), sep=' '))) for g in parts]
    for geom, volume in zip(parts, volumes):
        geom.set('mass', str(s['object_mass_kg']*volume/sum(volumes)))
    for side, offset in zip(('left','right'), s['grasp_offsets']):
        site = body.find(f"site[@name='{side}_grasp_site']")
        site.set('pos', numbers(offset))
        # EE local y closes along tray x; EE local z points downward.
        site.set('quat', numbers([0,2**-.5,2**-.5,0]))


def initial_state(config, seed):
    validate_geometry(config)
    state = sample_tray_state(config, seed)
    # reference origin sits at the full shell bounding-box midpoint, not COM.
    # bottom local z = -object_half_size[2].
    expected = target_surface_z(config['scene']) + config['scene']['object_half_size'][2]
    if not np.isclose(state['target_position'][2], expected):
        raise ValueError('Target origin does not place the tray bottom on the platform')
    return state


def preflight(config, seeds, demonstration_seeds=(), engineering_seeds=()):
    from rollout.protocol import preflight_cases, _waypoint_checks
    from simulation.mujoco.dual_arm.task_controller import DualArmTaskController
    seeds = list(seeds)
    report = preflight_cases(config, seeds, demonstration_seeds, engineering_seeds)
    c = DualArmTaskController(config)
    checks=[]
    cases = {row['seed']: row for row in report['cases']}
    for seed in seeds:
        c.reset(seed)
        pos=np.array(c.initial_state['object_position'])
        pos[2]=target_surface_z(config['scene'])+config['scene']['object_half_size'][2]+config['scene']['platform_task']['transport_clearance_m']
        above_target=pos.copy();above_target[:2]=c.target_position[:2]
        waypoints = (_waypoint_checks(c,pos,c.get_status()['object_yaw_rad'],'clear_platform')
                     + _waypoint_checks(c,above_target,c.target_yaw,'above_platform'))
        checks.append({'seed':seed, 'high_platform_waypoints':waypoints})
        # These are additional required reachability checks, not informational
        # annotations. Preserve every seed and propagate failures to the same
        # case flags and summary consumed by the formal rollout protocol.
        row = cases[seed]
        row['static_ik'].extend(waypoints)
        row['checks'].update({
            'platform_clearance_waypoints_in_workspace': all(w['inside_workspace'] for w in waypoints),
            'platform_clearance_waypoints_static_ik': all(w['static_ik_pass'] for w in waypoints),
        })
        row['checks']['all_waypoints_in_workspace'] &= row['checks']['platform_clearance_waypoints_in_workspace']
        row['checks']['all_waypoints_static_ik'] &= row['checks']['platform_clearance_waypoints_static_ik']
        row['flags'] = [key for key, passed in row['checks'].items() if not passed]
    report['high_platform_checks']=checks
    report['summary']['cases_with_flags'] = sum(bool(row['flags']) for row in report['cases'])
    report['summary']['flag_counts'] = {
        key: sum(not row['checks'][key] for row in report['cases'])
        for key in report['cases'][0]['checks']
    }
    return report


def collection_success(*args):
    from .success import collection_success as evaluate
    return evaluate(*args)


def rollout_monitor(config, initial_state, **options):
    from .success import PlatformSuccessMonitor
    return PlatformSuccessMonitor(config, initial_state, **options)
