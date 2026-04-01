                
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity

from conftest import get_test_device, initialize_entity


def test_compliance_body_ids():
    device = get_test_device()

    entity = Entity(get_g1_robot_cfg())
    entity, sim = initialize_entity(entity, device, num_envs=1)

    monitored_bodies = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    body_ids, body_names = entity.find_bodies(monitored_bodies)

    print("body_ids:", body_ids)
    print("body_names:", body_names)

    assert len(body_ids) == 2
    assert body_names == monitored_bodies
    