import numpy as np
import pytest
from pytest import raises
from typing import List, Tuple

from evogym.utils import (
    VOXEL_TYPES,
    get_uniform, draw, sample_robot,
    is_connected, has_actuator, get_full_connectivity
)
 
@pytest.mark.lite
def test_get_uniform():
    ones = get_uniform(1)
    assert np.allclose(ones, np.ones(1)), (
        f"Expected {np.ones(1)}, got {ones}"
    )
    
    one_thirds = get_uniform(3)
    assert np.allclose(one_thirds, np.ones(3) / 3), (
        f"Expected {np.ones(3) / 3}, got {one_thirds}"
    )

@pytest.mark.lite    
def test_draw():
    result = draw([0.2])
    assert result == 0, f"Expected 0, got {result}"
    
    result = draw([0.2, 0])
    assert result == 0, f"Expected 0, got {result}"
    
    result = draw([0, 15])
    assert result == 1, f"Expected 1, got {result}"
    
    pd = np.zeros(10)
    pd[5] = 1
    result = draw(pd)
    assert result == 5, f"Expected 5, got {result}"
    
    pd = np.ones(10)
    for i in range(10):
        result = draw(pd)
        assert result in list(range(10)), f"Expected result to be between 0 and 9, got {result}"
    
@pytest.mark.lite    
def test_has_actuator():
    h_act, v_act = VOXEL_TYPES['H_ACT'], VOXEL_TYPES['V_ACT']
    others = [
        i for i in VOXEL_TYPES.values() if i not in [h_act, v_act]
    ]
    
    robot = np.zeros((1, 1))
    robot[:, :] = others[0]
    assert not has_actuator(robot), "Expected no actuator"
    
    robot[:, :] = h_act
    assert has_actuator(robot), "Expected actuator"
    
    robot[:, :] = v_act
    assert has_actuator(robot), "Expected actuator"
    
    robot = np.random.choice(others, (10, 10), replace=True)
    assert not has_actuator(robot), "Expected no actuator"
    
    robot[5, 5] = h_act
    assert has_actuator(robot), "Expected actuator"
    
    robot[5, 5] = v_act
    assert has_actuator(robot), "Expected actuator"
    
    robot[1, 1] = h_act
    assert has_actuator(robot), "Expected actuator"
    
    robot = np.random.choice([h_act, v_act], (10, 10), replace=True)
    assert has_actuator(robot), "Expected actuator"
    
def test_is_connected():
    empty = VOXEL_TYPES['EMPTY']
    others = [
        i for i in VOXEL_TYPES.values() if i != empty
    ]
    
    robot = np.zeros((1, 1))
    robot[:, :] = empty
    assert not is_connected(robot), "Expected not connected"
    
    for val in others:
        robot[:, :] = val
        assert is_connected(robot), "Expected connected"
    
    robot = np.array([[others[0]], [empty], [others[1]]])
    assert not is_connected(robot), "Expected not connected"
    assert not is_connected(robot.T), "Expected not connected"
    
    robot = np.array([
        [others[0], empty, others[0]],
        [others[1], empty, others[3]],
        [others[2], others[1], others[0]]
    ])
    assert is_connected(robot), "Expected connected"
    assert is_connected(robot.T), "Expected connected"
    
    robot = np.array([
        [empty, empty, empty],
        [empty, others[2], empty],
        [empty, empty, empty]
    ])
    assert is_connected(robot), "Expected connected"
    
    robot = np.array([
        [others[0], others[1], empty],
        [others[1], empty, others[1]],
        [empty, others[1], others[0]]
    ])
    assert not is_connected(robot), "Expected not connected"

@pytest.mark.lite    
def test_get_full_connectivity():
    empty = VOXEL_TYPES['EMPTY']
    others = [
        i for i in VOXEL_TYPES.values() if i != empty
    ]
    
    robot = np.zeros((1, 1))
    robot[:, :] = empty
    assert get_full_connectivity(robot).shape[1] == 0, "Expected no connections"
    assert get_full_connectivity(robot).shape[0] == 2, "Expected 2"
    
    robot[:, :] = others[0]
    assert get_full_connectivity(robot).shape[1] == 0, "Expected no connections"
    assert get_full_connectivity(robot).shape[0] == 2, "Expected 2"
    
    robot = np.array([[others[0], empty, others[0]]])
    connections = get_full_connectivity(robot)
    assert connections.shape[1] == 0, "Expected no connections"
    
    def connections_contains_all(connections: np.ndarray, expected: List[Tuple[int, int]]):
        connections_as_tuples = [
            (c[0], c[1]) for c in connections.T
        ]
        for i, j in expected:
            if (i, j) not in connections_as_tuples or (j, i) not in connections_as_tuples:
                return False
        return True
    
    robot = np.array([
        [others[0], empty, others[0]],
        [others[1], empty, others[1]],
    ])
    connections = get_full_connectivity(robot)
    assert connections.shape[1] == 2, "Expected 2 connections"
    assert connections.shape[0] == 2, "Expected 2"
    connections_contains_all(connections, [(0, 3), (2, 5)])
    
    
    robot = np.array([
        [others[0], others[2], empty],
        [empty, others[3], others[1]],
    ])
    connections = get_full_connectivity(robot)
    assert connections.shape[1] == 3, "Expected 2 connections"
    assert connections.shape[0] == 2, "Expected 2"
    connections_contains_all(connections, [(0, 1), (1, 4), (4, 5)])
    
@pytest.mark.lite
def test_sample_robot():
    
    h_act, v_act, empty = VOXEL_TYPES['H_ACT'], VOXEL_TYPES['V_ACT'], VOXEL_TYPES['EMPTY']
    
    bad_pd = np.ones(5)
    bad_pd[h_act] = 0
    bad_pd[v_act] = 0
    with raises(Exception):
        sample_robot((5,5), bad_pd)
         
    bad_pd = np.zeros(5)
    bad_pd[empty] = 1
    with raises(Exception):
        sample_robot((5,5), bad_pd)
        
    def check_robot(robot: np.ndarray, connections: np.ndarray):
        assert robot.shape == (5, 5), f"Expected shape (5, 5), got {robot.shape}"
        assert is_connected(robot), "Expected robot to be connected"
        assert has_actuator(robot), "Expected robot to have an actuator"
        assert np.allclose(get_full_connectivity(robot), connections), "Expected connections to be the same"
        
    robot, connections = sample_robot((5, 5))
    check_robot(robot, connections)
    
    pd = np.ones(5)
    pd[h_act] = 0
    robot, connections = sample_robot((5, 5), pd=pd)
    check_robot(robot, connections)
    
    pd = np.ones(5)
    pd[v_act] = 0
    robot, connections = sample_robot((5, 5), pd=pd)
    check_robot(robot, connections)
    
    pd = np.ones(5)
    pd[empty] = 0
    robot, connections = sample_robot((5, 5), pd=pd)
    check_robot(robot, connections)
    
    pd = np.zeros(5)
    pd[v_act] = 1
    robot, connections = sample_robot((5, 5), pd=pd)
    check_robot(robot, connections)
    
    pd = np.zeros(5)
    pd[h_act] = 1
    robot, connections = sample_robot((5, 5), pd=pd)
    check_robot(robot, connections)
    