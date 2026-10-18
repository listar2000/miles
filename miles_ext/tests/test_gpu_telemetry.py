from miles_ext.obs.gpu_telemetry import NodeRole, node_roles

IPS = {"a": "10.0.0.1", "b": "10.0.0.2", "c": "10.0.0.3"}


def test_disaggregated_nodes_get_one_label_each():
    gpus = [(node, gpu) for node in "abc" for gpu in range(2)]
    assert node_roles(gpus, num_train_gpus=4, colocate=False, node_ips=IPS) == [
        NodeRole("train-node0", "a", "10.0.0.1", [0, 1]),
        NodeRole("train-node1", "b", "10.0.0.2", [0, 1]),
        NodeRole("rollout-node2", "c", "10.0.0.3", [0, 1]),
    ]


def test_node_split_between_roles_gets_two_labels():
    gpus = [("a", 3), ("a", 1), ("a", 2), ("a", 0)]
    assert node_roles(gpus, num_train_gpus=2, colocate=False, node_ips=IPS) == [
        NodeRole("train-node0", "a", "10.0.0.1", [1, 3]),
        NodeRole("rollout-node0", "a", "10.0.0.1", [0, 2]),
    ]


def test_colocated_gpus_do_both():
    gpus = [("a", 0), ("a", 1)]
    assert node_roles(gpus, num_train_gpus=0, colocate=True, node_ips=IPS) == [
        NodeRole("colocated-node0", "a", "10.0.0.1", [0, 1])
    ]
