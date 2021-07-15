import json as json
import sciris as sc
import covasim as cv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Set, Union

CONFIG = {
    "params": {
        "pop_size": 2e3,
        "pop_infected": 1,
        "start_day": '2020-04-01',
        "end_day": '2020-05-25'
    },
    "daily_testing": {
        "prob": 0.1,
        "start_date": '2020-04-02',
        "end_date": '2020-05-25'
    },
    "output_json": "demo.json",
    "sim": {
        "maximum_repeats": 5,
        "initial_seed": 1
    }
}

test_spec = cv.test_prob(
    symp_prob = CONFIG["daily_testing"]["prob"],
    start_day = CONFIG["daily_testing"]["start_date"],
    end_day = CONFIG["daily_testing"]["end_date"]
)

sim = cv.Sim(
    CONFIG["params"],
    interventions=test_spec
)
sim.set_seed(CONFIG["sim"]["initial_seed"])
sim.run()

sim_count = 1
while sim_count < CONFIG["sim"]["maximum_repeats"]:
    if sim.summary["cum_diagnoses"] > 1:
        break
    else:
        print("repeating the simulation...")
        sim = cv.Sim(
            CONFIG["params"],
            interventions=test_spec
        )
        sim.set_seed(CONFIG["sim"]["initial_seed"] + sim_count)
        sim.run()
        sim_count += 1


all_people = sim.people.to_people()
transmission_tree = sim.make_transtree(to_networkx=True)
seed_uids = [
    e["target"]
    for e in transmission_tree.infection_log
    if e["layer"] == "seed_infection"
]
diagnosed_people = [p for p in all_people if p.diagnosed]

def first_pass_uids(leaf_people: List[cv.Person],
                    tt: cv.TransTree,
                    max_loops: int) -> Set[np.int32]:
    """return a list of all the people that are ancestral to one of the leaf
    people."""
    curr_people = set()
    for p in leaf_people:
        curr_people.add(p.uid)
    next_people = set()

    result = set()
    for p in curr_people:
        result.add(p)

    loop_counter = 0
    while loop_counter < max_loops:
        for cp in curr_people:
            for np in tt.graph.predecessors(cp):
                if np is not None:
                    next_people.add(np)

        if len(next_people) > 0:
            for p in next_people:
                result.add(p)
            curr_people, next_people = next_people, set()
            loop_counter += 1
        else:
            break

    return result

fp_uids = first_pass_uids(diagnosed_people, transmission_tree, 100)
sub_trans_tree = transmission_tree.graph.subgraph(fp_uids)

is_diagnosed = {p.uid: p.diagnosed for p in all_people}
diagnosis_dates = {dp.uid: dp.date_diagnosed for dp in diagnosed_people}

infection_date = {p.uid: p.date_exposed for p in all_people if not np.isnan(p.date_exposed)}

def predecessors(t, n):
    assert t.has_node(n)
    return list(t.predecessors(n))

def has_single_pred(t, n):
    return len(predecessors(t, n)) == 1

def successors(t, n):
    assert t.has_node(n)
    return list(t.successors(n))

def has_single_succ(t, n):
    return len(successors(t, n)) == 1

def remove_undiagnosed(t, n, is_diagnosed):
    assert not is_diagnosed[n]
    assert t.has_node(n)
    assert has_single_pred(t, n)
    assert has_single_succ(t, n)

    pred = predecessors(t, n)[0]
    succ = successors(t, n)[0]
    t.add_edge(pred, succ)
    t.remove_node(n)
    return None

def resolve_diagnosed(t, n, is_diagnosed, diag_date_dict):
    assert is_diagnosed[n]
    assert t.has_node(n)
    assert has_single_pred(t, n)
    assert has_single_succ(t, n)

    pred = predecessors(t, n)[0]
    succ = successors(t, n)[0]
    nid = "internal diagnosis node: {n} on day {d}".format(n=n, d=diag_date_dict[n])
    nx.relabel.relabel_nodes(t, {n: nid}, copy=False)


def split_node(t, n, is_diagnosed, diag_date_func, inf_date_dict):
    assert t.has_node(n)
    assert has_single_pred(t, n)
    assert not has_single_succ(t, n)

    if is_diagnosed[n]:
        _split_diagnosed(t, n, diag_date_func[n], inf_date_dict)
    else:
        _split_undiagnosed(t, n, inf_date_dict)


def _split_diagnosed(t, n, diag_date, inf_date_dict):
    pred = predecessors(t, n)[0]
    succs = successors(t, n)

    inf_dates = list(set(inf_date_dict[s] for s in succs))
    inf_dates.sort()

    if diag_date in inf_dates:
        raise NotImplemented("case of diagnosis occurring on the same day as infection.")
    else:
        pre_diag_inf_dates = filter(lambda d: d < diag_date, inf_dates)
        post_diag_inf_dates = filter(lambda d: d > diag_date, inf_dates)

        tmp = pred
        for inf_d in pre_diag_inf_dates:
            ss = filter(lambda s: inf_date_dict[s] == inf_d, succs)
            inf_node_id = "infection by {n} on {inf_d}".format(n=n, inf_d=inf_d)
            t.add_node(inf_node_id)
            t.add_edge(tmp, inf_node_id)
            for s in ss:
                t.add_edge(inf_node_id, s)
            tmp = inf_node_id

        nid = "internal diagnosis node: {n}".format(n=n)
        t.add_node(nid)
        t.add_edge(tmp, nid)
        tmp = nid

        for inf_d in post_diag_inf_dates:
            ss = filter(lambda s: inf_date_dict[s] == inf_d, succs)
            inf_node_id = "infection by {n} on {inf_d}".format(n=n, inf_d=inf_d)
            t.add_node(inf_node_id)
            t.add_edge(tmp, inf_node_id)
            for s in ss:
                t.add_edge(inf_node_id, s)
            tmp = inf_node_id

        t.remove_node(n)


def _split_undiagnosed(t, n, inf_date_dict):
    pred = predecessors(t, n)[0]
    succs = successors(t, n)

    inf_dates = list(set(inf_date_dict[s] for s in succs))
    inf_dates.sort()

    tmp = pred
    for inf_d in inf_dates:
        ss = [s for s in succs if inf_date_dict[s] == inf_d]
        inf_node_id = "infection by {n} on {inf_d}".format(n=n, inf_d=inf_d)
        t.add_node(inf_node_id)
        t.add_edge(tmp, inf_node_id)
        for s in ss:
            t.add_edge(inf_node_id, s)
        tmp = inf_node_id
    t.remove_node(n)

def second_pass_reconstruction(t: nx.DiGraph,
                               root_uid: np.int64,
                               max_loops: int) -> None:
    curr_nodes: List[np.int64] = [root_uid]
    loop_count: int = 0
    cn: np.int64
    while len(curr_nodes) > 0 and loop_count < max_loops:
        loop_count += 1
        cn = curr_nodes.pop()
        succs = successors(t, cn)
        num_succs = len(succs)
        curr_nodes = succs + curr_nodes
        if has_single_pred(t, cn):
            if num_succs == 1:
                if is_diagnosed[cn]:
                    resolve_diagnosed(t, cn, is_diagnosed, diagnosis_dates)
                else:
                    remove_undiagnosed(t, cn, is_diagnosed)
            elif num_succs > 1:
                split_node(t, cn, is_diagnosed, diagnosis_dates, infection_date)
            else:
                # this must be a leaf node.
                pass
        else:
            # this must be the root node.
            pass

    assert loop_count < max_loops, "more loops are probably needed!"
    return None

# Example of how to mutate the transmission tree.

assert len(seed_uids) == 1
assert not is_diagnosed[seed_uids[0]]

tmp2 = sub_trans_tree.copy()

nx.draw_planar(tmp2, with_labels = True)
plt.savefig("demo-tmp2-preprocessing.png")
plt.clf()

second_pass_reconstruction(tmp2, seed_uids[0], 200)

nx.draw_planar(tmp2, with_labels = True)
plt.savefig("demo-tmp2-postprocessing.png")
plt.clf()


