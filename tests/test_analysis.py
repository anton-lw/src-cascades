import math

import networkx as nx

from src_cascades.analysis import analyze_network_criticality, analyze_traveling_wave

def test_analyze_traveling_wave_returns_paper_aligned_values():
    result = analyze_traveling_wave(0.038, 3.0)

    assert math.isclose(result["mu"], 3.3975167889, rel_tol=1e-6)
    assert math.isclose(result["v_max"], 0.5414269367, rel_tol=1e-6)
    assert math.isclose(result["n_c"], 10.65729105, rel_tol=1e-6)

def test_analyze_traveling_wave_is_nan_below_criticality():
    result = analyze_traveling_wave(0.02, 3.0)

    assert math.isnan(result["mu"])
    assert math.isnan(result["v_max"])
    assert math.isnan(result["n_c"])

def test_analyze_network_criticality_on_small_path_graph():
    graph = nx.path_graph(5)
    result = analyze_network_criticality(graph, p=0.4)

    assert math.isclose(result["effective_ell"], 0.75)
    assert result["p_c_network"] == math.inf
    assert result["is_supercritical"] is False
