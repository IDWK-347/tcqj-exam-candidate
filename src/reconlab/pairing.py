"""Own declared pair graphs; no tensor processing or backend imports."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

ORDER_POLICY = 'complete_manifest_forward_block_then_reverse_v1'
BASIS = 'complete_all_unordered_pairs'
BASES = {'complete': BASIS, 'sequential_window': 'declared_capture_sequence_window',
         'explicit': 'explicit_manifest_edges'}


class PairingError(ValueError):
    """Invalid graph; disconnected graphs retain their components."""
    def __init__(self, message, *, components=()):
        super().__init__(message)
        self.components = components


def _digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _pair_id(id_a, id_b):
    return 'pair_' + _digest(sorted((id_a, id_b)))


def _validate_nodes(nodes):
    if not isinstance(nodes, tuple) or len(nodes) < 2:
        raise PairingError('Pairing requires an ordered tuple of at least two image IDs')
    if any(not isinstance(value, str) or not value.strip() for value in nodes):
        raise PairingError('Image IDs must be nonempty strings')
    if len(set(nodes)) != len(nodes):
        raise PairingError('Duplicate image IDs')


def validate_pairing_declaration(declaration, nodes):
    """Validate optional manifest metadata and return an owned copy.

Absence is represented by {}; explicit null is never a valid declaration.
Loader may validate a one-image batch, while constructing a graph requires two.
"""
    if not isinstance(declaration, dict) or set(declaration) - {'order_semantics', 'explicit_edges'}:
        raise PairingError('Manifest pairing must contain only order_semantics/explicit_edges')
    if ('order_semantics' in declaration
            and declaration['order_semantics'] not in ('capture_sequence', 'unordered')):
        raise PairingError('Unsupported manifest order_semantics; no order is inferred')
    if 'explicit_edges' in declaration:
        edges = declaration['explicit_edges']
        if not isinstance(edges, list):
            raise PairingError('explicit_edges must be a list of image-ID pairs')
        seen = set()
        for pair in edges:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(not isinstance(v, str) or v not in nodes for v in pair)):
                raise PairingError('Explicit endpoints must be known image IDs')
            key = tuple(sorted(pair))
            if pair[0] == pair[1] or key in seen:
                raise PairingError('Explicit self loops and duplicate/reverse edges are forbidden')
            seen.add(key)
    return deepcopy(declaration)


def _components(nodes, edges):
    neighbors = {node: set() for node in nodes}
    for edge in edges:
        neighbors[edge.id_a].add(edge.id_b)
        neighbors[edge.id_b].add(edge.id_a)
    components, visited = [], set()
    for node in nodes:
        if node in visited:
            continue
        reached, pending = set(), [node]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(neighbors[current] - reached)
        components.append(tuple(value for value in nodes if value in reached))
        visited.update(reached)
    return tuple(components)


@dataclass(frozen=True)
class PairEdge:
    id_a: str
    id_b: str
    pair_id: str
    basis: str = BASIS


@dataclass(frozen=True)
class PairGraph:
    nodes: tuple
    edges: tuple
    strategy: str = 'complete'
    schema_version: int = 1
    window: object = None
    order_semantics: object = None

    @property
    def connected_components(self):
        return _components(self.nodes, self.edges)

    def _payload(self):
        parameters = {'strategy': self.strategy}
        if self.strategy == 'sequential_window':
            parameters['window'] = self.window
        data = {'schema_version': self.schema_version, 'nodes': list(self.nodes),
            'edges': [dict(id_a=edge.id_a, id_b=edge.id_b, pair_id=edge.pair_id,
                           basis=edge.basis) for edge in self.edges],
            'strategy_parameters': parameters,
            'directed_order_policy': (ORDER_POLICY if self.strategy == 'complete'
                else 'manifest_forward_block_then_reverse_v1'), 'symmetrize': True}
        if self.strategy == 'sequential_window':
            data['order_semantics'] = self.order_semantics
        return data

    @property
    def graph_hash(self):
        return _digest(self._payload())

    def directed_items(self):
        validate_pair_graph(self)
        indexes = {node: index for index, node in enumerate(self.nodes)}
        forward = []
        for edge in self.edges:
            first, second = sorted((edge.id_a, edge.id_b), key=indexes.get, reverse=True)
            forward.append({'pair_id': edge.pair_id, 'source': first, 'target': second})
        return forward + [dict(pair_id=item['pair_id'], source=item['target'], target=item['source'])
                          for item in forward]

    def metadata(self):
        validate_pair_graph(self)
        return {**self._payload(), 'graph_hash': self.graph_hash,
            'connected_components': [list(values) for values in self.connected_components],
            'logical_edge_count': len(self.edges), 'directed_item_count': 2 * len(self.edges),
            'directed_items': self.directed_items(),
            'actual_model_forward_calls': {'status': 'unavailable', 'value': None,
                'reason': 'Pairing does not observe model execution'},
            'visual_overlap': {'status': 'unverified',
                'reason': 'Structural connectivity does not prove visual overlap'}}


def validate_pair_graph(graph):
    if not isinstance(graph, PairGraph):
        raise PairingError('Expected an own PairGraph')
    if type(graph.schema_version) is not int or graph.schema_version != 1:
        raise PairingError('PairGraph requires integer schema_version=1')
    if not isinstance(graph.strategy, str) or graph.strategy not in BASES:
        raise PairingError('Unsupported pairing strategy')
    _validate_nodes(graph.nodes)
    if graph.strategy == 'sequential_window':
        if type(graph.window) is not int or graph.window <= 0 or graph.order_semantics != 'capture_sequence':
            raise PairingError('Sequential graph requires positive window and capture_sequence')
    elif graph.window is not None or graph.order_semantics is not None:
        raise PairingError('Unused graph window/order_semantics')
    if not isinstance(graph.edges, tuple):
        raise PairingError('Edges must be an ordered tuple')
    nodes, seen = set(graph.nodes), set()
    for edge in graph.edges:
        if not isinstance(edge, PairEdge):
            raise PairingError('Edges must be own PairEdge objects')
        if edge.id_a not in nodes or edge.id_b not in nodes:
            raise PairingError('Edge endpoint is outside the graph nodes')
        if edge.id_a == edge.id_b:
            raise PairingError('Self loops are forbidden')
        key = tuple(sorted((edge.id_a, edge.id_b)))
        if key in seen:
            raise PairingError('Duplicate undirected edge')
        seen.add(key)
        if (edge.id_a, edge.id_b) != key:
            raise PairingError('Edge endpoints must use canonical ID order')
        if edge.pair_id != _pair_id(*key) or edge.basis != BASES[graph.strategy]:
            raise PairingError('Edge pair ID or generation basis is inconsistent')
    components = graph.connected_components
    if len(components) != 1:
        raise PairingError(f'Disconnected pair graph: {components}', components=components)
    expected = [tuple(sorted((graph.nodes[i], graph.nodes[j])))
                for i in range(len(graph.nodes)) for j in range(i)
                if (graph.strategy == 'complete'
                    or graph.strategy == 'sequential_window' and i - j <= graph.window
                    or graph.strategy == 'explicit' and tuple(sorted((graph.nodes[i], graph.nodes[j]))) in seen)]
    if [(edge.id_a, edge.id_b) for edge in graph.edges] != expected:
        raise PairingError('Strategy topology or manifest-based edge order differs from contract')
    return graph


def build_pair_graph(images, pairing_config):
    """Consume ImageBatch identities only; keep manifest order unchanged."""
    if not isinstance(pairing_config, dict):
        raise PairingError('Pairing config must be a dictionary')
    strategy = pairing_config.get('strategy')
    if not isinstance(strategy, str) or strategy not in BASES:
        raise PairingError('Unsupported pairing strategy')
    expected_keys = {'strategy', 'window'} if strategy == 'sequential_window' else {'strategy'}
    if set(pairing_config) != expected_keys:
        raise PairingError('Missing or unused strategy parameters')
    window = pairing_config.get('window')
    if strategy == 'sequential_window' and (type(window) is not int or window <= 0):
        raise PairingError('window must be a positive integer')
    records = getattr(images, 'records', None)
    if not isinstance(records, (list, tuple)):
        raise PairingError('Expected ImageBatch.records in manifest order')
    nodes = tuple(getattr(record, 'image_id', None) for record in records)
    _validate_nodes(nodes)
    for index, record in enumerate(records):
        if type(record.manifest_index) is not int or record.manifest_index != index:
            raise PairingError('ImageBatch manifest_index differs from record order')
    declaration = getattr(images, 'pairing_declaration', None)
    declaration = validate_pairing_declaration({} if declaration is None else declaration, nodes)
    order = None
    selected = None
    if strategy == 'sequential_window':
        order = declaration.get('order_semantics')
        if order != 'capture_sequence':
            raise PairingError('sequential_window requires manifest capture_sequence; no order is inferred')
    elif strategy == 'explicit':
        if 'explicit_edges' not in declaration:
            raise PairingError('explicit requires manifest explicit_edges')
        selected = {tuple(sorted(edge)) for edge in declaration['explicit_edges']}
    edges = []
    for i in range(len(nodes)):
        for j in range(i):
            id_a, id_b = sorted((nodes[i], nodes[j]))
            if (strategy == 'sequential_window' and i - j > window
                    or strategy == 'explicit' and (id_a, id_b) not in selected):
                continue
            edges.append(PairEdge(id_a, id_b, _pair_id(id_a, id_b), BASES[strategy]))
    return validate_pair_graph(PairGraph(nodes, tuple(edges), strategy, window=window, order_semantics=order))


def to_directed_pairs(graph, backend_images, *, image_ids):
    """Bridge existing backend dictionaries by reference, without copying pixels."""
    validate_pair_graph(graph)
    if tuple(image_ids) != graph.nodes or len(backend_images) != len(graph.nodes):
        raise PairingError('Backend image identities/order/count differ from graph nodes')
    for index, image in enumerate(backend_images):
        if (not isinstance(image, dict) or type(image.get('idx')) is not int
                or image['idx'] != index):
            raise PairingError('Backend image idx differs from manifest order')
    indexes = {node: index for index, node in enumerate(graph.nodes)}
    return [(backend_images[indexes[item['source']]], backend_images[indexes[item['target']]])
            for item in graph.directed_items()]
