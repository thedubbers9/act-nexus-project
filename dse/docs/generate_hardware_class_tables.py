#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

BASE = Path('/scratch/krish/MLIR-hardware-analysis/submodule/act/dse/docs')
XLA_CSV = BASE / 'xla_hlo_operation_reference.csv'
JOIN_CSV = BASE / 'taidl_xla_join_reference.csv'
CLASS_CSV = BASE / 'hardware_abstraction_classes.csv'
OP_CSV = BASE / 'xla_taidl_hardware_class_view.csv'

CLASS_INFO = {
    'logical_view': {
        'description': 'Logical or view-level tensor reinterpretation. These ops usually change shape, type view, or layout description without necessarily moving data.',
        'hardware_meaning': 'Usually handled by compiler/runtime metadata, address interpretation, or lightweight scalar/control logic unless later materialized.',
    },
    'contiguous_move': {
        'description': 'Regular, structured tensor movement with predictable contiguous or affine address patterns.',
        'hardware_meaning': 'Typically mapped to DMA engines, tensor movers, or scratchpad interconnect paths.',
    },
    'indexed_move': {
        'description': 'Indirect or data-dependent movement where addresses are computed from indices rather than simple affine ranges.',
        'hardware_meaning': 'Requires AGUs, gather/scatter hardware, or more capable memory-side engines than plain DMA.',
    },
    'tensor_compute': {
        'description': 'High-arithmetic-intensity tensor algebra such as matrix multiplication and convolution.',
        'hardware_meaning': 'Best matched to matrix engines, systolic arrays, tensor cores, or other MAC-dense compute blocks.',
    },
    'vector_compute': {
        'description': 'Regular elementwise arithmetic and logic across tensor lanes.',
        'hardware_meaning': 'Maps naturally to SIMD/vector ALUs with lane-wise execution.',
    },
    'special_math': {
        'description': 'Elementwise nonlinear or transcendental math operations.',
        'hardware_meaning': 'Often executed on SFUs, LUT/CORDIC blocks, or specialized vector math pipelines.',
    },
    'reduction': {
        'description': 'Ops that combine many values into fewer values, including windowed or associative aggregations.',
        'hardware_meaning': 'Typically handled by reduction trees, vector reduction hardware, or decomposed tensor engines depending on the pattern.',
    },
    'predication_select': {
        'description': 'Comparison, masking, clamping, and selection logic.',
        'hardware_meaning': 'Usually implemented in vector ALUs with predicate support or small dedicated compare/select logic.',
    },
    'control_runtime': {
        'description': 'Control flow, token/order management, runtime dispatch, or compiler/runtime structure ops.',
        'hardware_meaning': 'Handled by scalar/control processors, sequencers, or runtime support rather than bulk data engines.',
    },
    'collective_comm': {
        'description': 'Cross-core or cross-chip communication and synchronization primitives.',
        'hardware_meaning': 'Maps to network interfaces, collective engines, and interconnect protocols.',
    },
    'io_transfer': {
        'description': 'Explicit input/output transfer between accelerator and external environment.',
        'hardware_meaning': 'Typically serviced by IO fabrics, host interfaces, or DMA/control paths.',
    },
    'tuple_structural': {
        'description': 'IR-structural tuple packing/unpacking rather than real tensor math.',
        'hardware_meaning': 'Mostly runtime/compiler structure, often not a direct compute primitive.',
    },
    'random_generation': {
        'description': 'Random-number generation or RNG state manipulation.',
        'hardware_meaning': 'Requires RNG engines or specialized scalar/vector support.',
    },
    'specialized_compute': {
        'description': 'Domain-specific kernels like batch norm, FFT, or triangular solve that may require dedicated decomposition or specialized units.',
        'hardware_meaning': 'Often hybrid: may decompose into vector/tensor primitives or target dedicated IP.',
    },
    'input_literal': {
        'description': 'Program inputs, constants, and literal values entering the dataflow.',
        'hardware_meaning': 'Represents sources/sinks of data rather than active compute.',
    },
    'uncategorized': {
        'description': 'Fallback bucket for ops not yet assigned a clearer hardware-oriented abstraction.',
        'hardware_meaning': 'Needs manual review before serious cost modeling.',
    },
}

XLA_TO_CLASS = {
    'Constant': 'input_literal',
    'Parameter': 'input_literal',
    'Tuple': 'tuple_structural',
    'GetTupleElement': 'tuple_structural',
    'Bitcast': 'logical_view',
    'BitcastConvertType': 'logical_view',
    'Collapse': 'logical_view',
    'ConvertElementType': 'logical_view',
    'DynamicReshape': 'logical_view',
    'GetDimensionSize': 'logical_view',
    'SetDimensionSize': 'logical_view',
    'Iota': 'logical_view',
    'Reshape': 'logical_view',
    'Broadcast': 'logical_view',
    'Copy': 'contiguous_move',
    'ConcatInDim (Concatenate)': 'contiguous_move',
    'DynamicUpdateSlice': 'contiguous_move',
    'Pad': 'contiguous_move',
    'Rev (reverse)': 'contiguous_move',
    'Slice': 'contiguous_move',
    'Transpose': 'contiguous_move',
    'DynamicSlice': 'indexed_move',
    'Gather': 'indexed_move',
    'Scatter': 'indexed_move',
    'Sort': 'indexed_move',
    'Conv (Convolution)': 'tensor_compute',
    'Dot': 'tensor_compute',
    'Add': 'vector_compute',
    'And': 'vector_compute',
    'Atan2': 'vector_compute',
    'Complex': 'vector_compute',
    'Div': 'vector_compute',
    'Max': 'vector_compute',
    'Min': 'vector_compute',
    'Mul': 'vector_compute',
    'Or': 'vector_compute',
    'Pow': 'vector_compute',
    'Rem': 'vector_compute',
    'ShiftLeft': 'vector_compute',
    'ShiftRightArithmetic': 'vector_compute',
    'ShiftRightLogical': 'vector_compute',
    'Sub': 'vector_compute',
    'Xor': 'vector_compute',
    'Abs': 'special_math',
    'Cbrt': 'special_math',
    'Ceil': 'special_math',
    'Clz': 'special_math',
    'Cos': 'special_math',
    'Cosh': 'special_math',
    'Erf': 'special_math',
    'Exp': 'special_math',
    'Expm1': 'special_math',
    'Floor': 'special_math',
    'Imag': 'special_math',
    'IsFinite': 'special_math',
    'Log': 'special_math',
    'Log1p': 'special_math',
    'Logistic': 'special_math',
    'Neg': 'special_math',
    'Not': 'special_math',
    'PopulationCount': 'special_math',
    'Real': 'special_math',
    'Round': 'special_math',
    'Rsqrt': 'special_math',
    'Sign': 'special_math',
    'Sin': 'special_math',
    'Sqrt': 'special_math',
    'Tan': 'special_math',
    'Tanh': 'special_math',
    'Reduce': 'reduction',
    'ReducePrecision': 'reduction',
    'ReduceScatter': 'reduction',
    'ReduceWindow': 'reduction',
    'Scan': 'reduction',
    'TopK': 'reduction',
    'Clamp': 'predication_select',
    'Compare': 'predication_select',
    'Select': 'predication_select',
    'SelectAndScatter': 'predication_select',
    'AddDependency': 'control_runtime',
    'AfterAll': 'control_runtime',
    'Async': 'control_runtime',
    'Call': 'control_runtime',
    'Conditional': 'control_runtime',
    'CustomCall': 'control_runtime',
    'Domain': 'control_runtime',
    'Fusion': 'control_runtime',
    'Map': 'control_runtime',
    'OptimizationBarrier': 'control_runtime',
    'While': 'control_runtime',
    'AllGather': 'collective_comm',
    'AllReduce': 'collective_comm',
    'AllToAll': 'collective_comm',
    'CollectiveBroadcast': 'collective_comm',
    'CollectivePermute': 'collective_comm',
    'PartitionID': 'collective_comm',
    'ReplicaId': 'collective_comm',
    'Infeed': 'io_transfer',
    'Outfeed': 'io_transfer',
    'Recv': 'io_transfer',
    'Send': 'io_transfer',
    'BatchNormGrad': 'specialized_compute',
    'BatchNormInference': 'specialized_compute',
    'BatchNormTraining': 'specialized_compute',
    'Cholesky': 'specialized_compute',
    'Fft': 'specialized_compute',
    'RngBitGenerator': 'random_generation',
    'RngGetAndUpdateState': 'random_generation',
    'RngNormal': 'random_generation',
    'RngUniform': 'random_generation',
    'TriangularSolve': 'specialized_compute',
}

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    xla_rows = read_csv(XLA_CSV)
    join_rows = read_csv(JOIN_CSV)

    taidl_by_xla = defaultdict(list)
    for row in join_rows:
        taidl_by_xla[row['matched_xla_hlo_op']].append(row)

    op_rows = []
    class_examples = defaultdict(list)
    class_taidl_ops = defaultdict(list)

    for row in xla_rows:
        op = row['xla_hlo_op']
        klass = XLA_TO_CLASS.get(op, 'uncategorized')
        class_examples[klass].append(op)

        taidl_matches = taidl_by_xla.get(op, [])
        taidl_ops = sorted({m['taidl_primitive'] for m in taidl_matches})
        taidl_statuses = sorted({m['taidl_status'] for m in taidl_matches})
        for t in taidl_ops:
            if t not in class_taidl_ops[klass]:
                class_taidl_ops[klass].append(t)

        op_rows.append({
            'xla_hlo_op': op,
            'description': row['description'],
            'hardware_abstraction_class': klass,
            'class_description': CLASS_INFO[klass]['description'],
            'hardware_meaning': CLASS_INFO[klass]['hardware_meaning'],
            'in_taidl': bool(taidl_ops),
            'taidl_primitives': '|'.join(taidl_ops),
            'taidl_statuses': '|'.join(taidl_statuses),
            'source_url': row['source_url'],
            'notes': row['notes'],
        })

    class_rows = []
    for klass, info in CLASS_INFO.items():
        examples = class_examples.get(klass, [])
        taidl_ops = class_taidl_ops.get(klass, [])
        class_rows.append({
            'hardware_abstraction_class': klass,
            'class_description': info['description'],
            'hardware_meaning': info['hardware_meaning'],
            'example_xla_hlo_ops': '|'.join(examples[:12]),
            'num_xla_ops_in_class': len(examples),
            'taidl_primitives_in_class': '|'.join(taidl_ops),
            'num_taidl_primitives_in_class': len(taidl_ops),
            'coverage_note': 'Based on current XLA op reference CSV (111 rows) and current TAIDL grammar/join sheet.',
        })

    class_rows.sort(key=lambda r: r['hardware_abstraction_class'])
    op_rows.sort(key=lambda r: (r['hardware_abstraction_class'], r['xla_hlo_op']))

    write_csv(CLASS_CSV,
        ['hardware_abstraction_class','class_description','hardware_meaning','example_xla_hlo_ops','num_xla_ops_in_class','taidl_primitives_in_class','num_taidl_primitives_in_class','coverage_note'],
        class_rows)

    write_csv(OP_CSV,
        ['xla_hlo_op','description','hardware_abstraction_class','class_description','hardware_meaning','in_taidl','taidl_primitives','taidl_statuses','source_url','notes'],
        op_rows)

    print(CLASS_CSV)
    print(OP_CSV)

if __name__ == '__main__':
    main()
