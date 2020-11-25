import matplotlib.pyplot as plt
from graphviz import Digraph


g = Digraph(
    format='pdf',
    edge_attr=dict(fontsize='20', fontname="times"),
    node_attr=dict(style='filled', shape='circle', align='center', fontsize='20', height='0.5', width='0.5',
                   penwidth='2', fontname="times"),
    engine='dot'
    )

g.node('0', fillcolor='darkseagreen2')
g.node('1', fillcolor='darkseagreen2')
g.node('2', fillcolor='darkseagreen2')
g.node('3', fillcolor='darkseagreen2')

g.edge('0', '1', 'zeroize(0)', fillcolor="lightblue", style='dashed')
g.edge('0', '2', '1×1 conv(1)', fillcolor="palegoldenrod")
g.edge('0', '3', 'skip-con(2)', fillcolor="palegoldenrod")
g.edge('1', '2', '3×3 conv(3)', fillcolor="palegoldenrod")
g.edge('1', '3', 'avg-pool(4)', fillcolor="palegoldenrod")
g.edge('2', '3', '3×3 conv(5)', fillcolor="palegoldenrod")

g.render(filename='nas201_before', directory='./graph/', view=False, cleanup=True, format='png')

g = Digraph(
    format='pdf',
    edge_attr=dict(fontsize='20', fontname="times"),
    node_attr=dict(style='filled', shape='circle', align='center', fontsize='20', height='0.5', width='0.5',
                   penwidth='2', fontname="times"),
    engine='dot'
    )

g.node('0', fillcolor='darkseagreen2')
g.node('1', fillcolor='darkseagreen2')
g.node('2', fillcolor='darkseagreen2')
g.node('3', fillcolor='darkseagreen2')

g.edge('0', '1', 'zeroize(0)', fillcolor="lightblue", style='dashed')
g.edge('0', '2', '1×1 conv(1)', fillcolor="palegoldenrod")
g.edge('0', '3', 'skip-con(2)', fillcolor="palegoldenrod")
g.edge('1', '2', 'zeroize(3)', fillcolor="lightblue", style='dashed')
g.edge('1', '3', 'zeroize(4)', fillcolor="lightblue", style='dashed')
g.edge('2', '3', '3×3 conv(5)', fillcolor="palegoldenrod")

g.render(filename='nas201_after', directory='./graph/', view=False, cleanup=True, format='png')
