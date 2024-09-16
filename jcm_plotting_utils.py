import numpy as np

import matplotlib
import matplotlib.patches
import matplotlib.collections

matplotlib.rcParams['font.family'] = ['serif']
matplotlib.rcParams['font.serif'] = ['Arial']


def load_grid(filename):
    with open(filename) as fd:
        # first a block with metadata
        while True:
            l = fd.readline()
            if l == '*/\n':  # end of header
                break
            elif l.startswith('<I>NPoints'):
                _, npoints = l.strip().split('=')
                npoints = int(npoints)
            elif l.startswith('<I>NQuadrilaterals'):
                _, nquads = l.strip().split('=')
                nquads = int(nquads)
            elif l.startswith('<I>NTriangles'):
                _, ntr = l.strip().split('=')
                ntr = int(ntr)

        points = np.empty((npoints, 2), dtype=np.float64)   # x, y
        triangles = np.empty((ntr, 3, 2), dtype=np.float64) # p1, p2, p3
        triangles_id = np.empty(ntr, dtype=np.uint)
        quads = np.empty((nquads, 4, 2), dtype=np.float64)  # p1, p2, p3, p4
        quads_id = np.empty(nquads, dtype=np.uint)

        if not fd.readline().startswith('# Points'):
            raise ValueError('Points section not found')

        for i in range(npoints):
            n = int(fd.readline())
            if i+1 != n:
                raise ValueError('Unexpected point')
            x = float(fd.readline())
            y = float(fd.readline())
            points[i] = (x, y)

        if not fd.readline().startswith('# Triangles'):
            raise ValueError('Triangles section not found')

        for i in range(ntr):
            for p in range(3):
                pid = int(fd.readline()) - 1
                triangles[i][p] = points[pid]
            triangles_id[i] = int(fd.readline())

        if not fd.readline().startswith('# Quadrilaterals'):
            raise ValueError('Quadrilaterals section not found')

        for i in range(nquads):
            for p in range(4):
                pid = int(fd.readline()) - 1
                quads[i][p] = points[pid]
            quads_id[i] = int(fd.readline())

        # we do not care for the rest

    return (triangles, triangles_id), (quads, quads_id)

def load_grid_binary(filename):
    with open(filename, 'rb') as fd:
        # first a block with metadata
        while True:
            l = fd.readline()
            if l == b'*/\r\n':  # end of header
                break
            elif l.startswith(b'<I>NPoints'):
                _, npoints = l.strip().split(b'=')
                npoints = int(npoints)
            elif l.startswith(b'<I>NQuadrilaterals'):
                _, nquads = l.strip().split(b'=')
                nquads = int(nquads)
            elif l.startswith(b'<I>NTriangles'):
                _, ntr = l.strip().split(b'=')
                ntr = int(ntr)

        points = np.empty((npoints, 2), dtype=np.float64)   # x, y
        triangles = np.empty((ntr, 3, 2), dtype=np.float64) # p1, p2, p3
        triangles_id = np.empty(ntr, dtype=np.uint)
        quads = np.empty((nquads, 4, 2), dtype=np.float64)  # p1, p2, p3, p4
        quads_id = np.empty(nquads, dtype=np.uint)

        if not fd.readline().startswith(b'# Points'):
            print(fd.readline())
            raise ValueError('Points section not found')

        for i in range(npoints):
            n = int(fd.readline())
            if i+1 != n:
                raise ValueError('Unexpected point')
            x = float(fd.readline())
            y = float(fd.readline())
            points[i] = (x, y)

        if not fd.readline().startswith(b'# Triangles'):
            raise ValueError('Triangles section not found')

        for i in range(ntr):
            for p in range(3):
                pid = int(fd.readline()) - 1
                triangles[i][p] = points[pid]
            triangles_id[i] = int(fd.readline())

        if not fd.readline().startswith(b'# Quadrilaterals'):
            raise ValueError('Quadrilaterals section not found')

        for i in range(nquads):
            for p in range(4):
                pid = int(fd.readline()) - 1
                quads[i][p] = points[pid]
            quads_id[i] = int(fd.readline())

        # we do not care for the rest

    return (triangles, triangles_id), (quads, quads_id)

def plot_grid(filename, *, ax, colors, shift_x=0, shift_y=0, **kwargs):
    (triangles, triangles_id), (quads, quads_id) = load_grid(filename)
    triangles *= 1e9
    quads *= 1e9
    triangles[:, :, 0] += shift_x
    quads[:, :, 0] += shift_x
    triangles[:, :, 1] += shift_y
    quads[:, :, 1] += shift_y
    for i in range(len(triangles_id)):
        ax.add_patch(matplotlib.patches.Polygon(triangles[i], color=colors[triangles_id[i]], **kwargs))

    for i in range(len(quads_id)):
        ax.add_patch(matplotlib.patches.Polygon(quads[i], color=colors[quads_id[i]], **kwargs))

def plot_grid_1(filename, *, ax, facecolors, edgecolors, shift_x=0, shift_y=0, **kwargs):
    (triangles, triangles_id), (quads, quads_id) = load_grid(filename)
    triangles *= 1e9
    quads *= 1e9
    triangles[:, :, 0] += shift_x
    quads[:, :, 0] += shift_x
    triangles[:, :, 1] += shift_y
    quads[:, :, 1] += shift_y
    for i in range(len(triangles_id)):
        ax.add_patch(matplotlib.patches.Polygon(triangles[i],
                                                facecolor=facecolors[triangles_id[i]],
                                                edgecolor=edgecolors[triangles_id[i]],
                                                **kwargs))

    for i in range(len(quads_id)):
        ax.add_patch(matplotlib.patches.Polygon(quads[i],
                                                facecolor=facecolors[quads_id[i]],
                                                edgecolor=edgecolors[triangles_id[i]],
                                                **kwargs))
    

