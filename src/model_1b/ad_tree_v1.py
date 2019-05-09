import time
from pprint import pprint
import numpy as np


class ADNode(object):

    def __init__(self, count):
        self.count = count
        self.vary_nodes = {}

    def get(self, i):
        return self.vary_nodes.get(i)

class VaryNode(object):

    def __init__(self):
        self.mcv = None
        self.adnodes = {}

    def get(self, a):
        return self.adnodes.get(a)


# In[8]:


class ADTree(object):

    def __init__(self, dimensions, records):
        self.dimensions = dimensions
        self.records = self._translate_records(records)
        self.root = self.make_adnode(0, self.records)
        self._node_id = 0

    def get_record(self, i):
        return self.records[i]

    def _translate_records(self, records):
        newrecords = []

        for r in records:
            newr = []
            for k in self.dimensions:
                val = r.get(k, '')
                if val is None:
                    val = ''
                newr.append(val)
            newrecords.append(newr)

        return newrecords

    def make_adnode(self, i, records):
        n = ADNode(len(records))

        for j in range(i, len(self.dimensions)):
            n.vary_nodes[j] = self.make_varynode(j, records)

        return n

    def make_varynode(self, i, records):
        v = VaryNode()
        child_records = {}

        for j in records:
            if j[i] not in child_records:
                child_records[j[i]] = [j]
            else:
                child_records[j[i]].append(j)

        v.mcv = max(child_records, key=lambda k: len(child_records[k]))

        for k, rec in child_records.items():
            if k == v.mcv:
                continue
            else:
                v.adnodes[k] = self.make_adnode(i + 1, rec)

        return v

    def count(self, **kwargs):
        q = []

        for k in self.dimensions:
            q.append(kwargs.get(k, None))

        query = []

        for i, n in enumerate(q):
            if n is None:
                continue
            else:
                query.append((i, n))

        return self._count(self.root, query)

    def _count(self, adnode, query):
        if not query:
            return adnode.count

        vn = adnode.get(query[0][0])

        if query[0][1] == vn.mcv:
            c = self._count(adnode, query[1:])

            for k, ad in vn.adnodes.items():
                c -= self._count(ad, query[1:])

            return c

        ad = vn.get(query[0][1])

        if not ad:
            return 0
        return self._count(ad, query[1:])

    def table(self, *q):
        query = []

        for i, k in enumerate(self.dimensions):
            for a in q:
                if a == k:
                    query.append(i)

        return self._make_table(self.root, query)

    def _make_table(self, adnode, query):
        if not query:
            return [[adnode.count]]

        vn = adnode.get(query[0])
        rows = []
        total = 0

        for k, ad in vn.adnodes.items():
            for r in self._make_table(ad, query[1:]):
                rows.append([k] + r)
                total += r[-1]

        mcv_rows = []

        for m in self._make_table(adnode, query[1:]):
            if len(m) == 1:
                mcv_rows.append([vn.mcv, m[-1] - total])
            else:
                match = False
                for s in rows:
                    if m[:-1] == s[1:-1]:
                        m[-1] -= s[-1]
                mcv_rows.append([vn.mcv] + m)

        return rows + [r for r in mcv_rows if r[-1] > 0]

    def query(self, *q, **c):
        query = []
        conditions = []

        for i, k in enumerate(self.dimensions):
            v = c.get(k)
            if v is not None:
                conditions.append((i, v))
            else:
                for a in q:
                    if a == k:
                        query.append(i)

        return self._query(self.root, query, conditions)

    def _query(self, adnode, query, conditions):
        if not conditions:
            return self._make_table(adnode, query)

        if not query:
            return [[self._count(adnode, conditions)]]

        if query[0] < conditions[0][0]:
            vn = adnode.get(query[0])
            rows = []
            total = 0

            for k, ad in vn.adnodes.items():
                for r in self._query(ad, query[1:], conditions):
                    if r[-1] > 0:
                        rows.append([k] + r)
                        total += r[-1]

            mcv_rows = []

            for m in self._query(adnode, query[1:], conditions):
                if m[-1] > 0:
                    if len(m) == 1:
                        mcv_rows.append([vn.mcv, m[-1] - total])
                    else:
                        for s in rows:
                            if m[:-1] == s[1:-1]:
                                m[-1] -= s[-1]
                        mcv_rows.append([vn.mcv] + m)

            return rows + [r for r in mcv_rows if r[-1] > 0]
        else:
            vn = adnode.get(conditions[0][0])

            if conditions[0][1] == vn.mcv:
                rows = []
                total = 0

                for k, ad in vn.adnodes.items():
                    for r in self._query(ad, query, conditions[1:]):
                        if r[-1] > 0:
                            rows.append(r)
                            total += r[-1]

                mcv_rows = []

                for m in self._query(adnode, query, conditions[1:]):
                    if m[-1] > 0:
                        if len(m) == 1:
                            mcv_rows.append([vn.mcv, m[-1] - total])
                        else:
                            for s in rows:
                                if m[:-1] == s[:-1]:
                                    m[-1] -= s[-1]
                            mcv_rows.append(m)

                return [r for r in mcv_rows if r[-1] > 0]

            ad = vn.get(conditions[0][1])

            if not ad:
                return []

            return self._query(ad, query, conditions[1:])

    def _print(self, val, adnode, depth):
        print(depth * "  ") + "%s (%s)" % (val, adnode.count)
        depth += 1
        for i, n in adnode.vary_nodes.items():
            print(depth * "\t") + "a%s (mcv=%s)" % (i, n.mcv)
            for k, an in n.adnodes.items():
                if an is not None:
                    self._print(k, an, depth + 1)

    def _node_id_iter(self):
        i = 0
        while 1:
            yield str(i)
            i += 1

    def size(self):
        size = {
            'dimensions': len(self.dimensions),
            'records': len(self.records),
            'record_pointers': 0,
            'adnodes': 0,
            'varynodes': 0
        }

        self._size(self.root, size)
        return size

    def _size(self, adnode, size):
        size['adnodes'] += 1
        size['record_pointers'] += adnode.count

        for i, vn in adnode.vary_nodes.items():
            size['varynodes'] += 1
            for k, an in vn.adnodes.items():
                if an:
                    self._size(an, size)


# ----------- Lets do some software engineering --------------#
# ----------- Adapter (Wrapper) - the hero we need! not the one we deserve!! -------- #

class ADT:
    def __init__(self):
        return

    # input : data (np array )
    def setup(self, records):
        # inherent weakness
        attr_fake_name = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        i = 1
        for _ in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            attr_fake_name.append( _ + str(i))

        self.name2domainID = {attr_fake_name[i]: i for i in range(records.shape[1])}
        self.domainID2name = {i: attr_fake_name[i] for i in range(records.shape[1])}
        self.attr_names = list(self.name2domainID.keys())
        print(self.name2domainID)
        records = [dict([
            (k, v)
            for k, v in zip(self.attr_names, r)]
        ) for r in records]

        self.tree = ADTree(self.attr_names, records)
        return

    # query should be of form [ 1:2 , 2:4,...]
    def get_count(self, query):
        _dict = {}
        for domain, value in query.items():
            # create dict
            _dict[self.domainID2name[domain]] = value
        result = self.tree.count(**_dict)
        return result


# ------------------------- #

def test():
    records = [
        (1, 2, 4),
        (2, 2, 4),
        (1, 3, 2),
        (1, 2, 1),
        (5, 3, 4),
        (1, 2, 4),
        (1, 4, 5),
        (7, 2, 0),
        (3, 4, 1),
        (1, 2, 1)
    ]
    records = np.array(records)
    print(records)
    tree = ADT()
    tree.setup(records)
    res = tree.get_count({1:2})
    print(res)



